"""
Based on Pong's SAC implementation

https://github.com/vitchyr/rlkit
"""

import math
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

from collections import OrderedDict
from itertools import chain

import robolearn.torch.pytorch_util as ptu
from robolearn.core import logger
from robolearn.core import eval_util
from robolearn.utils.samplers import InPlacePathSampler
from robolearn.torch.rl_algos.torch_incremental_rl_algorithm \
    import TorchIncrementalRLAlgorithm
from robolearn.policies.make_deterministic import MakeDeterministic
from robolearn.torch.policies import WeightedMultiPolicySelector
from robolearn.utils.data_management.normalizer import RunningNormalizer

from tensorboardX import SummaryWriter


class IUWeightedMultiDDPG(TorchIncrementalRLAlgorithm):
    """Intentional-Unintentional Deep Deterministic Policy Gradient (IU-DDPG)
    with MultiHead Networks.

    """
    def __init__(
            self,
            env,
            policy,
            exploration_policy,
            u_qf,

            replay_buffer,
            batch_size=1024,
            normalize_obs=True,

            i_qf=None,
            eval_env=None,

            reparameterize=True,
            action_prior='uniform',

            i_policy_lr=1e-4,
            u_policies_lr=1e-4,
            u_mixing_lr=1e-4,

            i_qf_lr=1e-3,
            u_qf_lr=1e-3,

            i_policy_pre_activation_weight=0.,
            i_policy_mixing_coeff_weight=1e-3,
            u_policy_pre_activation_weight=None,

            i_policy_weight_decay=0.,
            u_policy_weight_decay=0.,
            i_q_weight_decay=0.,
            u_q_weight_decay=0.,

            optimizer_class=optim.Adam,
            # optimizer_class=optim.SGD,
            amsgrad=True,

            i_soft_target_tau=1e-2,
            u_soft_target_tau=1e-2,
            i_target_update_interval=1,
            u_target_update_interval=1,

            u_reward_scales=None,

            min_q_value=-np.inf,
            max_q_value=np.inf,

            residual_gradient_weight=0,

            eval_with_target_policy=False,
            save_replay_buffer=False,
            log_tensorboard=False,
            **kwargs
    ):

        # ###### #
        # Models #
        # ###### #

        # Deterministic Policies
        self._policy = policy
        self._target_policy = policy.copy()

        # Exploration Policy
        self._exploration_policy = exploration_policy

        # Evaluation Policy
        if eval_with_target_policy:
            eval_policy = self._target_policy
        else:
            eval_policy = self._policy

        # Observation Normalizer
        if normalize_obs:
            self._obs_normalizer = RunningNormalizer(shape=env.obs_dim)
        else:
            self._obs_normalizer = None

        TorchIncrementalRLAlgorithm.__init__(
            self,
            env=env,
            exploration_policy=self._exploration_policy,
            eval_env=eval_env,
            eval_policy=eval_policy,
            obs_normalizer=self._obs_normalizer,
            **kwargs
        )

        # Number of Unintentional Tasks (Composable Tasks)
        self._n_unintentional = self._policy.n_heads

        # Important algorithm hyperparameters
        self._reparameterize = reparameterize
        assert self._reparameterize == self._policy.reparameterize
        self._action_prior = action_prior

        # Intentional (Main Task) Q-function
        self._i_qf = i_qf
        self._i_target_qf = i_qf.copy()

        # Unintentional (Composable Tasks) Q-function
        self._u_qf = u_qf
        self._u_target_qf = u_qf.copy()

        self._min_q_value = min_q_value
        self._max_q_value = max_q_value
        self._residual_gradient_weight = residual_gradient_weight

        # Soft-update rate for target Vfs
        self._i_soft_target_tau = i_soft_target_tau
        self._u_soft_target_tau = u_soft_target_tau
        self._i_target_update_interval = i_target_update_interval
        self._u_target_update_interval = u_target_update_interval

        # Unintentional Reward Scales
        if u_reward_scales is None:
            reward_scale = kwargs['reward_scale']
            u_reward_scales = [reward_scale
                               for _ in range(self._n_unintentional)]
        self._u_reward_scales = ptu.FloatTensor(u_reward_scales)

        # Replay Buffer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.save_replay_buffer = save_replay_buffer

        # ########## #
        # Optimizers #
        # ########## #

        # Q-function and V-function Optimization Criteria
        self._u_qf_criterion = nn.MSELoss()
        self._i_qf_criterion = nn.MSELoss()

        # Q-function optimizers
        self._u_qf_optimizer = optimizer_class(
            self._u_qf.parameters(),
            lr=u_qf_lr,
            amsgrad=amsgrad,
            weight_decay=u_q_weight_decay
        )
        self._i_qf_optimizer = optimizer_class(
            self._i_qf.parameters(),
            lr=i_qf_lr,
            amsgrad=amsgrad,
            weight_decay=i_q_weight_decay
        )

        # Policy optimizer
        # self._policy_optimizer = optimizer_class([
        #     {'params': self._policy.shared_parameters(),
        #      'lr': i_policy_lr},
        #     {'params': self._policy.policies_parameters(),
        #      'lr': i_policy_lr},
        #     {'params': self._policy.mixing_parameters(),
        #      'lr': i_policy_lr},
        # ])
        self._policy_optimizer = optimizer_class(
            self._policy.parameters(),
            lr=i_policy_lr,
            amsgrad=amsgrad,
            weight_decay=i_policy_weight_decay,
        )
        self._mixing_optimizer = optimizer_class(
            chain(self._policy.shared_parameters(),
                  self._policy.mixing_parameters()),
            lr=u_mixing_lr,
            amsgrad=amsgrad,
            weight_decay=i_policy_weight_decay
        )
        self._policies_optimizer = optimizer_class(
            chain(self._policy.shared_parameters(),
                  self._policy.policies_parameters()),
            lr=u_policies_lr,
            amsgrad=amsgrad,
            weight_decay=u_policy_weight_decay
        )

        # Policy regularization coefficients (weights)
        self._i_policy_pre_activation_weight = i_policy_pre_activation_weight
        self._i_policy_mixing_coeff_weight = i_policy_mixing_coeff_weight

        if u_policy_pre_activation_weight is None:
            u_policy_pre_activation_weight = \
                [i_policy_pre_activation_weight
                 for _ in range(self._n_unintentional)]
        self._u_policy_pre_activation_weight = \
            ptu.FloatTensor(u_policy_pre_activation_weight)

        # Evaluation Sampler (One for each unintentional)
        self.eval_u_samplers = [
            InPlacePathSampler(
                env=env,
                policy=WeightedMultiPolicySelector(eval_policy, idx),
                total_samples=self.num_steps_per_eval,
                max_path_length=self.max_path_length,
                deterministic=None,
            )
            for idx in range(self._n_unintentional)
        ]

        # Useful Variables for logging
        self.logging_raw_pol_loss = np.zeros((self.num_train_steps_per_epoch,
                                              self._n_unintentional + 1))
        self.logging_pol_loss = np.zeros((self.num_train_steps_per_epoch,
                                          self._n_unintentional + 1))
        self.logging_qf_loss = np.zeros((self.num_train_steps_per_epoch,
                                         self._n_unintentional + 1))
        self.logging_rewards = np.zeros((self.num_train_steps_per_epoch,
                                         self._n_unintentional + 1))
        self.logging_actions = np.zeros((self.num_train_steps_per_epoch,
                                         self._n_unintentional + 1,
                                         self.env.action_dim))
        self.logging_mixing_coeff = np.zeros((self.num_train_steps_per_epoch,
                                              self._n_unintentional,
                                              self.env.action_dim))

        self._log_tensorboard = log_tensorboard
        self._summary_writer = SummaryWriter(log_dir=logger.get_snapshot_dir())

    def pretrain(self, n_pretrain_samples):
        # We do not require any pretrain (I think...)
        observation = self.env.reset()
        for ii in range(n_pretrain_samples):
            action = self.env.action_space.sample()
            # Interact with environment
            next_ob, raw_reward, terminal, env_info = (
                self.env.step(action)
            )
            agent_info = None

            # Increase counter
            self._n_env_steps_total += 1
            # Create np.array of obtained terminal and reward
            reward = raw_reward * self.reward_scale
            terminal = np.array([terminal])
            reward = np.array([reward])
            # Add to replay buffer
            self.replay_buffer.add_sample(
                observation=observation,
                action=action,
                reward=reward,
                terminal=terminal,
                next_observation=next_ob,
                agent_info=agent_info,
                env_info=env_info,
            )
            observation = next_ob

            if self._obs_normalizer is not None:
                self._obs_normalizer.update(np.array([observation]))

            if terminal:
                self.env.reset()

    def _do_training(self):
        # Get batch of samples
        batch = self.get_batch()

        # Get common data from batch
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        # Get the idx for logging
        step_idx = self._n_epoch_train_steps

        # ###################### #
        # Get Policy Data: s_t+1 #
        # ###################### #
        i_next_actions, policy_info = \
            self._target_policy(next_obs, pol_idx=None, optimize_policies=False)

        u_next_actions = policy_info['pol_actions']

        # ########################## #
        # Unintentional Critics Step #
        # ########################## #
        # Speed up computation by not backpropping these gradients
        u_next_actions.detach()

        # Get unintentional rewards and terminals
        u_rewards = \
            (batch['reward_vectors'] * self._u_reward_scales).unsqueeze(-1)
        u_terminals = (batch['terminal_vectors']).unsqueeze(-1)

        # Get unintentional target Q Values: Q(s_t+1, a_t+1)
        u_target_q_values = [
            self._u_target_qf(next_obs, u_next_actions[:, uu, :])[0][uu]
            for uu in range(self._n_unintentional)
        ]
        u_target_q_values = torch.cat([qq.unsqueeze(1)
                                       for qq in u_target_q_values],
                                      dim=1)

        # Calculate Unintentional QF Losses (Bellman Eq.)
        u_q_target = u_rewards + (1. - u_terminals) * self.discount * u_target_q_values
        u_q_target = u_q_target.detach()
        u_q_target = torch.clamp(u_q_target, self._min_q_value, self._max_q_value)
        # Hack for ICLR rebuttal
        if hasattr(self, 'reward_type') and self.reward_type == 'indicator':
            # TODO: CHECK THIS
            u_q_target = \
                torch.clamp(u_q_target, -self.reward_scale/(1-self.discount), 0)

        # u_q_pred = [self._u_qf(obs, actions)[0][uu]
        #             for uu in range(self._n_unintentional)]
        u_q_pred = self._u_qf(obs, actions)[0]
        u_q_pred = torch.cat([qq.unsqueeze(1) for qq in u_q_pred], dim=1)

        u_qf_loss = \
            0.5*torch.mean((u_q_pred - u_q_target)**2, dim=0).squeeze(-1)
        total_u_qf_loss = torch.sum(u_qf_loss)

        if self._residual_gradient_weight > 0:
            raise NotImplementedError

        # Update Unintentional Q-value model parameters
        self._u_qf_optimizer.zero_grad()
        total_u_qf_loss.backward()
        self._u_qf_optimizer.step()

        # ####################### #
        # Intentional Critic Step #
        # ####################### #
        # Speed up computation by not backpropping these gradients
        i_next_actions.detach()

        # Get Intentional rewards and terminals
        i_rewards = batch['rewards']# * self.reward_scale
        i_terminals = batch['terminals']

        # Get intentional target Q Values: Q(s_t+1, a_t+1)
        i_target_q_values = self._i_target_qf(next_obs, i_next_actions)[0]

        # Calculate Intentional QF Losses (Bellman Eq.)
        i_q_target = i_rewards + (1. - i_terminals) * self.discount * i_target_q_values
        i_q_target = i_q_target.detach()
        i_q_target = torch.clamp(i_q_target, self._min_q_value, self._max_q_value)
        # Hack for ICLR rebuttal
        if hasattr(self, 'reward_type') and self.reward_type == 'indicator':
            # TODO: CHECK THIS
            i_q_target = \
                torch.clamp(i_q_target, -self.reward_scale/(1-self.discount), 0)

        i_q_pred = self._i_qf(obs, actions)[0]

        i_qf_loss = \
            0.5*torch.mean((i_q_pred - i_q_target)**2, dim=0)

        if self._residual_gradient_weight > 0:
            raise NotImplementedError

        # Update Intentional Q-value model parameters
        self._i_qf_optimizer.zero_grad()
        i_qf_loss.backward()
        self._i_qf_optimizer.step()

        # #################### #
        # Get Policy Data: s_t #
        # #################### #
        i_policy_actions, policy_info = \
            self._policy(obs, pol_idx=None, optimize_policies=False)
        i_pre_tanh_value = policy_info['pre_tanh_value']
        mixing_coeff = policy_info['mixing_coeff']

        u_policy_actions = policy_info['pol_actions']
        u_pre_tanh_value = policy_info['pol_pre_tanh_values']

        # ######################## #
        # Unintentional Actor Step #
        # ######################## #
        if self._action_prior == 'normal':
            raise NotImplementedError
        else:
            u_policy_prior_log_probs = 0.0  # Uniform prior

        # Get Unintentional Q1(s_t, a_t)
        u_q_output = [self._u_qf(obs, u_policy_actions[:, uu, :])[0][uu]
                      for uu in range(self._n_unintentional)]
        u_q_output = torch.cat([qq.unsqueeze(1) for qq in u_q_output], dim=1)

        # Get Unintentional Policies KL loss: - (E_a[Q(s_t, a_t_) - H(.)])
        u_raw_policy_loss = -u_q_output.mean(dim=0).squeeze(-1)

        # Get Unintentional Policies regularization loss
        u_pre_activation_reg_loss = \
            self._u_policy_pre_activation_weight * \
            (u_pre_tanh_value**2).sum(dim=-1).mean(dim=0).mean(dim=-1)

        # Get Unintentional Policies Total loss
        u_policy_loss = (u_raw_policy_loss + u_pre_activation_reg_loss)
        total_u_policy_loss = torch.sum(u_policy_loss)

        # Update Unintentional Policies
        self._policy_optimizer.zero_grad()
        # accum_u_policy_loss.backward()
        # self._policy_optimizer.step()
        self._policies_optimizer.zero_grad()
        total_u_policy_loss.backward()
        self._policies_optimizer.step()

        # ####################### #
        # Intentional Actor  Step #
        # ####################### #
        if self._action_prior == 'normal':
            raise NotImplementedError
        else:
            i_policy_prior_log_probs = 0.0  # Uniform prior

        # Get Intentional Q1(s_t, a_t)
        i_q_output = self._i_qf(obs, i_policy_actions)[0]

        # Get Intentional KL loss: - (E_a[Q(s_t, a_t_) - H(.)])
        i_raw_policy_loss = -i_q_output.mean()

        # Get Intentional Policy regularization loss
        i_pre_activation_reg_loss = \
            self._i_policy_pre_activation_weight * \
            (i_pre_tanh_value**2).sum(dim=-1).mean()
        mixing_coeff_loss = self._i_policy_mixing_coeff_weight * \
            (mixing_coeff ** 2).sum(dim=-1).mean()  # TODO: CHECK THIS
        i_policy_regu_loss = (i_pre_activation_reg_loss + mixing_coeff_loss)

        # Get Intentional Policy Total loss
        i_policy_loss = (i_raw_policy_loss + i_policy_regu_loss)

        # Update Intentional Policies
        self._policy_optimizer.zero_grad()
        # accum_u_policy_loss.backward()
        # self._policy_optimizer.step()
        self._mixing_optimizer.zero_grad()
        i_policy_loss.backward()
        self._mixing_optimizer.step()

        # Update Unintentional V Target Networks
        if self._n_train_steps_total % self._u_target_update_interval == 0:
            ptu.soft_update_from_to(self._u_qf, self._u_target_qf,
                                    self._u_soft_target_tau)

        # Update Intentional V Target Networks
        if self._n_train_steps_total % self._i_target_update_interval == 0:
            ptu.soft_update_from_to(self._i_qf, self._i_target_qf,
                                    self._i_soft_target_tau)

        # Update Policy Network
        if self._n_train_steps_total % self._i_target_update_interval == 0:
            ptu.soft_update_from_to(self._policy, self._target_policy,
                                    self._i_soft_target_tau)

        # ############### #
        # LOG Useful Data #
        # ############### #
        self.logging_raw_pol_loss[step_idx, :-1] = \
            ptu.get_numpy(u_raw_policy_loss.squeeze(-1))
        self.logging_raw_pol_loss[step_idx, -1] = \
            ptu.get_numpy(i_raw_policy_loss)

        self.logging_pol_loss[step_idx, :-1] = \
            ptu.get_numpy(u_policy_loss.squeeze(-1))
        self.logging_pol_loss[step_idx, -1] = \
            ptu.get_numpy(i_policy_loss)

        self.logging_qf_loss[step_idx, :-1] = \
            ptu.get_numpy(u_qf_loss.squeeze(-1))
        self.logging_qf_loss[step_idx, -1] = ptu.get_numpy(i_qf_loss)

        self.logging_rewards[step_idx, :-1] = \
            ptu.get_numpy(u_rewards.mean(dim=0).squeeze(-1))
        self.logging_rewards[step_idx, -1] = \
            ptu.get_numpy(i_rewards.mean(dim=0).squeeze(-1))

        self.logging_mixing_coeff[step_idx, :, :] = \
            ptu.get_numpy(mixing_coeff.mean(dim=0))

        self.logging_actions[step_idx, :-1, :] = \
            ptu.get_numpy(u_policy_actions.mean(dim=0))
        self.logging_actions[step_idx, -1, :] = \
            ptu.get_numpy(i_policy_actions.mean(dim=0))

        if self._log_tensorboard:
            self._summary_writer.add_scalar('TrainingI/qf_loss',
                                            ptu.get_numpy(i_qf_loss),
                                            self._n_env_steps_total)
            self._summary_writer.add_scalar('TrainingI/avg_reward',
                                            ptu.get_numpy(i_rewards.mean()),
                                            self._n_env_steps_total)
            self._summary_writer.add_scalar('TrainingI/policy_loss',
                                            ptu.get_numpy(i_policy_loss),
                                            self._n_env_steps_total)
            self._summary_writer.add_scalar('TrainingI/q_vals',
                                            ptu.get_numpy(i_q_output.mean()),
                                            self._n_env_steps_total)

    def _do_not_training(self):
        return

    @property
    def torch_models(self):
        networks_list = [
            self._policy,
            self._target_policy,
            self._i_qf,
            self._i_target_qf,
            self._u_qf,
            self._u_target_qf,
        ]
        if self._i_target_qf is not None:
            networks_list.append(self._i_target_qf)

        if self._target_policy is not None:
            networks_list.append(self._target_policy)

        return networks_list

    @staticmethod
    def compute_gae(next_value, rewards, masks, values, gamma=0.99,
                    tau=0.95):
        # FROM: https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def get_epoch_snapshot(self, epoch):
        """
        Stuff to save in file.
        Args:
            epoch:

        Returns:

        """
        if self._epoch_plotter is not None:
            self._epoch_plotter.draw()
            self._epoch_plotter.save_figure(epoch)

        snapshot = TorchIncrementalRLAlgorithm.get_epoch_snapshot(self, epoch)

        snapshot.update(
            policy=self._policy,
            target_policy=self._target_policy,
            exploration_policy=self._exploration_policy,
            qf=self._i_qf,
            target_qf=self._i_target_qf,
            u_qf=self._u_qf,
            target_u_qf=self._u_target_qf,
        )

        if self.env.online_normalization or self.env.normalize_obs:
            snapshot.update(
                obs_mean=self.env.obs_mean,
                obs_var=self.env.obs_var,
            )

        # Observation Normalizer
        snapshot.update(
            obs_normalizer=self._obs_normalizer,
        )

        # Replay Buffer
        if self.save_replay_buffer:
            snapshot.update(
                replay_buffer=self.replay_buffer,
            )

        return snapshot

    def _update_logging_data(self):
        max_step = max(self._n_epoch_train_steps, 1)

        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
        # Unintentional info
        for uu in range(self._n_unintentional):
            self.eval_statistics['[U-%02d] Qf Loss' % uu] = \
                np.nan_to_num(np.mean(self.logging_qf_loss[:max_step, uu]))
            self.eval_statistics['[U-%02d] Raw Policy Loss' % uu] = \
                np.nan_to_num(np.mean(self.logging_raw_pol_loss[:max_step, uu]))
            self.eval_statistics['[U-%02d] Policy Loss' % uu] = \
                np.nan_to_num(np.mean(self.logging_pol_loss[:max_step, uu]))
            self.eval_statistics['[U-%02d] Rewards' % uu] = \
                np.nan_to_num(np.mean(self.logging_rewards[:max_step, uu]))
            self.eval_statistics['[U-%02d] Mixing Weights' % uu] = \
                np.nan_to_num(np.mean(self.logging_mixing_coeff[:max_step, uu]))

        # Intentional info
        self.eval_statistics['[I] Qf Loss'] = \
            np.nan_to_num(np.mean(self.logging_qf_loss[:max_step, -1]))
        self.eval_statistics['[I] Raw Policy Loss'] = \
            np.nan_to_num(np.mean(self.logging_raw_pol_loss[:max_step, -1]))
        self.eval_statistics['[I] Policy Loss'] = \
            np.nan_to_num(np.mean(self.logging_pol_loss[:max_step, -1]))
        self.eval_statistics['[I] Rewards'] = \
            np.nan_to_num(np.mean(self.logging_rewards[:max_step, -1]))

    def evaluate(self, epoch):
        statistics = OrderedDict()
        self._update_logging_data()
        statistics.update(self.eval_statistics)
        self.eval_statistics = None

        # Interaction Paths for each unintentional policy
        test_paths = [None for _ in range(self._n_unintentional)]
        for unint_idx in range(self._n_unintentional):
            logger.log("[U-%02d] Collecting samples for evaluation" % unint_idx)
            test_paths[unint_idx] = \
                self.eval_u_samplers[unint_idx].obtain_samples()

            statistics.update(eval_util.get_generic_path_information(
                test_paths[unint_idx], stat_prefix="[U-%02d] Test" % unint_idx,
            ))

            average_rewards = \
                eval_util.get_average_multigoal_rewards(test_paths[unint_idx],
                                                        unint_idx)
            avg_txt = '[U-%02d] Test AverageReward' % unint_idx
            statistics[avg_txt] = average_rewards * \
                ptu.get_numpy(self._u_reward_scales[unint_idx])

            average_returns = \
                eval_util.get_average_multigoal_returns(test_paths[unint_idx],
                                                        unint_idx)
            avg_txt = '[U-%02d] Test AverageReturn' % unint_idx
            statistics[avg_txt] = average_returns * \
                ptu.get_numpy(self._u_reward_scales[unint_idx])

            if self._log_tensorboard:
                self._summary_writer.add_scalar(
                    'EvaluationU%02d/avg_return' % unint_idx,
                    statistics['[U-%02d] Test AverageReturn' % unint_idx],
                    self._n_epochs
                )

                self._summary_writer.add_scalar(
                    'EvaluationU%02d/avg_reward' % unint_idx,
                    statistics['[U-%02d] Test AverageReward' % unint_idx],
                    self._n_epochs
                )

        # Interaction Paths for the intentional policy
        logger.log("[I] Collecting samples for evaluation")
        i_test_path = self.eval_sampler.obtain_samples()
        statistics.update(eval_util.get_generic_path_information(
            i_test_path, stat_prefix="[I] Test",
        ))
        average_return = eval_util.get_average_returns(i_test_path)
        statistics['[I] Test AverageReturn'] = average_return * self.reward_scale

        if self._exploration_paths:
            statistics.update(eval_util.get_generic_path_information(
                self._exploration_paths, stat_prefix="Exploration",
            ))
        else:
            statistics.update(eval_util.get_generic_path_information(
                i_test_path, stat_prefix="Exploration",
            ))

        if self._log_tensorboard:
            self._summary_writer.add_scalar('EvaluationI/avg_return',
                                            statistics['[I] Test AverageReturn'],
                                            self._n_epochs)

            self._summary_writer.add_scalar(
                'EvaluationI/avg_reward',
                statistics['[I] Test Rewards Mean'] * self.reward_scale,
                self._n_epochs
            )

        if hasattr(self.env, "log_diagnostics"):
            pass
            # # TODO: CHECK ENV LOG_DIAGNOSTICS
            # print('%03d' % self._n_epochs,
            #       'TODO: WE NEED LOG_DIAGNOSTICS IN ENV')
            # # self.env.log_diagnostics(test_paths[demon])

        # Record the data
        for key, value in statistics.items():
            logger.record_tabular(key, value)

        for u_idx in range(self._n_unintentional):
            if self.render_eval_paths:
                # TODO: CHECK ENV RENDER_PATHS
                print('TODO: RENDER_PATHS')
                pass
                # self.env.render_paths(test_paths[demon])

        # Epoch Plotter
        if self._epoch_plotter is not None:
            self._epoch_plotter.draw()

        # RESET
        self.logging_raw_pol_loss.fill(0)
        self.logging_pol_loss.fill(0)
        self.logging_qf_loss.fill(0)
        self.logging_rewards.fill(0)
        self.logging_actions.fill(0)
        self.logging_mixing_coeff.fill(0)

    def get_batch(self):
        batch = self.replay_buffer.random_batch(self.batch_size)

        if self._obs_normalizer is not None:
            batch['observations'] = \
                self._obs_normalizer.normalize(batch['observations'])
            batch['next_observations'] = \
                self._obs_normalizer.normalize(batch['next_observations'])

        return ptu.np_to_pytorch_batch(batch)

    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
    ):
        """
        Implement anything that needs to happen after every step
        :return:
        """
        # Add to replay buffer
        self.replay_buffer.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            agent_info=agent_info,
            env_info=env_info,
        )

        # Update observation normalizer (if applicable)
        if self._obs_normalizer is not None:
            self._obs_normalizer.update(np.array([observation]))

        TorchIncrementalRLAlgorithm._handle_step(
            self,
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            agent_info=agent_info,
            env_info=env_info,
        )

    def _handle_rollout_ending(self):
        """
        Implement anything that needs to happen after every rollout.
        """

        self.replay_buffer.terminate_episode()

        TorchIncrementalRLAlgorithm._handle_rollout_ending(self)

