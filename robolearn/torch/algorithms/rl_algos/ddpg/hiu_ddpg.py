"""
Based on Pong's SAC implementation

https://github.com/vitchyr/rlkit
"""

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

from collections import OrderedDict
from itertools import chain

import robolearn.torch.utils.pytorch_util as ptu
from robolearn.utils.logging import logger
from robolearn.utils import eval_util
from robolearn.utils.samplers import InPlacePathSampler

from robolearn.algorithms.rl_algos import IncrementalRLAlgorithm
from robolearn.torch.algorithms.torch_algorithm import TorchAlgorithm

from robolearn.torch.policies import WeightedMultiPolicySelector
from robolearn.utils.data_management.normalizer import RunningNormalizer

from tensorboardX import SummaryWriter


class HIUDDPG(IncrementalRLAlgorithm, TorchAlgorithm):
    """
    Hierarchical Intentional-Unintentional Deep Deterministic Policy Gradient
    (HIU-DDPG). Incremental Version.
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

            # optimizer='adam',
            optimizer='rmsprop',
            # optimizer='sgd',
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

        IncrementalRLAlgorithm.__init__(
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

        # Important algorithm hyperparameters
        self._reparameterize = reparameterize
        assert self._reparameterize == self._policy.reparameterize
        self._action_prior = action_prior

        # Intentional (Main Task) Q-function
        self._i_qf = i_qf
        self._i_target_qf = i_qf.copy()

        # Unintentional (Composable Tasks) Q-functions
        self._u_qf = u_qf
        self._u_target_qf = u_qf.copy()

        self._min_q_value = min_q_value
        self._max_q_value = max_q_value
        self._residual_gradient_weight = residual_gradient_weight

        # Soft-update rate for target V-functions
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
        if optimizer.lower() == 'adam':
            optimizer_class = optim.Adam
            optimizer_params = dict(
                amsgrad=amsgrad,
            )
        elif optimizer.lower() == 'rmsprop':
            optimizer_class = optim.RMSprop
            optimizer_params = dict(

            )
        else:
            raise ValueError('Wrong optimizer')

        # Q-function and V-function Optimization Criteria
        self._u_qf_criterion = nn.MSELoss()
        self._i_qf_criterion = nn.MSELoss()

        # Q-function optimizers
        self._u_qf_optimizer = optimizer_class(
            self._u_qf.parameters(),
            lr=u_qf_lr,
            weight_decay=u_q_weight_decay,
            **optimizer_params
        )
        self._i_qf_optimizer = optimizer_class(
            self._i_qf.parameters(),
            lr=i_qf_lr,
            weight_decay=i_q_weight_decay,
            **optimizer_params
        )

        # Policy optimizer
        self._policy_optimizer = optimizer_class(
            self._policy.parameters(),
            lr=i_policy_lr,
            weight_decay=i_policy_weight_decay,
            **optimizer_params
        )

        # Policy regularization coefficients (weights)
        self._i_pol_pre_activ_weight = i_policy_pre_activation_weight
        self._i_pol_mixing_coeff_weight = i_policy_mixing_coeff_weight

        if u_policy_pre_activation_weight is None:
            u_policy_pre_activation_weight = [
                i_policy_pre_activation_weight
                for _ in range(self._n_unintentional)
            ]
        self._u_policy_pre_activ_weight = \
            ptu.FloatTensor(u_policy_pre_activation_weight)

        # Useful Variables for logging
        self.log_data = dict()
        self.log_data['Raw Pol Loss'] = np.zeros((
            self.num_train_steps_per_epoch,
            self._n_unintentional + 1,
        ))
        self.log_data['Pol Loss'] = np.zeros((
            self.num_train_steps_per_epoch,
            self._n_unintentional + 1,
        ))
        self.log_data['Qf Loss'] = np.zeros((
            self.num_train_steps_per_epoch,
            self._n_unintentional + 1,
        ))
        self.log_data['Rewards'] = np.zeros((
            self.num_train_steps_per_epoch,
            self._n_unintentional + 1,
        ))
        self.log_data['Policy Action'] = np.zeros((
            self.num_train_steps_per_epoch,
            self._n_unintentional + 1,
            self.env.action_dim,
        ))
        self.log_data['Mixing Weights'] = np.zeros((
            self.num_train_steps_per_epoch,
            self._n_unintentional,
            self.env.action_dim,
        ))

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

        # ######################## #
        # Get Next Obs Policy Info #
        # ######################## #
        i_next_actions, policy_info = self._target_policy(
            next_obs,
            pol_idx=None,
            optimize_policies=False,
        )
        u_next_actions = policy_info['pol_actions'].detach()

        # ########################## #
        # Unintentional Critics Step #
        # ########################## #
        u_rewards = \
            (batch['reward_vectors'] * self._u_reward_scales).unsqueeze(-1)
        u_terminals = (batch['terminal_vectors']).unsqueeze(-1)

        # Unintentional Q Values: Q(s', a')
        u_next_q = torch.cat(
            [
             self._u_target_qf(next_obs, u_next_actions[:, uu, :])[0][uu].unsqueeze(1)
             for uu in range(self._n_unintentional)
            ],
            dim=1
        )

        # Calculate Bellman Backup for Unintentional Q-values
        u_q_backup = u_rewards + (1. - u_terminals) * self.discount * u_next_q
        u_q_backup = u_q_backup.detach()
        u_q_backup = torch.clamp(u_q_backup, self._min_q_value, self._max_q_value)
        # Hack for ICLR rebuttal
        if hasattr(self, 'reward_type') and self.reward_type == 'indicator':
            # TODO: CHECK THIS
            u_q_backup = \
                torch.clamp(u_q_backup, -self.reward_scale/(1-self.discount), 0)

        u_q_pred = torch.cat([qq.unsqueeze(1)
                              for qq in self._u_qf(obs, actions)[0]],
                             dim=1)

        # Unintentional QF Loss: Mean Squared Bellman Equation (MSBE)
        u_qf_loss = \
            0.5*torch.mean((u_q_backup - u_q_pred)**2, dim=0).squeeze(-1)
        # MSBE Q Loss For all unintentional policies
        total_u_qf_loss = torch.sum(u_qf_loss)

        if self._residual_gradient_weight > 0:
            raise NotImplementedError

        # Update Unintentional Q-value functions
        self._u_qf_optimizer.zero_grad()
        total_u_qf_loss.backward()
        self._u_qf_optimizer.step()

        # ####################### #
        # Intentional Critic Step #
        # ####################### #
        # Get Intentional rewards and terminals
        i_rewards = batch['rewards']# * self.reward_scale
        i_terminals = batch['terminals']

        # Intentional target Q Values: Q(s', a')
        i_next_q = self._i_target_qf(next_obs, i_next_actions)[0]

        # Calculate Intentional QF Losses (Bellman Eq.)
        i_q_backup = i_rewards + (1. - i_terminals) * self.discount * i_next_q
        i_q_backup = i_q_backup.detach()
        i_q_backup = torch.clamp(i_q_backup, self._min_q_value, self._max_q_value)
        # Hack for ICLR rebuttal
        if hasattr(self, 'reward_type') and self.reward_type == 'indicator':
            # TODO: CHECK THIS
            i_q_backup = \
                torch.clamp(i_q_backup, -self.reward_scale/(1-self.discount), 0)

        i_q_pred = self._i_qf(obs, actions)[0]

        i_qf_loss = \
            0.5*torch.mean((i_q_backup - i_q_pred)**2, dim=0)

        if self._residual_gradient_weight > 0:
            raise NotImplementedError

        # Update Intentional Q-value model parameters
        self._i_qf_optimizer.zero_grad()
        i_qf_loss.backward()
        self._i_qf_optimizer.step()

        # #################### #
        # Unintentional Actors #
        # #################### #

        # Get Obs Policy Info #
        i_new_actions, policy_info = self._policy(
            obs,
            pol_idx=None,
            optimize_policies=False,
        )
        u_new_actions = policy_info['pol_actions']

        i_new_pre_tanh_value = policy_info['pre_tanh_value']
        u_new_pre_tanh_values = policy_info['pol_pre_tanh_values']
        new_mixing_coeff = policy_info['mixing_coeff']

        if self._action_prior == 'normal':
            raise NotImplementedError
        else:
            u_policy_prior_log_probs = 0.0  # Uniform prior

        # Get Unintentional Q1(s, a)
        u_q_new_actions = torch.cat(
            [self._u_qf(obs, u_new_actions[:, uu, :])[0][uu].unsqueeze(1)
             for uu in range(self._n_unintentional)
             ],
            dim=1
        )

        # Unintentional Policies KL loss: - (E_a[Q(s, a)])
        u_raw_policy_loss = -u_q_new_actions.mean(dim=0).squeeze(-1)

        # Get Unintentional Policies regularization loss
        u_pre_activation_reg_loss = \
            self._u_policy_pre_activ_weight * \
            (u_new_pre_tanh_values**2).sum(dim=-1).mean(dim=0).mean(dim=-1)
        u_policy_regu_loss = u_pre_activation_reg_loss + 0

        # Get Unintentional Policies Total loss
        u_policy_loss = (u_raw_policy_loss + u_policy_regu_loss)
        total_u_policy_loss = torch.sum(u_policy_loss)

        # ################# #
        # Intentional Actor #
        # ################# #
        if self._action_prior == 'normal':
            raise NotImplementedError
        else:
            i_policy_prior_log_probs = 0.0  # Uniform prior

        # Intentional Q(s, a)
        i_q_new_actions = self._i_qf(obs, i_new_actions)[0]

        # Intentional KL loss: - (E_a[Q(s, a)])
        i_raw_policy_loss = -i_q_new_actions.mean()

        # Intentional policy regularization loss
        i_pre_activation_reg_loss = \
            self._i_pol_pre_activ_weight * \
            (i_new_pre_tanh_value**2).sum(dim=-1).mean()
        # TODO: Check the mixing coeff loss:
        mixing_coeff_loss = self._i_pol_mixing_coeff_weight * \
            0.5*((new_mixing_coeff ** 2).sum(dim=-1)).mean()
        i_policy_regu_loss = (i_pre_activation_reg_loss + mixing_coeff_loss)

        # Intentional Policy Total loss
        i_policy_loss = (i_raw_policy_loss + i_policy_regu_loss)

        # Update both Intentional and Unintentional Policies at the same time
        self._policy_optimizer.zero_grad()
        total_iu_loss = total_u_policy_loss + i_policy_loss
        total_iu_loss.backward()
        self._policy_optimizer.step()

        # ###################### #
        # Update Target Networks #
        # ###################### #
        if self._n_train_steps_total % self._u_target_update_interval == 0:
            ptu.soft_update_from_to(
                source=self._u_qf,
                target=self._u_target_qf,
                tau=self._u_soft_target_tau
            )
        if self._n_train_steps_total % self._i_target_update_interval == 0:
            ptu.soft_update_from_to(
                source=self._i_qf,
                target=self._i_target_qf,
                tau=self._i_soft_target_tau
            )
        if self._n_train_steps_total % self._i_target_update_interval == 0:
            ptu.soft_update_from_to(
                source=self._policy,
                target=self._target_policy,
                tau=self._i_soft_target_tau
            )

        # ############### #
        # LOG Useful Data #
        # ############### #
        self.log_data['Raw Pol Loss'][step_idx, :-1] = \
            ptu.get_numpy(u_raw_policy_loss.squeeze(-1))
        self.log_data['Raw Pol Loss'][step_idx, -1] = \
            ptu.get_numpy(i_raw_policy_loss)

        self.log_data['Pol Loss'][step_idx, :-1] = \
            ptu.get_numpy(u_policy_loss.squeeze(-1))
        self.log_data['Pol Loss'][step_idx, -1] = \
            ptu.get_numpy(i_policy_loss)

        self.log_data['Qf Loss'][step_idx, :-1] = \
            ptu.get_numpy(u_qf_loss.squeeze(-1))
        self.log_data['Qf Loss'][step_idx, -1] = ptu.get_numpy(i_qf_loss)

        self.log_data['Rewards'][step_idx, :-1] = \
            ptu.get_numpy(u_rewards.mean(dim=0).squeeze(-1))
        self.log_data['Rewards'][step_idx, -1] = \
            ptu.get_numpy(i_rewards.mean(dim=0).squeeze(-1))

        self.log_data['Mixing Weights'][step_idx, :, :] = \
            ptu.get_numpy(new_mixing_coeff.mean(dim=0))

        self.log_data['Policy Action'][step_idx, :-1, :] = \
            ptu.get_numpy(u_new_actions.mean(dim=0))
        self.log_data['Policy Action'][step_idx, -1, :] = \
            ptu.get_numpy(i_new_actions.mean(dim=0))

        if self._log_tensorboard:
            self._summary_writer.add_scalar(
                'TrainingI/qf_loss',
                ptu.get_numpy(i_qf_loss),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'TrainingI/avg_reward',
                ptu.get_numpy(i_rewards.mean()),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'TrainingI/policy_loss',
                ptu.get_numpy(i_policy_loss),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'TrainingI/q_vals',
                ptu.get_numpy(i_q_new_actions.mean()),
                self._n_env_steps_total
            )

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

        snapshot = IncrementalRLAlgorithm.get_epoch_snapshot(self, epoch)

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
                np.nan_to_num(np.mean(
                    self.log_data['Qf Loss'][:max_step, uu]
                ))
            self.eval_statistics['[U-%02d] Raw Policy Loss' % uu] = \
                np.nan_to_num(np.mean(
                    self.log_data['Raw Pol Loss'][:max_step, uu]
                ))
            self.eval_statistics['[U-%02d] Policy Loss' % uu] = \
                np.nan_to_num(np.mean(
                    self.log_data['Pol Loss'][:max_step, uu]
                ))
            self.eval_statistics['[U-%02d] Rewards' % uu] = \
                np.nan_to_num(np.mean(
                    self.log_data['Rewards'][:max_step, uu]
                ))
            self.eval_statistics['[U-%02d] Mixing Weights' % uu] = \
                np.nan_to_num(np.mean(
                    self.log_data['Mixing Weights'][:max_step, uu]
                ))

            for aa in range(self.env.action_dim):
                self.eval_statistics['[U-%02d] Policy Action [%02d]' % (uu, aa)] = \
                    np.nan_to_num(np.mean(
                        self.log_data['Policy Action'][:max_step, uu, aa]
                    ))

        # Intentional info
        self.eval_statistics['[I] Qf Loss'] = \
            np.nan_to_num(np.mean(
                self.log_data['Qf Loss'][:max_step, -1]
            ))
        self.eval_statistics['[I] Raw Policy Loss'] = \
            np.nan_to_num(np.mean(
                self.log_data['Raw Pol Loss'][:max_step, -1]
            ))
        self.eval_statistics['[I] Policy Loss'] = \
            np.nan_to_num(np.mean(
                self.log_data['Pol Loss'][:max_step, -1]
            ))
        self.eval_statistics['[I] Rewards'] = \
            np.nan_to_num(np.mean(
                self.log_data['Rewards'][:max_step, -1]
            ))

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
        i_test_paths = self.eval_sampler.obtain_samples()
        statistics.update(eval_util.get_generic_path_information(
            i_test_paths, stat_prefix="[I] Test",
        ))
        average_return = eval_util.get_average_returns(i_test_paths)
        statistics['[I] Test AverageReturn'] = average_return * self.reward_scale

        if self._exploration_paths:
            statistics.update(eval_util.get_generic_path_information(
                self._exploration_paths, stat_prefix="Exploration",
            ))
        else:
            statistics.update(eval_util.get_generic_path_information(
                i_test_paths, stat_prefix="Exploration",
            ))

        if self._log_tensorboard:
            self._summary_writer.add_scalar(
                'EvaluationI/avg_return',
                statistics['[I] Test AverageReturn'],
                self._n_epochs
            )

            self._summary_writer.add_scalar(
                'EvaluationI/avg_reward',
                statistics['[I] Test Rewards Mean'] * self.reward_scale,
                self._n_epochs
            )

        if hasattr(self.env, "log_diagnostics"):
            pass
            # # TODO: CHECK ENV LOG_DIAGNOSTICS
            # print('TODO: WE NEED LOG_DIAGNOSTICS IN ENV')

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

        # Reset log_data
        for key in self.log_data.keys():
            self.log_data[key].fill(0)

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

        IncrementalRLAlgorithm._handle_step(
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

        IncrementalRLAlgorithm._handle_rollout_ending(self)
