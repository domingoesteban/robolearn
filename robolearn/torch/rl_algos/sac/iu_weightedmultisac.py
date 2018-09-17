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

import robolearn.torch.pytorch_util as ptu
from robolearn.core import logger, eval_util
from robolearn.utils.samplers import InPlacePathSampler
from robolearn.torch.rl_algos.torch_incremental_rl_algorithm import TorchIncrementalRLAlgorithm
from robolearn.policies import MakeDeterministic
from robolearn.torch.policies import WeightedMultiPolicySelector
from torch.autograd import Variable

from tensorboardX import SummaryWriter


class IUWeightedMultiSAC(TorchIncrementalRLAlgorithm):
    """Intentional-Unintentional Soft Actor Critic (IU-SAC)
    with MultiHead Networks.

    """
    def __init__(
            self,
            env,
            policy,
            u_qf,
            u_vf,

            replay_buffer,
            batch_size=1024,
            i_qf=None,
            i_vf=None,
            eval_env=None,

            u_qf2=None,
            i_qf2=None,
            reparameterize=True,
            action_prior='uniform',

            i_entropy_scale=1.,
            u_entropy_scale=None,

            i_policy_lr=1e-3,
            u_policies_lr=1e-3,
            u_mixing_lr=1e-3,

            i_qf_lr=1e-3,
            i_qf2_lr=1e-3,
            i_vf_lr=1e-3,
            u_qf_lr=1e-3,
            u_qf2_lr=1e-3,
            u_vf_lr=1e-3,

            i_policy_mean_regu_weight=1e-3,
            i_policy_std_regu_weight=1e-3,
            i_policy_pre_activation_weight=0.,
            i_policy_mixing_coeff_weight=1e-3,

            u_policy_mean_regu_weight=None,
            u_policy_std_regu_weight=None,
            u_policy_pre_activation_weight=None,

            i_policy_weight_decay=0.,
            u_policy_weight_decay=0.,
            i_q_weight_decay=0.,
            u_q_weight_decay=0.,
            i_v_weight_decay=0.,
            u_v_weight_decay=0.,

            optimizer_class=optim.Adam,
            # optimizer_class=optim.SGD,
            amsgrad=True,

            i_soft_target_tau=1e-2,
            u_soft_target_tau=1e-2,
            i_target_update_interval=1,
            u_target_update_interval=1,

            u_reward_scales=None,

            save_replay_buffer=False,
            epoch_plotter=None,
            render_eval_paths=False,
            eval_deterministic=True,
            **kwargs
    ):

        # ######## #
        # Networks #
        # ######## #

        # Exploration Policy
        self._policy = policy

        # Evaluation Policy
        if eval_deterministic:
            eval_policy = MakeDeterministic(self._policy)
        else:
            eval_policy = self._policy

        TorchIncrementalRLAlgorithm.__init__(
            self,
            env=env,
            exploration_policy=self._policy,
            eval_env=eval_env,
            eval_policy=eval_policy,
            **kwargs
        )

        # Number of Unintentional Tasks (Composable Tasks)
        self._n_unintentional = self._policy.n_heads

        # Important algorithm hyperparameters
        self._reparameterize = reparameterize
        assert self._reparameterize == self._policy.reparameterize
        self._action_prior = action_prior
        self._i_entropy_scale = i_entropy_scale
        if u_entropy_scale is None:
            u_entropy_scale = [i_entropy_scale
                               for _ in range(self._n_unintentional)]
        self._u_entropy_scale = u_entropy_scale

        # Intentional (Main Task) Q-function and V-function
        self._i_qf = i_qf
        self._i_qf2 = i_qf2
        self._i_vf = i_vf
        self._i_target_vf = self._i_vf.copy()

        # Unintentional (Composable Tasks) Q-function and V-function
        self._u_qf = u_qf
        self._u_qf2 = u_qf2
        self._u_vf = u_vf
        self._u_target_vf = self._u_vf.copy()

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
        self._u_reward_scales = u_reward_scales

        # Replay Buffer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.save_replay_buffer = save_replay_buffer

        # ########## #
        # Optimizers #
        # ########## #

        # Q-function and V-function Optimization Criteria
        self._u_qf_criterion = nn.MSELoss()
        self._u_vf_criterion = nn.MSELoss()
        self._i_qf_criterion = nn.MSELoss()
        self._i_vf_criterion = nn.MSELoss()

        # Q-function optimizers
        self._u_qf_optimizer = optimizer_class(self._u_qf.parameters(),
                                               lr=u_qf_lr, amsgrad=amsgrad,
                                               weight_decay=u_q_weight_decay)
        self._i_qf_optimizer = optimizer_class(self._i_qf.parameters(),
                                               lr=i_qf_lr, amsgrad=amsgrad,
                                               weight_decay=i_q_weight_decay)
        if u_qf2 is not None:
            self._u_qf2_optimizer = optimizer_class(self._u_qf2.parameters(),
                                                    lr=u_qf2_lr,
                                                    amsgrad=amsgrad,
                                                    weight_decay=u_q_weight_decay)
        if i_qf2 is not None:
            self._i_qf2_optimizer = optimizer_class(self._i_qf2.parameters(),
                                                    lr=i_qf2_lr,
                                                    amsgrad=amsgrad,
                                                    weight_decay=i_q_weight_decay)

        # V-function optimizers
        self._u_vf_optimizer = optimizer_class(self._u_vf.parameters(),
                                               lr=u_vf_lr, amsgrad=amsgrad,
                                               weight_decay=u_v_weight_decay)
        self._i_vf_optimizer = optimizer_class(self._i_vf.parameters(),
                                               lr=i_vf_lr, amsgrad=amsgrad,
                                               weight_decay=i_v_weight_decay)

        # Policy optimizer
        # self._policy_optimizer = optimizer_class([
        #     {'params': self._policy.shared_parameters(),
        #      'lr': i_policy_lr},
        #     {'params': self._policy.policies_parameters(),
        #      'lr': i_policy_lr},
        #     {'params': self._policy.mixing_parameters(),
        #      'lr': i_policy_lr},
        # ])
        self._policy_optimizer = optimizer_class(self._policy.parameters(),
                                                 lr=i_policy_lr, amsgrad=amsgrad,
                                                 weight_decay=i_policy_weight_decay,
                                                 )
        self._mixing_optimizer = \
            optimizer_class(chain(self._policy.shared_parameters(),
                                  self._policy.mixing_parameters()),
                            lr=u_mixing_lr, amsgrad=amsgrad,
                            weight_decay=i_policy_weight_decay)
        self._policies_optimizer = \
            optimizer_class(chain(self._policy.shared_parameters(),
                                  self._policy.policies_parameters()),
                            lr=u_policies_lr, amsgrad=amsgrad,
                            weight_decay=u_policy_weight_decay)

        # Policy regularization coefficients (weights)
        self._i_policy_mean_regu_weight = i_policy_mean_regu_weight
        self._i_policy_std_regu_weight = i_policy_std_regu_weight
        self._i_policy_pre_activation_weight = i_policy_pre_activation_weight
        self._i_policy_mixing_coeff_weight = i_policy_mixing_coeff_weight

        if u_policy_mean_regu_weight is None:
            u_policy_mean_regu_weight = [i_policy_mean_regu_weight
                                         for _ in range(self._n_unintentional)]
        self._u_policy_mean_regu_weight = u_policy_mean_regu_weight
        if u_policy_std_regu_weight is None:
            u_policy_std_regu_weight = [i_policy_std_regu_weight
                                        for _ in range(self._n_unintentional)]
        self._u_policy_std_regu_weight = u_policy_std_regu_weight
        if u_policy_pre_activation_weight is None:
            u_policy_pre_activation_weight = [i_policy_pre_activation_weight
                                       for _ in range(self._n_unintentional)]
        self._u_policy_pre_activation_weight = u_policy_pre_activation_weight

        # ########### #
        # Other Stuff
        # ########### #
        self.eval_statistics = None
        self._epoch_plotter = epoch_plotter
        self.render_eval_paths = render_eval_paths

        self._summary_writer = SummaryWriter(log_dir=logger.get_snapshot_dir())

        # Evaluation Sampler (One for each unintentional)
        self.eval_samplers = [
            InPlacePathSampler(env=env,
                               policy=WeightedMultiPolicySelector(self._policy,
                                                                  idx),
                               max_samples=self.num_steps_per_eval,
                               max_path_length=self.max_path_length,
                               deterministic=True)
            for idx in range(self._n_unintentional)
        ]

        # Useful Variables for logging
        self.logging_pol_kl_loss = np.zeros((self.num_env_steps_per_epoch,
                                             self._n_unintentional + 1))
        self.logging_qf_loss = np.zeros((self.num_env_steps_per_epoch,
                                         self._n_unintentional + 1))
        self.logging_qf2_loss = np.zeros((self.num_env_steps_per_epoch,
                                         self._n_unintentional + 1))
        self.logging_vf_loss = np.zeros((self.num_env_steps_per_epoch,
                                         self._n_unintentional + 1))
        self.logging_rewards = np.zeros((self.num_env_steps_per_epoch,
                                         self._n_unintentional + 1))
        self.logging_policy_entropy = np.zeros((self.num_env_steps_per_epoch,
                                                self._n_unintentional + 1))
        self.logging_policy_log_std = np.zeros((self.num_env_steps_per_epoch,
                                                self.env.action_dim,
                                                self._n_unintentional + 1))
        self.logging_policy_mean = np.zeros((self.num_env_steps_per_epoch,
                                             self.env.action_dim,
                                             self._n_unintentional + 1))
        self.logging_mixing_coeff = np.zeros((self.num_env_steps_per_epoch,
                                              self.env.action_dim,
                                              self._n_unintentional))

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

            if terminal:
                self.env.reset()

    def _do_training(self):
        # Get batch of samples
        batch = self.get_batch()

        # Update Intentional-Unintentional Networks
        update_outputs = self._update_all_soft_networks(batch)
        self._print_soft_statistics(update_outputs)

    def _do_not_training(self):
        return
        # # Get batch of samples
        # if self.replay_buffer.num_steps_can_sample() < self.batch_size:
        #     return
        # batch = self.get_batch()
        #
        # obs = batch['observations']
        # actions = batch['actions']
        # next_obs = batch['next_observations']
        #
        # step_idx = int((self._n_env_steps_total - 1) %
        #                self.num_env_steps_per_epoch)
        #
        # # ############# #
        # # UNINTENTIONAL #
        # # ############# #
        #
        # policy = self._policy
        # qf = self._u_qf
        # vf = self._u_vf
        # target_vf = self._u_target_vf
        #
        # all_q_pred = qf(obs, actions)[0]
        # all_v_pred = vf(obs)[0]
        # all_target_v_values = target_vf(next_obs)[0]
        # for uu in range(self._n_unintentional):
        #     # print('TODO: SCALING REWARDSS')
        #     rewards = batch['reward_vectors'][:, uu].unsqueeze(-1) \
        #               * self._u_reward_scales[uu]# ** 2
        #     terminals = batch['terminal_vectors'][:, uu].unsqueeze(-1)
        #
        #     q_pred = all_q_pred[uu]
        #     v_pred = all_v_pred[uu]
        #     target_v_values = all_target_v_values[uu]
        #
        #     new_actions, policy_info = policy(obs, deterministic=False,
        #                                       return_log_prob=True,
        #                                       pol_idx=uu,
        #                                       optimize_policies=True)
        #     log_pi = policy_info['log_prob']
        #     policy_mean = policy_info['mean']
        #     policy_log_std = policy_info['log_std']
        #
        #     q_target = rewards + (1. - terminals) * self.discount * target_v_values
        #
        #     q_new_actions = qf(obs, new_actions)[0][uu]
        #     v_target = q_new_actions - log_pi
        #
        #     log_policy_target = q_new_actions - v_pred
        #
        #     self.logging_policy_entropy[step_idx, uu] = \
        #         ptu.get_numpy(-log_pi.mean(dim=0))
        #     self.logging_log_policy_target[step_idx, uu] = \
        #         ptu.get_numpy(log_policy_target.mean(dim=0))
        #     self.logging_policy_log_std[step_idx, uu] = \
        #         ptu.get_numpy(policy_log_std.mean())
        #     self.logging_policy_mean[step_idx, uu] = policy_mean.mean()
        #     self.logging_qf_loss[step_idx, uu] = \
        #         ptu.get_numpy(self._qf_criterion(q_pred, q_target.detach()))
        #     self.logging_vf_loss[step_idx, uu] = \
        #         ptu.get_numpy(self._vf_criterion(v_pred, v_target.detach()))
        #     self.logging_rewards[step_idx, uu] = \
        #         ptu.get_numpy(rewards.mean(dim=0))
        #
        # # ########### #
        # # INTENTIONAL #
        # # ########### #
        # rewards = batch['rewards']
        # terminals = batch['terminals']
        # policy = self._policy
        # qf = self._i_qf
        # vf = self._i_vf
        # target_vf = self._i_target_vf
        #
        # q_pred = qf(obs, actions)[0]
        # v_pred = vf(obs)[0]
        # new_actions, policy_info = policy(obs, deterministic=False,
        #                                   return_log_prob=True,
        #                                   pol_idx=None,
        #                                   optimize_policies=False)
        # log_pi = policy_info['log_prob']
        # policy_mean = policy_info['mean']
        # policy_log_std = policy_info['log_std']
        # pre_tanh_value = policy_info['pre_tanh_value']
        # mixing_coeff = policy_info['mixing_coeff']
        #
        # target_v_values = target_vf(next_obs)[0]
        #
        # q_target = rewards + (1. - terminals) * self.discount * target_v_values
        #
        # q_new_actions = qf(obs, new_actions)[0]
        # v_target = q_new_actions - log_pi
        #
        # log_policy_target = q_new_actions - v_pred
        #
        # self.logging_policy_entropy[step_idx, -1] = \
        #     ptu.get_numpy(-log_pi.mean(dim=0))
        # self.logging_log_policy_target[step_idx, -1] = \
        #     ptu.get_numpy(log_policy_target.mean(dim=0))
        # self.logging_policy_log_std[step_idx, -1] = \
        #     ptu.get_numpy(policy_log_std.mean())
        # self.logging_policy_mean[step_idx, -1] = \
        #     ptu.get_numpy(policy_mean.mean())
        # self.logging_qf_loss[step_idx, -1] = \
        #     ptu.get_numpy(self._qf_criterion(q_pred, q_target.detach()))
        # self.logging_vf_loss[step_idx, -1] = \
        #     ptu.get_numpy(self._vf_criterion(v_pred, v_target.detach()))
        # self.logging_rewards[step_idx, -1] = \
        #     ptu.get_numpy(rewards.mean(dim=0))
        # self.logging_mixing_coeff[step_idx, :] = \
        #     ptu.get_numpy(mixing_coeff.mean(dim=0))

    def _update_all_soft_networks(self, batch):
        # Get from batch
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        # Get the idx for logging
        step_idx = self._n_epoch_train_steps

        """
        ** ****************** **
        ** ****************** ** 
        ** UNINTENTIONAL LOSS ** 
        ** ****************** **
        ** ****************** ** 
        """
        policy = self._policy
        qf = self._u_qf
        qf2 = self._u_qf2
        vf = self._u_vf
        target_vf = self._u_target_vf

        # ########### #
        # Critic Step #
        # ########### #
        u_v_values_next = target_vf(next_obs)[0]  # Get all unintentional V-vals
        u_q_preds = qf(obs, actions)[0]  # Get all unintentional Q-values
        accum_u_qf_loss = 0
        if qf2 is not None:
            u_q2_preds = qf2(obs, actions)[0]  # Get all unintentional Q2-values
            accum_u_qf2_loss = 0
        for uu in range(self._n_unintentional):
            # Get batch rewards and terminal for unintentional tasks
            rewards = batch['reward_vectors'][:, uu].unsqueeze(-1) \
                      * self._u_reward_scales[uu]
            terminals = batch['terminal_vectors'][:, uu].unsqueeze(-1)

            # Calculate QF Loss (Soft Bellman Eq.)
            v_value_next = u_v_values_next[uu]
            q_target = rewards + (1. - terminals) * self.discount * v_value_next
            u_qf_loss = 0.5*self._u_qf_criterion(u_q_preds[uu],
                                                 q_target.detach())
            accum_u_qf_loss += u_qf_loss

            if qf2 is not None:
                u_qf2_loss = 0.5*self._u_qf_criterion(u_q2_preds[uu],
                                                      q_target.detach())
                accum_u_qf2_loss += u_qf2_loss

            # Log data
            self.logging_qf_loss[step_idx, uu] = ptu.get_numpy(u_qf_loss)
            if qf2 is not None:
                self.logging_qf2_loss[step_idx, uu] = ptu.get_numpy(u_qf2_loss)
            self.logging_rewards[step_idx, uu] = \
                ptu.get_numpy(rewards.mean(dim=0))

            self._summary_writer.add_scalar('TrainingU%2d/qf_loss' % uu,
                                            ptu.get_numpy(u_qf_loss),
                                            self._n_env_steps_total)
            if qf2 is not None:
                self._summary_writer.add_scalar('TrainingU%2d/qf2_loss' % uu,
                                                ptu.get_numpy(u_qf2_loss),
                                                self._n_env_steps_total)
            self._summary_writer.add_scalar('TrainingU%2d/avg_reward' % uu,
                                            ptu.get_numpy(rewards.mean()),
                                            self._n_env_steps_total)

        # Update Unintentional Q-Values
        self._u_qf_optimizer.zero_grad()
        accum_u_qf_loss.backward()
        self._u_qf_optimizer.step()

        if qf2 is not None:
            self._u_qf2_optimizer.zero_grad()
            accum_u_qf2_loss.backward()
            self._u_qf2_optimizer.step()

        # ############### #
        # Actor & Vf Step #
        # ############### #
        accum_u_policy_loss = 0
        accum_u_vf_loss = 0
        u_v_preds = vf(obs)[0]  # Get all unintentional V-vals
        for uu in range(self._n_unintentional):
            # Get Actions and Info from Unintentional Policy
            new_actions, policy_info = policy(obs, deterministic=False,
                                              return_log_prob=True,
                                              pol_idx=uu,
                                              optimize_policies=True)
            log_pi = policy_info['log_prob'] * self._u_entropy_scale[uu]
            policy_mean = policy_info['mean']
            policy_log_std = policy_info['log_std']
            pre_tanh_value = policy_info['pre_tanh_value']

            if self._action_prior == 'normal':
                raise NotImplementedError
            else:
                policy_prior_log_probs = 0.0  # Uniform prior

            v_pred = u_v_preds[uu]
            q_new_actions = qf(obs, new_actions)[0][uu]

            if qf2 is not None:
                q2_new_actions = qf2(obs, new_actions)[0][uu]
                q_new_actions = torch.min(q_new_actions, q2_new_actions)

            advantages_new_actions = q_new_actions - v_pred.detach()

            # KL loss
            if self._reparameterize:
                # TODO: In HAarnoja code it does not use the min, but the one from self._qf
                # policy_kl_loss = torch.mean(log_pi - q_new_actions)
                # policy_kl_loss = -torch.mean(q_new_actions - log_pi)
                policy_kl_loss = -torch.mean(advantages_new_actions - log_pi)
            else:
                policy_kl_loss = (
                        log_pi * (log_pi - q_new_actions + v_pred
                                  - policy_prior_log_probs).detach()
                ).mean()

            # Regularization loss
            mean_reg_loss = self._u_policy_mean_regu_weight[uu] * \
                (policy_mean ** 2).mean()
            std_reg_loss = self._u_policy_std_regu_weight[uu] * \
                (policy_log_std ** 2).mean()
            pre_activation_reg_loss = self._u_policy_pre_activation_weight[uu] * \
                (pre_tanh_value**2).sum(dim=-1).mean()
            policy_regu_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss

            accum_u_policy_loss += (policy_kl_loss + policy_regu_loss)

            # Calculate Intentional Vf Loss
            v_target = q_new_actions - log_pi + policy_prior_log_probs
            u_vf_loss = 0.5*self._u_vf_criterion(v_pred, v_target.detach())

            accum_u_vf_loss += u_vf_loss

            # ############### #
            # LOG Useful Data #
            # ############### #
            self.logging_policy_entropy[step_idx, uu] = \
                ptu.get_numpy(-log_pi.mean(dim=0))
            self.logging_policy_log_std[step_idx, :, uu] = \
                ptu.get_numpy(policy_log_std.mean(dim=0))
            self.logging_policy_mean[step_idx, :, uu] = \
                ptu.get_numpy(policy_mean.mean(dim=0))
            self.logging_vf_loss[step_idx, uu] = \
                ptu.get_numpy(u_vf_loss)
            self.logging_pol_kl_loss[step_idx, uu] = \
                ptu.get_numpy(policy_kl_loss)

            self._summary_writer.add_scalar('TrainingU%2d/vf_loss' % uu,
                                            ptu.get_numpy(u_vf_loss),
                                            self._n_env_steps_total)
            self._summary_writer.add_scalar('TrainingU%2d/policy_loss' % uu,
                                            ptu.get_numpy((policy_kl_loss +
                                                           policy_regu_loss)),
                                            self._n_env_steps_total)
            self._summary_writer.add_scalar('TrainingU%2d/policy_entropy' % uu,
                                            ptu.get_numpy(-log_pi.mean()),
                                            self._n_env_steps_total)
            self._summary_writer.add_scalar('TrainingU%2d/policy_mean' % uu,
                                            ptu.get_numpy(policy_mean.mean()),
                                            self._n_env_steps_total)
            self._summary_writer.add_scalar('TrainingU%2d/policy_std' % uu,
                                            np.exp(ptu.get_numpy(
                                                policy_log_std.mean())),
                                            self._n_env_steps_total)
            self._summary_writer.add_scalar('TrainingU%2d/q_vals' % uu,
                                            ptu.get_numpy(q_new_actions.mean()),
                                            self._n_env_steps_total)
            self._summary_writer.add_scalar('TrainingU%2d/avg_advantage' % uu,
                                            ptu.get_numpy(advantages_new_actions.mean()),
                                            self._n_env_steps_total)

        # Update Unintentional (Composable) Policies
        # self._policy_optimizer.zero_grad()
        self._policies_optimizer.zero_grad()
        accum_u_policy_loss.backward()
        self._policies_optimizer.step()

        # Update Unintentional V-value
        self._u_vf_optimizer.zero_grad()
        accum_u_vf_loss.backward()
        self._u_vf_optimizer.step()

        # Update V Target Network
        if self._n_train_steps_total % self._u_target_update_interval == 0:
            self._update_v_target_network(vf=self._u_vf,
                                          target_vf=self._u_target_vf,
                                          soft_target_tau=self._u_soft_target_tau)

        """
        ** **************** **
        ** **************** ** 
        ** INTENTIONAL STEP ** 
        ** **************** ** 
        ** **************** ** 
        """
        rewards = batch['rewards']
        terminals = batch['terminals']

        policy = self._policy
        qf = self._i_qf
        qf2 = self._i_qf2
        vf = self._i_vf
        target_vf = self._i_target_vf

        # ########### #
        # Critic Step #
        # ########### #
        # Calculate QF Loss (Soft Bellman Eq.)
        v_value_next = target_vf(next_obs)[0]
        q_pred = qf(obs, actions)[0]
        q_target = rewards + (1. - terminals) * self.discount * v_value_next
        i_qf_loss = 0.5*self._i_qf_criterion(q_pred, q_target.detach())

        # Update Intentional Q-value
        self._i_qf_optimizer.zero_grad()
        i_qf_loss.backward()
        self._i_qf_optimizer.step()

        if qf2 is not None:
            q2_pred = qf2(obs, actions)[0]
            q2_target = rewards + (1. - terminals) * self.discount * v_value_next
            i_qf2_loss = 0.5*self._i_qf_criterion(q2_pred, q2_target.detach())

            # Update Intentional Q-value
            self._i_qf2_optimizer.zero_grad()
            i_qf2_loss.backward()
            self._i_qf2_optimizer.step()

        # ########## #
        # Actor Step #
        # ########## #
        # Calculate Intentional Policy Loss
        new_actions, policy_info = policy(obs, deterministic=False,
                                          return_log_prob=True,
                                          pol_idx=None,
                                          optimize_policies=False)
                                          # optimize_policies=True)

        log_pi = policy_info['log_prob'] * self._i_entropy_scale
        policy_mean = policy_info['mean']
        policy_log_std = policy_info['log_std']
        pre_tanh_value = policy_info['pre_tanh_value']
        mixing_coeff = policy_info['mixing_coeff']

        if self._action_prior == 'normal':
            raise NotImplementedError
        else:
            policy_prior_log_probs = 0.0

        v_pred = vf(obs)[0]
        q_new_actions = qf(obs, new_actions)[0]

        if qf2 is not None:
            q2_new_actions = qf2(obs, new_actions)[0]
            q_new_actions = torch.min(q_new_actions, q2_new_actions)

        advantages_new_actions = q_new_actions - v_pred.detach()

        # KL loss
        if self._reparameterize:
            # TODO: In HAarnoja code it does not use the min, but the one from self._qf
            # policy_kl_loss = torch.mean(log_pi - q_new_actions)
            # policy_kl_loss = -torch.mean(q_new_actions - log_pi)
            policy_kl_loss = -torch.mean(advantages_new_actions - log_pi)
        else:
            policy_kl_loss = (
                    log_pi * (log_pi - q_new_actions + v_pred
                              - policy_prior_log_probs).detach()
            ).mean()

        # Regularization loss
        mean_reg_loss = self._i_policy_mean_regu_weight * \
            (policy_mean ** 2).mean()
        std_reg_loss = self._i_policy_std_regu_weight * \
            (policy_log_std ** 2).mean()
        pre_activation_reg_loss = self._i_policy_pre_activation_weight * \
            (pre_tanh_value**2).sum(dim=-1).mean()
        mixing_coeff_loss = self._i_policy_mixing_coeff_weight * \
            (mixing_coeff ** 2).sum(dim=-1).mean()  # TODO: CHECK THIS

        policy_regu_loss = mean_reg_loss + std_reg_loss + \
                          pre_activation_reg_loss + mixing_coeff_loss

        i_policy_loss = policy_kl_loss + policy_regu_loss

        # Update Intentional Policy
        # self._policy_optimizer.zero_grad()
        self._mixing_optimizer.zero_grad()
        i_policy_loss.backward()
        self._mixing_optimizer.step()

        # ############### #
        # V-function Step #
        # ############### #
        # Calculate Intentional Vf Loss
        v_target = q_new_actions - log_pi + policy_prior_log_probs
        i_vf_loss = 0.5*self._i_vf_criterion(v_pred, v_target.detach())

        # Update Intentional V-value
        self._i_vf_optimizer.zero_grad()
        i_vf_loss.backward()
        self._i_vf_optimizer.step()

        # Update Intentional V Target Network
        if self._n_train_steps_total % self._i_target_update_interval == 0:
            self._update_v_target_network(vf=self._i_vf,
                                          target_vf=self._i_target_vf,
                                          soft_target_tau=self._i_soft_target_tau)

        # # TEMPORAL
        # if isinstance(self._policy, TanhGaussianWeightedMultiPolicy2):
        #     mixing_coeff = mixing_coeff.mean(dim=-2)  # Average over dA

        # ########################### #
        # LOG Useful Intentional Data #
        # ########################### #
        self.logging_policy_entropy[step_idx, -1] = \
            ptu.get_numpy(-log_pi.mean(dim=0))
        self.logging_policy_log_std[step_idx, :, -1] = \
            ptu.get_numpy(policy_log_std.mean(dim=0))
        self.logging_policy_mean[step_idx, :, -1] = \
            ptu.get_numpy(policy_mean.mean(dim=0))
        self.logging_qf_loss[step_idx, -1] = ptu.get_numpy(i_qf_loss)
        self.logging_vf_loss[step_idx, -1] = ptu.get_numpy(i_vf_loss)
        self.logging_pol_kl_loss[step_idx, -1] = ptu.get_numpy(policy_kl_loss)
        self.logging_rewards[step_idx, -1] = \
            ptu.get_numpy(rewards.mean(dim=0))
        self.logging_mixing_coeff[step_idx, :, :] = \
            ptu.get_numpy(mixing_coeff.mean(dim=0))

        self._summary_writer.add_scalar('TrainingI/qf_loss',
                                        ptu.get_numpy(i_qf_loss),
                                        self._n_env_steps_total)
        if qf2 is not None:
            self._summary_writer.add_scalar('TrainingI/qf2_loss',
                                            ptu.get_numpy(i_qf2_loss),
                                            self._n_env_steps_total)
        self._summary_writer.add_scalar('TrainingI/vf_loss',
                                        ptu.get_numpy(i_vf_loss),
                                        self._n_env_steps_total)
        self._summary_writer.add_scalar('TrainingI/avg_reward',
                                        ptu.get_numpy(rewards.mean()),
                                        self._n_env_steps_total)
        self._summary_writer.add_scalar('TrainingI/policy_loss',
                                        ptu.get_numpy(i_policy_loss),
                                        self._n_env_steps_total)
        self._summary_writer.add_scalar('TrainingI/policy_entropy',
                                        ptu.get_numpy(-log_pi.mean()),
                                        self._n_env_steps_total)
        self._summary_writer.add_scalar('TrainingI/policy_mean',
                                        ptu.get_numpy(policy_mean.mean()),
                                        self._n_env_steps_total)
        self._summary_writer.add_scalar('TrainingI/policy_std',
                                        np.exp(ptu.get_numpy(policy_log_std.mean())),
                                        self._n_env_steps_total)
        self._summary_writer.add_scalar('TrainingI/q_vals',
                                        ptu.get_numpy(q_new_actions.mean()),
                                        self._n_env_steps_total)
        self._summary_writer.add_scalar('TrainingI/avg_advantage',
                                        ptu.get_numpy(advantages_new_actions.mean()),
                                        self._n_env_steps_total)

        for uu in range(self._n_unintentional):
            self._summary_writer.add_scalar('TrainingI/weight%02d' % uu,
                                            ptu.get_numpy(mixing_coeff[:, uu].mean()),
                                            self._n_env_steps_total)

        # LOG NN VALUES AND GRADIENTS
        if self._n_env_steps_total % 500 == 0:
            for name, param in self._policy.named_parameters():
                self._summary_writer.add_histogram('policy/'+name,
                                                   param.data.cpu().numpy(),
                                                   self._n_env_steps_total)
                self._summary_writer.add_histogram('policy_grad/'+name,
                                                   param.grad.data.cpu().numpy(),
                                                   self._n_env_steps_total)

            for name, param in self._u_qf.named_parameters():
                self._summary_writer.add_histogram('u_qf/'+name,
                                                   param.data.cpu().numpy(),
                                                   self._n_env_steps_total)
                self._summary_writer.add_histogram('u_qf_grad/'+name,
                                                   param.grad.data.cpu().numpy(),
                                                   self._n_env_steps_total)
            for name, param in self._i_qf.named_parameters():
                self._summary_writer.add_histogram('i_qf/'+name,
                                                   param.data.cpu().numpy(),
                                                   self._n_env_steps_total)
                self._summary_writer.add_histogram('i_qf_grad/'+name,
                                                   param.grad.data.cpu().numpy(),
                                                   self._n_env_steps_total)

            if self._u_qf2 is not None:
                for name, param in self._u_qf2.named_parameters():
                    self._summary_writer.add_histogram('u_qf2/'+name,
                                                       param.data.cpu().numpy(),
                                                       self._n_env_steps_total)
                    self._summary_writer.add_histogram('u_qf2_grad/'+name,
                                                       param.grad.data.cpu().numpy(),
                                                       self._n_env_steps_total)
            if self._i_qf2 is not None:
                for name, param in self._i_qf2.named_parameters():
                    self._summary_writer.add_histogram('i_qf2/'+name,
                                                       param.data.cpu().numpy(),
                                                       self._n_env_steps_total)
                    self._summary_writer.add_histogram('i_qf2_grad/'+name,
                                                       param.grad.data.cpu().numpy(),
                                                       self._n_env_steps_total)

            for name, param in self._u_vf.named_parameters():
                self._summary_writer.add_histogram('u_vf/'+name,
                                                   param.data.cpu().numpy(),
                                                   self._n_env_steps_total)
                self._summary_writer.add_histogram('u_vf_grad/'+name,
                                                   param.grad.data.cpu().numpy(),
                                                   self._n_env_steps_total)

            for name, param in self._i_vf.named_parameters():
                self._summary_writer.add_histogram('i_vf/'+name,
                                                   param.data.cpu().numpy(),
                                                   self._n_env_steps_total)
                self._summary_writer.add_histogram('i_vf_grad/'+name,
                                                   param.grad.data.cpu().numpy(),
                                                   self._n_env_steps_total)

            for name, param in self._u_target_vf.named_parameters():
                self._summary_writer.add_histogram('u_vf_target/'+name,
                                                   param.cpu().data.numpy(),
                                                   self._n_env_steps_total)
            for name, param in self._i_target_vf.named_parameters():
                self._summary_writer.add_histogram('i_vf_target/'+name,
                                                   param.cpu().data.numpy(),
                                                   self._n_env_steps_total)

        return (i_policy_loss, i_qf_loss, i_vf_loss,
                accum_u_policy_loss, accum_u_qf_loss, accum_u_vf_loss)

    def _print_soft_statistics(self, update_outputs):
        policy_loss = update_outputs[0]
        i_qf_loss = update_outputs[1]
        i_vf_loss = update_outputs[2]
        u_qf_loss = update_outputs[3]
        u_vf_loss = update_outputs[4]

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()

        # self.eval_statistics['[ACCUM] QF Loss'] = \
        #     np.mean(ptu.get_numpy(i_qf_loss))
        # self.eval_statistics['[ACCUM] VF Loss'] = \
        #     np.mean(ptu.get_numpy(i_vf_loss))
        # self.eval_statistics['[ACCUM] Policy Loss'] = \
        #     np.mean(ptu.get_numpy(policy_loss))

    @property
    def networks(self):
        networks_list = [
            self._policy,
            self._i_qf,
            self._u_qf,
            self._i_vf,
            self._u_vf,
            self._u_target_vf
        ]
        if self._i_target_vf is not None:
            networks_list.append(self._i_target_vf)

        if self._i_qf2 is not None:
            networks_list.append(self._i_qf2)

        if self._u_qf2 is not None:
            networks_list.append(self._u_qf2)

        return networks_list

    @staticmethod
    def _update_v_target_network(vf, target_vf, soft_target_tau):
        ptu.soft_update_from_to(vf, target_vf, soft_target_tau)

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

        snapshot = super(IUWeightedMultiSAC, self).get_epoch_snapshot(epoch)

        snapshot.update(
            policy=self._policy,
            qf=self._i_qf,
            qf2=self._i_qf2,
            vf=self._i_vf,
            target_vf=self._i_target_vf,
            u_qf=self._u_qf,
            u_qf2=self._u_qf,
            u_vf=self._u_vf,
            target_u_vf=self._u_target_vf,
        )

        if self.env.online_normalization or self.env.normalize_obs:
            snapshot.update(
                obs_mean=self.env.obs_mean,
                obs_var=self.env.obs_var,
            )

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
            self.eval_statistics['[U-%02d] Policy Entropy' % uu] = \
                np.nan_to_num(np.mean(self.logging_policy_entropy[:max_step, uu]))
            self.eval_statistics['[U-%02d] Qf Loss' % uu] = \
                np.nan_to_num(np.mean(self.logging_qf_loss[:max_step, uu]))
            self.eval_statistics['[U-%02d] Vf Loss' % uu] = \
                np.nan_to_num(np.mean(self.logging_vf_loss[:max_step, uu]))
            self.eval_statistics['[U-%02d] Pol KL Loss' % uu] = \
                np.nan_to_num(np.mean(self.logging_pol_kl_loss[:max_step, uu]))
            self.eval_statistics['[U-%02d] Rewards' % uu] = \
                np.nan_to_num(np.mean(self.logging_rewards[:max_step, uu]))
            self.eval_statistics['[U-%02d] Mixing Weights' % uu] = \
                np.nan_to_num(np.mean(self.logging_mixing_coeff[:max_step, uu]))

            for aa in range(self.env.action_dim):
                self.eval_statistics['[U-%02d] Policy Std [%02d]' % (uu, aa)] = \
                    np.nan_to_num(np.mean(np.exp(self.logging_policy_log_std[:max_step, aa, uu])))
                self.eval_statistics['[U-%02d] Policy Mean [%02d]' % (uu, aa)] = \
                    np.nan_to_num(np.mean(self.logging_policy_mean[:max_step, aa, uu]))

        # Intentional info
        self.eval_statistics['[I] Policy Entropy'] = \
            np.nan_to_num(np.mean(self.logging_policy_entropy[:max_step, -1]))
        self.eval_statistics['[I] Qf Loss'] = \
            np.nan_to_num(np.mean(self.logging_qf_loss[:max_step, -1]))
        self.eval_statistics['[I] Vf Loss'] = \
            np.nan_to_num(np.mean(self.logging_vf_loss[:max_step, -1]))
        self.eval_statistics['[I] Pol KL Loss'] = \
            np.nan_to_num(np.mean(self.logging_pol_kl_loss[:max_step, -1]))
        self.eval_statistics['[I] Rewards'] = \
            np.nan_to_num(np.mean(self.logging_rewards[:max_step, -1]))
        for aa in range(self.env.action_dim):
            self.eval_statistics['[I] Policy Std [%02d]'] = \
                np.nan_to_num(np.mean(np.exp(self.logging_policy_log_std[:max_step, aa, -1])))
            self.eval_statistics['[I] Policy Mean [%02d]'] = \
                np.nan_to_num(np.mean(self.logging_policy_mean[:max_step, aa, -1]))

    def evaluate(self, epoch):
        statistics = OrderedDict()
        self._update_logging_data()
        statistics.update(self.eval_statistics)
        self.eval_statistics = None

        test_paths = [None for _ in range(self._n_unintentional)]
        for unint_idx in range(self._n_unintentional):
            logger.log("[U-%02d] Collecting samples for evaluation" % unint_idx)
            test_paths[unint_idx] = \
                self.eval_samplers[unint_idx].obtain_samples()

            statistics.update(eval_util.get_generic_path_information(
                test_paths[unint_idx], stat_prefix="[U-%02d] Test" % unint_idx,
            ))

            average_rewards = \
                eval_util.get_average_multigoal_rewards(test_paths[unint_idx],
                                                        unint_idx)
            avg_txt = '[U-%02d] Test AverageReward' % unint_idx
            statistics[avg_txt] = average_rewards * self._u_reward_scales[unint_idx]

            average_returns = \
                eval_util.get_average_multigoal_returns(test_paths[unint_idx],
                                                        unint_idx)
            avg_txt = '[U-%02d] Test AverageReturn' % unint_idx
            statistics[avg_txt] = average_returns * self._u_reward_scales[unint_idx]

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

        self._summary_writer.add_scalar('EvaluationI/avg_return',
                                        statistics['[I] Test AverageReturn'],
                                        self._n_epochs)

        self._summary_writer.add_scalar(
            'EvaluationI/avg_reward',
            statistics['[I] Test Rewards Mean'] * self.reward_scale,
            self._n_epochs
        )

        if hasattr(self.env, "log_diagnostics"):
            # TODO: CHECK ENV LOG_DIAGNOSTICS
            print('TODO: WE NEED LOG_DIAGNOSTICS IN ENV')
            # self.env.log_diagnostics(test_paths[demon])

        # Record the data
        for key, value in statistics.items():
            logger.record_tabular(key, value)

        for u_idx in range(self._n_unintentional):
            if self.render_eval_paths:
                # TODO: CHECK ENV RENDER_PATHS
                print('TODO: RENDER_PATHS')
                pass
                # self.env.render_paths(test_paths[demon])

        if self._epoch_plotter is not None:
            self._epoch_plotter.draw()

        # RESET
        self.logging_policy_entropy[:] = 0
        self.logging_policy_log_std[:] = 0
        self.logging_policy_mean[:] = 0
        self.logging_qf_loss[:] = 0
        self.logging_qf2_loss[:] = 0
        self.logging_vf_loss[:] = 0
        self.logging_pol_kl_loss[:] = 0
        self.logging_rewards[:] = 0
        self.logging_mixing_coeff[:] = 0

    def get_batch(self):
        batch = self.replay_buffer.random_batch(self.batch_size)
        return np_to_pytorch_batch(batch)

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


def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }


def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return Variable(ptu.from_numpy(elem_or_tuple).float(), requires_grad=False)


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


