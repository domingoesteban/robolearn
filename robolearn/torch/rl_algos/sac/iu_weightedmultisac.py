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
from robolearn.core import logger
from robolearn.core import eval_util
from robolearn.utils.samplers import InPlacePathSampler
from robolearn.torch.rl_algos.torch_incremental_rl_algorithm \
    import TorchIncrementalRLAlgorithm
from robolearn.policies.make_deterministic import MakeDeterministic
from robolearn.torch.policies import WeightedMultiPolicySelector
from robolearn.utils.data_management.normalizer import RunningNormalizer

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
            normalize_obs=True,
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

            # optimizer='adam',
            optimizer='rmsprop',
            # optimizer='sgd',
            amsgrad=True,

            i_soft_target_tau=1e-2,
            u_soft_target_tau=1e-2,
            i_target_update_interval=1,
            u_target_update_interval=1,

            u_reward_scales=None,

            save_replay_buffer=False,
            eval_deterministic=True,
            log_tensorboard=False,
            **kwargs
    ):

        # ###### #
        # Models #
        # ###### #

        # Exploration Policy
        self._policy = policy

        # Evaluation Policy
        if eval_deterministic:
            eval_policy = MakeDeterministic(self._policy)
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
            exploration_policy=self._policy,
            eval_env=eval_env,
            eval_policy=eval_policy,
            obs_normalizer=self._obs_normalizer,
            **kwargs
        )

        # Number of Unintentional Tasks (Composable Tasks)
        self._n_unintentional = self._policy.n_heads

        # Evaluation Sampler (One for each unintentional)
        self.eval_u_samplers = [
            InPlacePathSampler(env=env,
                               policy=WeightedMultiPolicySelector(self._policy,
                                                                  idx),
                               total_samples=self.num_steps_per_eval,
                               max_path_length=self.max_path_length,
                               deterministic=True)
            for idx in range(self._n_unintentional)
        ]

        # Important algorithm hyperparameters
        self._reparameterize = reparameterize
        assert self._reparameterize == self._policy.reparameterize
        self._action_prior = action_prior
        self._i_entropy_scale = i_entropy_scale
        if u_entropy_scale is None:
            u_entropy_scale = [i_entropy_scale
                               for _ in range(self._n_unintentional)]
        self._u_entropy_scale = ptu.FloatTensor(u_entropy_scale)

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
        self._u_vf_criterion = nn.MSELoss()
        self._i_qf_criterion = nn.MSELoss()
        self._i_vf_criterion = nn.MSELoss()

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
        if self._u_qf2 is None:
            self._u_qf2_optimizer = None
        else:
            self._u_qf2_optimizer = optimizer_class(
                self._u_qf2.parameters(),
                lr=u_qf2_lr,
                weight_decay=u_q_weight_decay,
                **optimizer_params
            )
        if self._i_qf2 is None:
            self._i_qf2_optimizer = None
        else:
            self._i_qf2_optimizer = optimizer_class(
                self._i_qf2.parameters(),
                lr=i_qf2_lr,
                weight_decay=i_q_weight_decay,
                **optimizer_params
            )

        # V-function optimizers
        self._u_vf_optimizer = optimizer_class(
            self._u_vf.parameters(),
            lr=u_vf_lr,
            weight_decay=u_v_weight_decay,
            **optimizer_params
        )
        self._i_vf_optimizer = optimizer_class(
            self._i_vf.parameters(),
            lr=i_vf_lr,
            weight_decay=i_v_weight_decay,
            **optimizer_params
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
            weight_decay=i_policy_weight_decay,
            **optimizer_params
        )
        self._mixing_optimizer = optimizer_class(
            chain(self._policy.shared_parameters(),
                  self._policy.mixing_parameters()),
            lr=u_mixing_lr,
            weight_decay=i_policy_weight_decay,
            **optimizer_params
        )
        self._policies_optimizer = optimizer_class(
            chain(self._policy.shared_parameters(),
                  self._policy.policies_parameters()),
            lr=u_policies_lr,
            weight_decay=u_policy_weight_decay,
            **optimizer_params
        )

        # Policy regularization coefficients (weights)
        self._i_policy_mean_regu_weight = i_policy_mean_regu_weight
        self._i_policy_std_regu_weight = i_policy_std_regu_weight
        self._i_policy_pre_activation_weight = i_policy_pre_activation_weight
        self._i_policy_mixing_coeff_weight = i_policy_mixing_coeff_weight

        if u_policy_mean_regu_weight is None:
            u_policy_mean_regu_weight = [i_policy_mean_regu_weight
                                         for _ in range(self._n_unintentional)]
        self._u_policy_mean_regu_weight = \
            ptu.FloatTensor(u_policy_mean_regu_weight)
        if u_policy_std_regu_weight is None:
            u_policy_std_regu_weight = [i_policy_std_regu_weight
                                        for _ in range(self._n_unintentional)]
        self._u_policy_std_regu_weight = \
            ptu.FloatTensor(u_policy_std_regu_weight)
        if u_policy_pre_activation_weight is None:
            u_policy_pre_activation_weight = [i_policy_pre_activation_weight
                                       for _ in range(self._n_unintentional)]
        self._u_policy_pre_activation_weight = \
            ptu.FloatTensor(u_policy_pre_activation_weight)

        # Useful Variables for logging
        self.logging_pol_kl_loss = np.zeros((self.num_train_steps_per_epoch,
                                             self._n_unintentional + 1))
        self.logging_qf_loss = np.zeros((self.num_train_steps_per_epoch,
                                         self._n_unintentional + 1))
        self.logging_qf2_loss = np.zeros((self.num_train_steps_per_epoch,
                                         self._n_unintentional + 1))
        self.logging_vf_loss = np.zeros((self.num_train_steps_per_epoch,
                                         self._n_unintentional + 1))
        self.logging_rewards = np.zeros((self.num_train_steps_per_epoch,
                                         self._n_unintentional + 1))
        self.logging_policy_entropy = np.zeros((self.num_train_steps_per_epoch,
                                                self._n_unintentional + 1))
        self.logging_policy_log_std = np.zeros((self.num_train_steps_per_epoch,
                                                self._n_unintentional + 1,
                                                self.env.action_dim,
                                                ))
        self.logging_policy_mean = np.zeros((self.num_train_steps_per_epoch,
                                             self._n_unintentional + 1,
                                             self.env.action_dim,
                                             ))
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

        # ########################## #
        # Unintentional Critics Step #
        # ########################## #
        u_rewards = \
            (batch['reward_vectors'] * self._u_reward_scales).unsqueeze(-1)
        u_terminals = (batch['terminal_vectors']).unsqueeze(-1)

        u_v_value_next = torch.cat([vv.unsqueeze(1)
                                    for vv in self._u_target_vf(next_obs)[0]],
                                   dim=1)

        u_q_pred = torch.cat([qq.unsqueeze(1) for qq in self._u_qf(obs, actions)[0]],
                             dim=1)

        # Calculate QF Loss (Soft Bellman Eq.)
        u_q_target = u_rewards + (1. - u_terminals) * self.discount * u_v_value_next
        u_qf_loss = 0.5*torch.mean((u_q_pred - u_q_target.detach())**2,
                                   dim=0).squeeze(-1)
        total_u_qf_loss = torch.sum(u_qf_loss)

        # Update Unintentional Q-values
        self._u_qf_optimizer.zero_grad()
        total_u_qf_loss.backward()
        self._u_qf_optimizer.step()

        if self._i_qf2 is not None:
            u_q2_pred = [self._u_qf2(obs, actions)[0][uu]
                         for uu in range(self._n_unintentional)]
            u_q2_pred = torch.cat([qq.unsqueeze(1) for qq in u_q2_pred],
                                  dim=1)

            # Calculate QF2 Loss (Soft Bellman Eq.)
            u_qf2_loss = 0.5*torch.mean((u_q2_pred - u_q_target.detach())**2,
                                        dim=0).squeeze(-1)
            total_u_qf2_loss = torch.sum(u_qf2_loss)

            # Update Unintentional Q2-values
            self._u_qf2_optimizer.zero_grad()
            total_u_qf2_loss.backward()
            self._u_qf2_optimizer.step()

        # ####################### #
        # Intentional Critic Step #
        # ####################### #
        i_rewards = batch['rewards']# * self.reward_scale
        i_terminals = batch['terminals']

        i_v_value_next = self._i_target_vf(next_obs)[0]
        i_q_pred = self._i_qf(obs, actions)[0]

        # Calculate QF Loss (Soft Bellman Eq.)
        i_q_target = i_rewards + (1. - i_terminals) * self.discount * i_v_value_next
        i_qf_loss = 0.5*self._i_qf_criterion(i_q_pred, i_q_target.detach())

        # Update Intentional Q-value
        self._i_qf_optimizer.zero_grad()
        i_qf_loss.backward()
        self._i_qf_optimizer.step()

        if self._i_qf2 is not None:
            i_q2_pred = self._i_qf2(obs, actions)[0]

            # Calculate QF2 Loss (Soft Bellman Eq.)
            i_qf2_loss = 0.5*self._i_qf_criterion(i_q2_pred, i_q_target.detach())

            # Update Intentional Q2-value
            self._i_qf2_optimizer.zero_grad()
            i_qf2_loss.backward()
            self._i_qf2_optimizer.step()

        # ############### #
        # Get Policy Info #
        # ############### #
        i_new_actions, policy_info = self._policy(obs, deterministic=False,
                                                  return_log_prob=True,
                                                  pol_idx=None,
                                                  optimize_policies=False)
        i_log_pi = policy_info['log_prob'] * self._i_entropy_scale
        i_policy_mean = policy_info['mean']
        i_policy_log_std = policy_info['log_std']
        i_pre_tanh_value = policy_info['pre_tanh_value']
        mixing_coeff = policy_info['mixing_coeff']

        u_new_actions = policy_info['pol_actions']
        u_log_pi = policy_info['pol_log_probs'] \
            * self._u_entropy_scale.unsqueeze(1)
        u_policy_mean = policy_info['pol_means']
        u_policy_log_std = policy_info['pol_log_stds']
        u_pre_tanh_value = policy_info['pol_pre_tanh_values']

        # ################### #
        # Unintentional Actor #
        # ################### #
        if self._action_prior == 'normal':
            raise NotImplementedError
        else:
            u_policy_prior_log_probs = 0.0  # Uniform prior

        # Get Unintentional V(s_t)
        u_v_pred = torch.cat([vv.unsqueeze(1) for vv in self._u_vf(obs)[0]],
                             dim=1)

        # Get Unintentional Q1(s_t, a_t)
        u_q1_new_actions = [self._u_qf(obs, u_new_actions[:, uu, :])[0][uu]
                            for uu in range(self._n_unintentional)]
        u_q1_new_actions = torch.cat([qq.unsqueeze(1)
                                      for qq in u_q1_new_actions],
                                     dim=1)
        if self._u_qf2 is not None:
            # Get Unintentional Q2(s_t, a_t)
            u_q2_new_actions = [self._u_qf2(obs, u_new_actions[:, uu, :])[0][uu]
                                for uu in range(self._n_unintentional)]
            u_q2_new_actions = torch.cat([qq.unsqueeze(1)
                                          for qq in u_q2_new_actions],
                                         dim=1)
            # Get Minimum Unintentional Q(s_t, a_t)
            u_q_new_actions = torch.min(u_q1_new_actions, u_q2_new_actions)
        else:
            u_q_new_actions = u_q1_new_actions

        # Get Unintentional A(s_t, a_t)
        u_advantage_new_actions = u_q_new_actions - u_v_pred.detach()

        # Get Unintentional Policies KL loss: - (E_a[Q(s_t, a_t_) - H(.)])
        if self._reparameterize:
            u_policy_kl_loss = -torch.mean(u_q_new_actions - u_log_pi,
                                           dim=0).squeeze(-1)
            # u_policy_kl_loss = -torch.mean(u_advantage_new_actions - u_log_pi,
            #                                dim=0).squeeze(-1)
        else:
            u_policy_kl_loss = (
                    u_log_pi * (u_log_pi - u_q_new_actions + u_v_pred
                                - u_policy_prior_log_probs).detach()
            ).mean(dim=0).squeeze(-1)

        # Get Unintentional Policies regularization loss
        u_mean_reg_loss = self._u_policy_mean_regu_weight * \
            (u_policy_mean ** 2).mean(dim=0).mean(dim=-1)
        u_std_reg_loss = self._u_policy_std_regu_weight * \
            (u_policy_log_std ** 2).mean(dim=0).mean(dim=-1)
        u_pre_activation_reg_loss = \
            self._u_policy_pre_activation_weight * \
            (u_pre_tanh_value**2).sum(dim=-1).mean(dim=0).mean(dim=-1)
        u_policy_regu_loss = (u_mean_reg_loss + u_std_reg_loss
                              + u_pre_activation_reg_loss)

        # Get Unintentional Policies Total loss
        u_policy_loss = (u_policy_kl_loss + u_policy_regu_loss*0)
        total_u_policy_loss = torch.sum(u_policy_loss)

        # Update Unintentional Policies
        self._policy_optimizer.zero_grad()
        total_u_policy_loss.backward()
        self._policy_optimizer.step()
        # self._policies_optimizer.zero_grad()
        # total_u_policy_loss.backward()
        # self._policies_optimizer.step()

        # ############################# #
        # Unintentional V-function Step #
        # ############################# #
        # Calculate Unintentional Vf Loss
        u_v_target = u_q_new_actions - u_log_pi + u_policy_prior_log_probs
        u_vf_loss = 0.5*torch.mean((u_v_pred - u_v_target.detach())**2,
                                   dim=0).squeeze(-1)
        total_u_vf_loss = torch.sum(u_vf_loss)

        # Update Unintentional V-value functions
        self._u_vf_optimizer.zero_grad()
        total_u_vf_loss.backward()
        self._u_vf_optimizer.step()

        # Update Unintentional V Target Networks
        if self._n_train_steps_total % self._u_target_update_interval == 0:
            self._update_v_target_network(
                vf=self._u_vf,
                target_vf=self._u_target_vf,
                soft_target_tau=self._u_soft_target_tau
            )

        # ########################### #
        # Intentional Actor & Vf Step #
        # ########################### #
        if self._action_prior == 'normal':
            raise NotImplementedError
        else:
            i_policy_prior_log_probs = 0.0  # Uniform prior

        # Get Intentional V(s_t)
        i_v_pred = self._i_vf(obs)[0]

        # Get Intentional Q1(s_t, a_t)
        i_q1_new_actions = self._i_qf(obs, i_new_actions)[0]

        if self._i_qf2 is not None:
            # Get Intentional Q2(s_t, a_t)
            i_q2_new_actions = self._i_qf2(obs, i_new_actions)[0]
            # Get Minimum Intentional Q(s_t, a_t)
            i_q_new_actions = torch.min(i_q1_new_actions, i_q2_new_actions)
        else:
            i_q_new_actions = i_q1_new_actions

        # Get Intentional A(s_t, a_t)
        i_advantage_new_actions = i_q_new_actions - i_v_pred.detach()

        # Get Intentional KL loss: - (E_a[Q(s_t, a_t_) - H(.)])
        if self._reparameterize:
            i_policy_kl_loss = -torch.mean(i_q_new_actions - i_log_pi)
            # i_policy_kl_loss = -torch.mean(i_advantage_new_actions - i_log_pi)
        else:
            i_policy_kl_loss = (
                    i_log_pi * (i_log_pi - i_q_new_actions + i_v_pred
                                - i_policy_prior_log_probs).detach()
            ).mean()

        # Get Intentional Policy regularization loss
        i_mean_reg_loss = self._i_policy_mean_regu_weight * \
            (i_policy_mean ** 2).mean()
        i_std_reg_loss = self._i_policy_std_regu_weight * \
            (i_policy_log_std ** 2).mean()
        i_pre_activation_reg_loss = \
            self._i_policy_pre_activation_weight * \
            (i_pre_tanh_value**2).sum(dim=-1).mean()
        mixing_coeff_loss = self._i_policy_mixing_coeff_weight * \
            (mixing_coeff ** 2).sum(dim=-1).mean()  # TODO: CHECK THIS
        i_policy_regu_loss = (i_mean_reg_loss + i_std_reg_loss
                              + i_pre_activation_reg_loss + mixing_coeff_loss)

        # Get Intentional Policy Total loss
        i_policy_loss = (i_policy_kl_loss + i_policy_regu_loss)

        # Update Intentional Policies
        self._policy_optimizer.zero_grad()
        # i_policy_loss.backward()
        # self._policy_optimizer.step()
        self._mixing_optimizer.zero_grad()
        i_policy_loss.backward()
        self._mixing_optimizer.step()

        # ########################### #
        # Intentional V-function Step #
        # ########################### #
        # Calculate Intentional Vf Loss
        i_v_target = i_q_new_actions - i_log_pi + i_policy_prior_log_probs
        i_vf_loss = 0.5*self._i_vf_criterion(i_v_pred, i_v_target.detach())

        # Update Intentional V-value function
        self._i_vf_optimizer.zero_grad()
        i_vf_loss.backward()
        self._i_vf_optimizer.step()

        # Update Intentional V Target Network
        if self._n_train_steps_total % self._i_target_update_interval == 0:
            self._update_v_target_network(
                vf=self._i_vf,
                target_vf=self._i_target_vf,
                soft_target_tau=self._i_soft_target_tau
            )

        # ############### #
        # LOG Useful Data #
        # ############### #
        self.logging_policy_entropy[step_idx, :-1] = \
            ptu.get_numpy(-u_log_pi.mean(dim=0).squeeze(-1))
        self.logging_policy_entropy[step_idx, -1] = \
            ptu.get_numpy(-i_log_pi.mean(dim=0))

        self.logging_policy_log_std[step_idx, :-1, :] = \
            ptu.get_numpy(u_policy_log_std.mean(dim=0))
        self.logging_policy_log_std[step_idx, -1, :] = \
            ptu.get_numpy(i_policy_log_std.mean(dim=0))

        self.logging_policy_mean[step_idx, :-1, :] = \
            ptu.get_numpy(u_policy_mean.mean(dim=0))
        self.logging_policy_mean[step_idx, -1, :] = \
            ptu.get_numpy(i_policy_mean.mean(dim=0))

        self.logging_pol_kl_loss[step_idx, :-1] = \
            ptu.get_numpy(u_policy_kl_loss)
        self.logging_pol_kl_loss[step_idx, -1] = \
            ptu.get_numpy(i_policy_kl_loss)

        self.logging_qf_loss[step_idx, :-1] = ptu.get_numpy(u_qf_loss)
        self.logging_qf_loss[step_idx, -1] = ptu.get_numpy(i_qf_loss)

        if self._u_qf2 is not None:
            self.logging_qf2_loss[step_idx, :-1] = ptu.get_numpy(u_qf2_loss)
        if self._i_qf2 is not None:
            self.logging_qf2_loss[step_idx, -1] = ptu.get_numpy(i_qf2_loss)

        self.logging_vf_loss[step_idx, :-1] = \
            ptu.get_numpy(u_vf_loss)
        self.logging_vf_loss[step_idx, -1] = \
            ptu.get_numpy(i_vf_loss)

        self.logging_rewards[step_idx, :-1] = \
            ptu.get_numpy(u_rewards.mean(dim=0).squeeze(-1))
        self.logging_rewards[step_idx, -1] = \
            ptu.get_numpy(i_rewards.mean(dim=0).squeeze(-1))

        self.logging_mixing_coeff[step_idx, :, :] = \
            ptu.get_numpy(mixing_coeff.mean(dim=0))

        if self._log_tensorboard:
            self._summary_writer.add_scalar(
                'TrainingI/qf_loss',
                ptu.get_numpy(i_qf_loss),
                self._n_env_steps_total
            )
            if self._i_qf2 is not None:
                self._summary_writer.add_scalar(
                    'TrainingI/qf2_loss',
                    ptu.get_numpy(i_qf2_loss),
                    self._n_env_steps_total
                )
            self._summary_writer.add_scalar(
                'TrainingI/vf_loss',
                ptu.get_numpy(i_vf_loss),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'TrainingI/avg_reward',
                ptu.get_numpy(i_rewards.mean()),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'TrainingI/avg_advantage',
                ptu.get_numpy(i_advantage_new_actions.mean()),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'TrainingI/policy_loss',
                ptu.get_numpy(i_policy_loss),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'TrainingI/policy_entropy',
                ptu.get_numpy(-i_log_pi.mean()),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'TrainingI/policy_mean',
                ptu.get_numpy(i_policy_mean.mean()),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'TrainingI/policy_std',
                np.exp(ptu.get_numpy(i_policy_log_std.mean())),
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

        snapshot = TorchIncrementalRLAlgorithm.get_epoch_snapshot(self, epoch)

        snapshot.update(
            policy=self._policy,
            qf=self._i_qf,
            qf2=self._i_qf2,
            vf=self._i_vf,
            target_vf=self._i_target_vf,
            u_qf=self._u_qf,
            u_qf2=self._u_qf2,
            u_vf=self._u_vf,
            target_u_vf=self._u_target_vf,
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
                    np.nan_to_num(np.mean(np.exp(self.logging_policy_log_std[:max_step, uu, aa])))
                self.eval_statistics['[U-%02d] Policy Mean [%02d]' % (uu, aa)] = \
                    np.nan_to_num(np.mean(self.logging_policy_mean[:max_step, uu, aa]))

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
                np.nan_to_num(np.mean(np.exp(self.logging_policy_log_std[:max_step, -1, aa])))
            self.eval_statistics['[I] Policy Mean [%02d]'] = \
                np.nan_to_num(np.mean(self.logging_policy_mean[:max_step, -1, aa]))

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
        self.logging_policy_entropy.fill(0)
        self.logging_policy_log_std.fill(0)
        self.logging_policy_mean.fill(0)
        self.logging_qf_loss.fill(0)
        self.logging_qf2_loss.fill(0)
        self.logging_vf_loss.fill(0)
        self.logging_pol_kl_loss.fill(0)
        self.logging_rewards.fill(0)
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
