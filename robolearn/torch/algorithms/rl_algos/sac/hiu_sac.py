"""
Based on Pong's SAC implementation

https://github.com/vitchyr/rlkit
"""

import numpy as np
import torch
import torch.optim as optim

from collections import OrderedDict
from itertools import chain

import robolearn.torch.utils.pytorch_util as ptu
from robolearn.utils.logging import logger
from robolearn.utils import eval_util
from robolearn.utils.samplers import InPlacePathSampler

from robolearn.algorithms.rl_algos import IncrementalRLAlgorithm
from robolearn.torch.algorithms.torch_algorithm import TorchAlgorithm

from robolearn.models.policies import MakeDeterministic
from robolearn.torch.policies import WeightedMultiPolicySelector
from robolearn.utils.data_management.normalizer import RunningNormalizer

import tensorboardX

# MAX_LOG_ALPHA = 9.21034037  # Alpha=10000  Before 01/07
MAX_LOG_ALPHA = 6.2146080984  # Alpha=500  From 09/07


class HIUSAC(IncrementalRLAlgorithm, TorchAlgorithm):
    """
    Hierarchical Intentional-Unintentional Soft Actor Critic (HIU-SAC).
    Incremental Version.
    """
    def __init__(
            self,
            env,
            policy,
            u_qf1,

            replay_buffer,
            batch_size=1024,
            normalize_obs=False,
            eval_env=None,

            i_qf1=None,
            u_qf2=None,
            i_qf2=None,
            i_vf=None,
            u_vf=None,
            action_prior='uniform',

            i_entropy_scale=1.,
            u_entropy_scale=None,
            auto_alphas=True,
            i_tgt_entro=None,
            u_tgt_entros=None,

            policy_lr=3e-4,
            qf_lr=3e-4,

            i_policy_mean_regu_weight=1e-3,
            i_policy_std_regu_weight=1e-3,
            i_policy_pre_activation_weight=0.,
            i_policy_mixing_coeff_weight=1e-3,

            u_policy_mean_regu_weight=None,
            u_policy_std_regu_weight=None,
            u_policy_pre_activation_weight=None,

            policy_weight_decay=0.,
            q_weight_decay=0.,

            optimizer='adam',
            # optimizer='rmsprop',
            # optimizer='sgd',
            optimizer_kwargs=None,

            i_soft_target_tau=5e-3,
            u_soft_target_tau=5e-3,
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

        IncrementalRLAlgorithm.__init__(
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
            InPlacePathSampler(
                env=env,
                policy=WeightedMultiPolicySelector(self._policy, idx),
                total_samples=self.num_steps_per_eval,
                max_path_length=self.max_path_length,
                deterministic=True,
            )
            for idx in range(self._n_unintentional)
        ]

        # Intentional (Main Task) Q-functions
        self._i_qf1 = i_qf1
        self._i_qf2 = i_qf2
        if i_vf is None:
            self._i_vf = None
            self._i_target_vf = None
            self._i_target_qf1 = self._i_qf1.copy()
            self._i_target_qf2 = \
                None if self._i_qf2 is None else self._i_qf2.copy()
        else:
            self._i_vf = i_vf
            self._i_target_vf = self._i_vf.copy()
            self._i_target_qf1 = None
            self._i_target_qf2 = None

        # Unintentional (Composable Tasks) Q-functions
        self._u_qf1 = u_qf1
        self._u_qf2 = u_qf2
        if u_vf is None:
            self._u_vf = None
            self._u_target_vf = None
            self._u_target_qf1 = self._u_qf1.copy()
            self._u_target_qf2 = self._u_qf2.copy()
        else:
            self._u_vf = u_vf
            self._u_target_vf = self._u_vf.copy()
            self._u_target_qf1 = None
            self._u_target_qf2 = None

        # Replay Buffer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.save_replay_buffer = save_replay_buffer

        # Soft-update rate for target V-functions
        self._i_soft_target_tau = i_soft_target_tau
        self._u_soft_target_tau = u_soft_target_tau
        self._i_target_update_interval = i_target_update_interval
        self._u_target_update_interval = u_target_update_interval

        # Important algorithm hyperparameters
        self._action_prior = action_prior
        self._i_entropy_scale = i_entropy_scale
        if u_entropy_scale is None:
            u_entropy_scale = [i_entropy_scale
                               for _ in range(self._n_unintentional)]
        self._u_entropy_scale = torch.tensor(u_entropy_scale, device=ptu.device)

        # Desired Alphas
        self._auto_alphas = auto_alphas
        if i_tgt_entro is None:
            i_tgt_entro = -env.action_dim
        self._i_tgt_entro = ptu.tensor([i_tgt_entro], device=ptu.device)
        if u_tgt_entros is None:
            u_tgt_entros = [i_tgt_entro for _ in range(self._n_unintentional)]
        self._u_tgt_entros = torch.tensor(u_tgt_entros, device=ptu.device)
        self._u_log_alphas = torch.zeros(self._n_unintentional,
                                         device=ptu.device, requires_grad=True)
        self._i_log_alpha = torch.zeros(1, device=ptu.device, requires_grad=True)

        # Unintentional Reward Scales
        if u_reward_scales is None:
            reward_scale = kwargs['reward_scale']
            u_reward_scales = [reward_scale
                               for _ in range(self._n_unintentional)]
        self._u_reward_scales = torch.tensor(u_reward_scales, device=ptu.device)

        # ########## #
        # Optimizers #
        # ########## #
        if optimizer.lower() == 'adam':
            optimizer_class = optim.Adam
            if optimizer_kwargs is None:
                optimizer_kwargs = dict(
                    amsgrad=True,
                    # amsgrad=False,
                )
        elif optimizer.lower() == 'rmsprop':
            optimizer_class = optim.RMSprop
            if optimizer_kwargs is None:
                optimizer_kwargs = dict(

                )
        else:
            raise ValueError('Wrong optimizer')

        # Values optimizer
        vals_params_list = [self._u_qf1.parameters(), self._i_qf1.parameters()]
        if self._u_qf2 is not None:
            vals_params_list.append(self._u_qf2.parameters())
        if self._i_qf2 is not None:
            vals_params_list.append(self._i_qf2.parameters())
        if self._u_vf is not None:
            vals_params_list.append(self._u_vf.parameters())
        if self._i_vf is not None:
            vals_params_list.append(self._i_vf.parameters())
        vals_params = chain(*vals_params_list)

        self._values_optimizer = optimizer_class(
            vals_params,
            lr=qf_lr,
            weight_decay=q_weight_decay,
            **optimizer_kwargs
        )

        # Policy optimizer
        self._policy_optimizer = optimizer_class(
            self._policy.parameters(),
            lr=policy_lr,
            weight_decay=policy_weight_decay,
            **optimizer_kwargs
        )

        # Alpha optimizers
        self._alphas_optimizer = optimizer_class(
            [self._u_log_alphas, self._i_log_alpha],
            lr=policy_lr,
            **optimizer_kwargs
        )

        # Weights for policy regularization coefficients
        self._i_pol_mean_regu_weight = i_policy_mean_regu_weight
        self._i_pol_std_regu_weight = i_policy_std_regu_weight
        self._i_pol_pre_activ_weight = i_policy_pre_activation_weight
        self._i_pol_mixing_coeff_weight = i_policy_mixing_coeff_weight

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
            u_policy_pre_activation_weight = [
                i_policy_pre_activation_weight
                for _ in range(self._n_unintentional)
            ]
        self._u_policy_pre_activ_weight = \
            ptu.FloatTensor(u_policy_pre_activation_weight)

        # Useful Variables for logging
        self.log_data = dict()
        self.log_data['Pol KL Loss'] = np.zeros((
            self.num_train_steps_per_epoch,
            self._n_unintentional + 1,
        ))
        self.log_data['Qf Loss'] = np.zeros((
            self.num_train_steps_per_epoch,
            self._n_unintentional + 1,
        ))
        self.log_data['Qf2 Loss'] = np.zeros((
            self.num_train_steps_per_epoch,
            self._n_unintentional + 1,
        ))
        self.log_data['Vf Loss'] = np.zeros((
            self.num_train_steps_per_epoch,
            self._n_unintentional + 1,
        ))
        self.log_data['Rewards'] = np.zeros((
            self.num_train_steps_per_epoch,
            self._n_unintentional + 1,
        ))
        self.log_data['Policy Entropy'] = np.zeros((
            self.num_train_steps_per_epoch,
            self._n_unintentional + 1,
        ))
        self.log_data['Policy Mean'] = np.zeros((
            self.num_train_steps_per_epoch,
            self._n_unintentional + 1,
            self.env.action_dim,
        ))
        self.log_data['Pol Log Std'] = np.zeros((
            self.num_train_steps_per_epoch,
            self._n_unintentional + 1,
            self.env.action_dim,
        ))
        self.log_data['Mixing Weights'] = np.zeros((
            self.num_train_steps_per_epoch,
            self._n_unintentional,
            self.env.action_dim,
        ))
        self.log_data['Alphas'] = np.zeros((
            self.num_train_steps_per_epoch,
            self._n_unintentional + 1,
        ))

        # Tensorboard-like Logging
        self._log_tensorboard = log_tensorboard
        if log_tensorboard:
            self._summary_writer = \
                tensorboardX.SummaryWriter(log_dir=logger.get_snapshot_dir())
        else:
            self._summary_writer = None

    def pretrain(self, n_pretrain_samples):
        # We do not require any pretrain (I think...)
        observation = self.env.reset()
        for ii in range(n_pretrain_samples):
            action = self.env.action_space.sample()
            # Interact with environment
            next_ob, reward, terminal, env_info = (
                self.env.step(action)
            )
            agent_info = None

            # Increase counter
            self._n_env_steps_total += 1
            # Create np.array of obtained terminal and reward
            reward = reward * self.reward_scale
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

        # ####################### #
        # Get All Obs Policy Info #
        # ####################### #

        # One pass for both s and s' instead of two
        obs_combined = torch.cat((obs, next_obs), dim=0)
        i_all_actions, policy_info = self._policy(
            obs_combined,
            deterministic=False,
            return_log_prob=True,
            pol_idx=None,
            optimize_policies=False,
        )
        # Intentional policy info
        i_new_actions = i_all_actions[:self.batch_size]
        i_next_actions = i_all_actions[self.batch_size:].detach()

        i_new_log_pi = policy_info['log_prob'][:self.batch_size]
        i_next_log_pi = policy_info['log_prob'][self.batch_size:].detach()

        i_new_policy_mean = policy_info['mean'][:self.batch_size]
        i_new_policy_log_std = policy_info['log_std'][:self.batch_size]
        i_new_pre_tanh_value = policy_info['pre_tanh_value'][:self.batch_size]
        new_mixing_coeff = policy_info['mixing_coeff'][:self.batch_size]

        # Unintentional policy info
        u_new_actions = policy_info['pol_actions'][:self.batch_size]
        u_new_log_pi = policy_info['pol_log_probs'][:self.batch_size]
        u_new_policy_mean = policy_info['pol_means'][:self.batch_size]
        u_new_policy_log_std = policy_info['pol_log_stds'][:self.batch_size]
        u_new_pre_tanh_value = policy_info['pol_pre_tanh_values'][:self.batch_size]

        u_next_actions = policy_info['pol_actions'][self.batch_size:].detach()
        u_next_log_pi = policy_info['pol_log_probs'][self.batch_size:].detach()

        # Alphas
        ialpha = self._i_entropy_scale*torch.clamp(self._i_log_alpha,
                                                   max=MAX_LOG_ALPHA).exp()
        ualphas = (self._u_entropy_scale*torch.clamp(self._u_log_alphas,
                                                     max=MAX_LOG_ALPHA).exp()
                   ).unsqueeze(1)

        # ########################## #
        # Unintentional Critics Step #
        # ########################## #
        u_rewards = \
            (batch['reward_vectors'] * self._u_reward_scales).unsqueeze(-1)
        u_terminals = (batch['terminal_vectors']).unsqueeze(-1)

        # Unintentional Q1(s', a')
        u_next_q1 = torch.cat(
            [
             self._u_target_qf1(next_obs, u_next_actions[:, uu, :])[0][uu].unsqueeze(1)
             for uu in range(self._n_unintentional)
             ],
            dim=1
        )
        if self._u_target_qf2 is not None:
            # Unintentional Q2(s', a')
            u_next_q2 = torch.cat(
                [
                 self._u_target_qf2(next_obs, u_next_actions[:, uu, :])[0][uu].unsqueeze(1)
                 for uu in range(self._n_unintentional)
                ],
                dim=1
            )
            # Minimum Unintentional Double-Q
            u_next_q = torch.min(u_next_q1, u_next_q2)
        else:
            u_next_q = u_next_q1

        # Unintentional Vtarget(s')
        u_next_v = u_next_q - ualphas*u_next_log_pi

        # Calculate Bellman Backup for Unintentional Q-values
        u_q_backup = u_rewards + (1. - u_terminals) * self.discount * u_next_v
        u_q_backup = u_q_backup.detach()

        # Unintentional Q1(s,a)
        u_q_pred = torch.cat([qq.unsqueeze(1)
                              for qq in self._u_qf1(obs, actions)[0]],
                             dim=1)

        # Unintentional QF1 Losses: Mean Squared Bellman Equation (MSBE)
        u_qf1_loss = \
            0.5*torch.mean((u_q_pred - u_q_backup)**2, dim=0).squeeze(-1)
        # MSBE Q1-Loss for all unintentional policies.
        total_u_qf1_loss = torch.sum(u_qf1_loss)

        if self._u_qf2 is not None:
            # Unintentional Q2(s,a)
            u_q2_pred = torch.cat([qq.unsqueeze(1)
                                   for qq in self._u_qf2(obs, actions)[0]],
                                  dim=1)

            # Unintentional QF2 Losses: Mean Squared Bellman Equation (MSBE)
            u_qf2_loss = 0.5*torch.mean((u_q2_pred - u_q_backup)**2,
                                        dim=0).squeeze(-1)
            # MSBE Q2-Loss for all unintentional policies.
            total_u_qf2_loss = torch.sum(u_qf2_loss)
        else:
            u_qf2_loss = 0
            total_u_qf2_loss = 0

        # ####################### #
        # Intentional Critic Step #
        # ####################### #
        i_rewards = batch['rewards'] * self.reward_scale
        i_terminals = batch['terminals']

        # Intentional Q1(s', a')
        i_next_q1 = self._i_target_qf1(next_obs, i_next_actions)[0]

        if self._i_target_qf2 is not None:
            # Intentional Q2(s', a')
            i_next_q2 = self._i_target_qf2(next_obs, i_next_actions)[0]

            # Minimum Unintentional Double-Q
            i_next_q = torch.min(i_next_q1, i_next_q2)
        else:
            i_next_q = i_next_q1

        # Intentional Vtarget(s')
        i_next_v = i_next_q - ialpha*i_next_log_pi

        # Calculate Bellman Backup for Intentional Q-value
        i_q_backup = i_rewards + (1. - i_terminals) * self.discount * i_next_v

        # Intentional Q1(s,a)
        i_q_pred = self._i_qf1(obs, actions)[0]

        # Intentional QF1 Loss: Mean Squared Bellman Equation (MSBE)
        i_qf1_loss = 0.5*torch.mean((i_q_backup.detach() - i_q_pred)**2)

        if self._i_qf2 is not None:
            # Intentional Q2(s,a)
            i_q2_pred = self._i_qf2(obs, actions)[0]

            # Intentional QF2 Loss: Mean Squared Bellman Equation (MSBE)
            i_qf2_loss = 0.5*torch.mean((i_q_backup.detach() - i_q2_pred)**2)
        else:
            i_qf2_loss = 0

        # #################### #
        # Unintentional Actors #
        # #################### #
        if self._action_prior == 'normal':
            raise NotImplementedError
        else:
            u_policy_prior_log_probs = 0.0  # Uniform prior

        # Unintentional Q1(s, a)
        u_q1_new_actions = torch.cat(
            [self._u_qf1(obs, u_new_actions[:, uu, :])[0][uu].unsqueeze(1)
             for uu in range(self._n_unintentional)
             ],
            dim=1
        )
        if self._u_qf2 is not None:
            # Unintentional Q2(s, a)
            u_q2_new_actions = torch.cat(
                [self._u_qf2(obs, u_new_actions[:, uu, :])[0][uu].unsqueeze(1)
                 for uu in range(self._n_unintentional)
                 ],
                dim=1
            )
            # Minimum Unintentional Double-Q
            u_q_new_actions = torch.min(u_q1_new_actions, u_q2_new_actions)
        else:
            u_q_new_actions = u_q1_new_actions

        # # Get Unintentional A(s, a)
        # u_advantage_new_actions = u_q_new_actions - u_v_pred.detach()

        # Get Unintentional Policies KL loss: - (E_a[Q(s, a) + H(.)])
        u_policy_kl_loss = -torch.mean(
            u_q_new_actions - ualphas*u_new_log_pi
            + u_policy_prior_log_probs,
            dim=0
        ).squeeze(-1)
        # u_policy_kl_loss = -torch.mean(
        #     u_advantage_new_actions - u_log_pi*ualphas,
        #     dim=0).squeeze(-1)

        # Get Unintentional Policies regularization loss
        u_mean_reg_loss = self._u_policy_mean_regu_weight * \
            (u_new_policy_mean ** 2).mean(dim=0).mean(dim=-1)
        u_std_reg_loss = self._u_policy_std_regu_weight * \
            (u_new_policy_log_std ** 2).mean(dim=0).mean(dim=-1)
        u_pre_activation_reg_loss = \
            self._u_policy_pre_activ_weight * \
            (u_new_pre_tanh_value**2).sum(dim=-1).mean(dim=0).mean(dim=-1)
        u_policy_regu_loss = (u_mean_reg_loss + u_std_reg_loss
                              + u_pre_activation_reg_loss)

        # Get Unintentional Policies Total loss
        u_policy_loss = (u_policy_kl_loss + u_policy_regu_loss)
        total_u_policy_loss = torch.sum(u_policy_loss)

        # ################# #
        # Intentional Actor #
        # ################# #
        if self._action_prior == 'normal':
            raise NotImplementedError
        else:
            i_policy_prior_log_probs = 0.0  # Uniform prior

        # Intentional Q1(s, a)
        i_q1_new_actions = self._i_qf1(obs, i_new_actions)[0]

        if self._i_qf2 is not None:
            # Intentional Q2(s, a)
            i_q2_new_actions = self._i_qf2(obs, i_new_actions)[0]

            # Minimum Intentional Double-Q
            i_q_new_actions = torch.min(i_q1_new_actions, i_q2_new_actions)
        else:
            i_q_new_actions = i_q1_new_actions

        # # Intentional A(s, a)
        # i_advantage_new_actions = i_q_new_actions - i_v_pred.detach()

        # Intentional policy KL loss: - (E_a[Q(s, a) + H(.)])
        i_policy_kl_loss = -torch.mean(
            i_q_new_actions - ialpha*i_new_log_pi
            + i_policy_prior_log_probs
        )
        # i_policy_kl_loss = -torch.mean(
        # i_advantage_new_actions - i_log_pi*ialpha
        # )

        # Intentional policy regularization loss
        i_mean_reg_loss = self._i_pol_mean_regu_weight * \
            (i_new_policy_mean ** 2).mean()
        i_std_reg_loss = self._i_pol_std_regu_weight * \
            (i_new_policy_log_std ** 2).mean()
        i_pre_activation_reg_loss = \
            self._i_pol_pre_activ_weight * \
            (i_new_pre_tanh_value**2).sum(dim=-1).mean()
        mixing_coeff_loss = self._i_pol_mixing_coeff_weight * \
            0.5*((new_mixing_coeff - 1/self._n_unintentional)**2).mean()
        i_policy_regu_loss = (i_mean_reg_loss + i_std_reg_loss
                              + i_pre_activation_reg_loss + mixing_coeff_loss)

        # Intentional Policy Total loss
        i_policy_loss = (i_policy_kl_loss + i_policy_regu_loss)

        # Update both Intentional and Unintentional Policies at the same time
        self._policy_optimizer.zero_grad()
        total_iu_loss = total_u_policy_loss + i_policy_loss
        total_iu_loss.backward()
        self._policy_optimizer.step()

        # ############### #
        # V-function Step #
        # ############### #
        if self._u_vf is None:
            u_v_pred = u_q_new_actions - ualphas*u_new_log_pi
            u_vf_loss = 0
            total_u_vf_loss = 0
        else:
            u_v_pred = torch.cat([vv.unsqueeze(1)
                                  for vv in self._u_vf(obs)[0]],
                                 dim=1)

            # Calculate Bellman Backup for Unintentional V-values
            u_v_backup = u_q_new_actions - ualphas*u_new_log_pi + u_policy_prior_log_probs
            u_v_backup = u_v_backup.detach()

            u_vf_loss = \
                0.5*torch.mean((u_v_backup - u_v_pred)**2, dim=0).squeeze(-1)
            total_u_vf_loss = torch.sum(u_vf_loss)

        if self._i_vf is None:
            i_v_pred = i_q_new_actions - ialpha*i_new_log_pi
            i_vf_loss = 0
        else:
            i_v_pred = self._i_vf(obs)[0]
            # Calculate Bellman Backup for V-value
            i_v_backup = i_q_new_actions - ialpha*i_new_log_pi + i_policy_prior_log_probs
            i_v_backup = i_v_backup.detach()
            # Calculate Intentional Vf Loss
            i_vf_loss = 0.5*torch.mean((i_v_backup - i_v_pred)**2)

        # Update both Intentional and Unintentional Values at the same time
        self._values_optimizer.zero_grad()
        values_loss = (total_u_qf1_loss + total_u_qf2_loss +
                       i_qf1_loss + i_qf2_loss +
                       total_u_vf_loss + i_vf_loss)
        values_loss.backward()
        self._values_optimizer.step()

        # ###################### #
        # Update Target Networks #
        # ###################### #
        if self._n_train_steps_total % self._u_target_update_interval == 0:
            if self._u_target_vf is None:
                ptu.soft_update_from_to(
                    source=self._u_qf1,
                    target=self._u_target_qf1,
                    tau=self._u_soft_target_tau
                )
                if self._u_target_qf2 is not None:
                    ptu.soft_update_from_to(
                        source=self._u_qf2,
                        target=self._u_target_qf2,
                        tau=self._u_soft_target_tau
                    )
            else:
                ptu.soft_update_from_to(
                    source=self._u_vf,
                    target=self._u_target_vf,
                    tau=self._u_soft_target_tau
                )

        if self._n_train_steps_total % self._i_target_update_interval == 0:
            if self._i_target_vf is None:
                ptu.soft_update_from_to(
                    source=self._i_qf1,
                    target=self._i_target_qf1,
                    tau=self._i_soft_target_tau
                )
                if self._i_target_qf2 is not None:
                    ptu.soft_update_from_to(
                        source=self._i_qf2,
                        target=self._i_target_qf2,
                        tau=self._i_soft_target_tau
                    )
            else:
                ptu.soft_update_from_to(
                    source=self._i_vf,
                    target=self._i_target_vf,
                    tau=self._i_soft_target_tau
                )

        # ################################## #
        # Intentional & Unintentional Alphas #
        # ################################## #
        if self._auto_alphas:
            u_log_alphas = self._u_log_alphas.clamp(max=MAX_LOG_ALPHA)
            u_alpha_loss = -(u_log_alphas.unsqueeze(-1) *
                             (u_new_log_pi + self._u_tgt_entros.unsqueeze(-1)
                              ).detach()
                             ).mean()

            i_log_alpha = self._i_log_alpha.clamp(max=MAX_LOG_ALPHA)
            i_alpha_loss = -(i_log_alpha * (
                    i_new_log_pi + self._i_tgt_entro).detach()).mean()
            self._alphas_optimizer.zero_grad()
            total_alpha_loss = u_alpha_loss + i_alpha_loss
            total_alpha_loss.backward()
            self._alphas_optimizer.step()

        # ############### #
        # LOG Useful Data #
        # ############### #
        self.log_data['Policy Entropy'][step_idx, :-1] = \
            ptu.get_numpy(-u_new_log_pi.mean(dim=0).squeeze(-1))
        self.log_data['Policy Entropy'][step_idx, -1] = \
            ptu.get_numpy(-i_new_log_pi.mean(dim=0))

        self.log_data['Pol Log Std'][step_idx, :-1, :] = \
            ptu.get_numpy(u_new_policy_log_std.mean(dim=0))
        self.log_data['Pol Log Std'][step_idx, -1, :] = \
            ptu.get_numpy(i_new_policy_log_std.mean(dim=0))

        self.log_data['Policy Mean'][step_idx, :-1, :] = \
            ptu.get_numpy(u_new_policy_mean.mean(dim=0))
        self.log_data['Policy Mean'][step_idx, -1, :] = \
            ptu.get_numpy(i_new_policy_mean.mean(dim=0))

        self.log_data['Pol KL Loss'][step_idx, :-1] = \
            ptu.get_numpy(u_policy_kl_loss)
        self.log_data['Pol KL Loss'][step_idx, -1] = \
            ptu.get_numpy(i_policy_kl_loss)

        self.log_data['Qf Loss'][step_idx, :-1] = \
            ptu.get_numpy(u_qf1_loss)
        self.log_data['Qf Loss'][step_idx, -1] = \
            ptu.get_numpy(i_qf1_loss)

        if self._u_qf2 is not None:
            self.log_data['Qf2 Loss'][step_idx, :-1] = \
                ptu.get_numpy(u_qf2_loss)
        if self._i_qf2 is not None:
            self.log_data['Qf2 Loss'][step_idx, -1] = \
                ptu.get_numpy(i_qf2_loss)

        if self._u_vf is not None:
            self.log_data['Vf Loss'][step_idx, :-1] = \
                ptu.get_numpy(u_vf_loss)
        if self._i_vf is not None:
            self.log_data['Vf Loss'][step_idx, -1] = \
                ptu.get_numpy(i_vf_loss)

        self.log_data['Rewards'][step_idx, :-1] = \
            ptu.get_numpy(u_rewards.mean(dim=0).squeeze(-1))
        self.log_data['Rewards'][step_idx, -1] = \
            ptu.get_numpy(i_rewards.mean(dim=0).squeeze(-1))

        self.log_data['Mixing Weights'][step_idx, :, :] = \
            ptu.get_numpy(new_mixing_coeff.mean(dim=0))

        self.log_data['Alphas'][step_idx, :-1] = \
            ptu.get_numpy(ualphas.squeeze(-1))
        self.log_data['Alphas'][step_idx, -1] = \
            ptu.get_numpy(ialpha)

        if self._log_tensorboard:
            self._summary_writer.add_scalar(
                'TrainingI/qf_loss',
                ptu.get_numpy(i_qf1_loss),
                self._n_env_steps_total
            )
            if self._i_qf2 is not None:
                self._summary_writer.add_scalar(
                    'TrainingI/qf2_loss',
                    ptu.get_numpy(i_qf2_loss),
                    self._n_env_steps_total
                )
            self._summary_writer.add_scalar(
                'TrainingI/avg_reward',
                ptu.get_numpy(i_rewards.mean()),
                self._n_env_steps_total
            )
            # self._summary_writer.add_scalar(
            #     'TrainingI/avg_advantage',
            #     ptu.get_numpy(i_advantage_new_actions.mean()),
            #     self._n_env_steps_total
            # )
            self._summary_writer.add_scalar(
                'TrainingI/policy_loss',
                ptu.get_numpy(i_policy_loss),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'TrainingI/policy_entropy',
                ptu.get_numpy(-i_new_log_pi.mean()),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'TrainingI/policy_mean',
                ptu.get_numpy(i_new_policy_mean.mean()),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'TrainingI/policy_std',
                np.exp(ptu.get_numpy(i_new_policy_log_std.mean())),
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
            self._i_qf1,
            self._u_qf1,
        ]
        if self._i_qf2 is not None:
            networks_list.append(self._i_qf2)
        if self._i_vf is not None:
            networks_list.append(self._i_vf)
        if self._i_target_qf1 is not None:
            networks_list.append(self._i_target_qf1)
        if self._i_target_qf2 is not None:
            networks_list.append(self._i_target_qf2)
        if self._i_target_vf is not None:
            networks_list.append(self._i_target_vf)
        if self._u_qf2 is not None:
            networks_list.append(self._u_qf2)
        if self._u_vf is not None:
            networks_list.append(self._u_vf)
        if self._u_target_qf1 is not None:
            networks_list.append(self._u_target_qf1)
        if self._u_target_qf2 is not None:
            networks_list.append(self._u_target_qf2)
        if self._u_target_vf is not None:
            networks_list.append(self._u_target_vf)

        return networks_list

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
            qf=self._i_qf1,
            qf2=self._i_qf2,
            target_qf=self._i_target_qf1,
            vf=self._i_vf,
            target_vf=self._i_target_vf,
            u_qf=self._u_qf1,
            u_qf2=self._u_qf2,
            u_vf=self._u_vf,
            target_u_qf1=self._u_target_qf1,
            target_u_qf2=self._u_target_qf2,
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
                np.nan_to_num(np.mean(
                    self.log_data['Policy Entropy'][:max_step, uu]
                ))
            self.eval_statistics['[U-%02d] Qf Loss' % uu] = \
                np.nan_to_num(np.mean(
                    self.log_data['Qf Loss'][:max_step, uu]
                ))
            self.eval_statistics['[U-%02d] Vf Loss' % uu] = \
                np.nan_to_num(np.mean(
                    self.log_data['Vf Loss'][:max_step, uu]
                ))
            self.eval_statistics['[U-%02d] Pol KL Loss' % uu] = \
                np.nan_to_num(np.mean(
                    self.log_data['Pol KL Loss'][:max_step, uu]
                ))
            self.eval_statistics['[U-%02d] Rewards' % uu] = \
                np.nan_to_num(np.mean(
                    self.log_data['Rewards'][:max_step, uu]
                ))
            self.eval_statistics['[U-%02d] Mixing Weights' % uu] = \
                np.nan_to_num(np.mean(
                    self.log_data['Mixing Weights'][:max_step, uu]
                ))
            self.eval_statistics['[U-%02d] Alphas' % uu] = \
                np.nan_to_num(np.mean(
                    self.log_data['Alphas'][:max_step, uu]
                ))

            for aa in range(self.env.action_dim):
                self.eval_statistics['[U-%02d] Policy Std [%02d]' % (uu, aa)] = \
                    np.nan_to_num(np.mean(
                        np.exp(self.log_data['Pol Log Std'][:max_step, uu, aa])
                    ))
                self.eval_statistics['[U-%02d] Policy Mean [%02d]' % (uu, aa)] = \
                    np.nan_to_num(np.mean(
                        self.log_data['Policy Mean'][:max_step, uu, aa]
                    ))

            if self._u_qf2 is not None:
                self.eval_statistics['[U-%02d] Qf2 Loss' % uu] = \
                    np.nan_to_num(np.mean(
                        self.log_data['Qf2 Loss'][:max_step, uu]
                    ))

        # Intentional info
        self.eval_statistics['[I] Policy Entropy'] = \
            np.nan_to_num(np.mean(
                self.log_data['Policy Entropy'][:max_step, -1]
            ))
        self.eval_statistics['[I] Qf Loss'] = \
            np.nan_to_num(np.mean(
                self.log_data['Qf Loss'][:max_step, -1]
            ))
        if self._i_qf2 is not None:
            self.eval_statistics['[I] Qf2 Loss'] = \
                np.nan_to_num(np.mean(
                    self.log_data['Qf2 Loss'][:max_step, -1]
                ))
        self.eval_statistics['[I] Vf Loss'] = \
            np.nan_to_num(np.mean(
                self.log_data['Vf Loss'][:max_step, -1]
            ))
        self.eval_statistics['[I] Pol KL Loss'] = \
            np.nan_to_num(np.mean(
                self.log_data['Pol KL Loss'][:max_step, -1]
            ))
        self.eval_statistics['[I] Rewards'] = \
            np.nan_to_num(np.mean(
                self.log_data['Rewards'][:max_step, -1]
            ))
        self.eval_statistics['[I] Alphas'] = \
            np.nan_to_num(np.mean(
                self.log_data['Alphas'][:max_step, -1]
            ))
        for aa in range(self.env.action_dim):
            self.eval_statistics['[I] Policy Std'] = \
                np.nan_to_num(np.mean(
                    np.exp(self.log_data['Pol Log Std'][:max_step, -1, aa])
                ))
            self.eval_statistics['[I] Policy Mean'] = \
                np.nan_to_num(np.mean(
                    self.log_data['Policy Mean'][:max_step, -1, aa]
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

        return batch

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
