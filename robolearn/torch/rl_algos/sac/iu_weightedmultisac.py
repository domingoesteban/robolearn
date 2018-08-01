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
from robolearn.torch.torch_incremental_rl_algorithm import TorchIncrementalRLAlgorithm
from robolearn.policies import MakeDeterministic
from robolearn.torch.policies import WeightedMultiPolicySelector
from torch.autograd import Variable


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

            policy_lr=1e-3,
            policies_lr=1e-3,
            mixing_lr=1e-3,

            i_qf_lr=1e-3,
            i_vf_lr=1e-3,
            u_qf_lr=1e-3,
            u_vf_lr=1e-3,

            i_policy_mean_regu_weight=1e-3,
            i_policy_std_regu_weight=1e-3,
            i_policy_pre_activation_weight=0.,
            i_policy_mixing_coeff_weight=1e-3,

            u_policy_mean_regu_weight=None,
            u_policy_std_regu_weight=None,
            u_policy_pre_activation_weight=None,

            optimizer_class=optim.Adam,
            # optimizer_class=optim.SGD,

            i_soft_target_tau=1e-2,
            u_soft_target_tau=1e-2,

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

        # Intentional (Main Task) Q-function and V-function
        self._i_qf = i_qf
        self._i_vf = i_vf
        self._i_target_vf = self._i_vf.copy()

        # Number of Unintentional Tasks (Composable Tasks)
        self._n_unintentional = self._policy.n_heads

        # Unintentional (Composable Tasks) Q-function and V-function
        self._u_qf = u_qf
        self._u_vf = u_vf
        self._u_target_vf = self._u_vf.copy()

        # Soft-update rate
        self._i_soft_target_tau = i_soft_target_tau
        self._u_soft_target_tau = u_soft_target_tau

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
        self._qf_criterion = nn.MSELoss()
        self._vf_criterion = nn.MSELoss()

        # Q-function optimizers
        self._u_qf_optimizer = optimizer_class(self._u_qf.parameters(),
                                               lr=u_qf_lr)
        self._i_qf_optimizer = optimizer_class(self._i_qf.parameters(),
                                               lr=i_qf_lr)

        # V-function optimizers
        self._u_vf_optimizer = optimizer_class(self._u_vf.parameters(),
                                               lr=u_vf_lr)
        self._i_vf_optimizer = optimizer_class(self._i_vf.parameters(),
                                               lr=i_vf_lr)

        # Policy optimizer
        # self._policy_optimizer = optimizer_class([
        #     {'params': self._policy.shared_parameters(),
        #      'lr': policy_lr},
        #     {'params': self._policy.policies_parameters(),
        #      'lr': policy_lr},
        #     {'params': self._policy.mixing_parameters(),
        #      'lr': policy_lr},
        # ])
        self._policy_optimizer = optimizer_class(self._policy.parameters(),
                                                 lr=policy_lr)
        self._mixing_optimizer = \
            optimizer_class(chain(self._policy.shared_parameters(),
                                  self._policy.mixing_parameters()),
                            lr=mixing_lr)
        self._policies_optimizer = \
            optimizer_class(chain(self._policy.shared_parameters(),
                                  self._policy.policies_parameters()),
                            lr=policies_lr)

        # Policy regularization coefficients (weights)
        self._i_policy_mean_reg_weight = i_policy_mean_regu_weight
        self._i_policy_std_reg_weight = i_policy_std_regu_weight
        self._i_policy_pre_activation_weight = i_policy_pre_activation_weight
        self._i_policy_mixing_coeff_weight = i_policy_mixing_coeff_weight

        if u_policy_mean_regu_weight is None:
            u_policy_mean_regu_weight = [i_policy_mean_regu_weight
                                         for _ in range(self._n_unintentional)]
        self._u_policy_mean_reg_weight = u_policy_mean_regu_weight
        if u_policy_std_regu_weight is None:
            u_policy_std_regu_weight = [i_policy_std_regu_weight
                                        for _ in range(self._n_unintentional)]
        self._u_policy_std_reg_weight = u_policy_std_regu_weight
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

        # Evaluation Sampler (One for each unintentional)
        self.eval_samplers = [
            InPlacePathSampler(env=env,
                               policy=WeightedMultiPolicySelector(self._policy,
                                                                  idx),
                               max_samples=self.num_steps_per_eval,
                               max_path_length=self.max_path_length,)
            for idx in range(self._n_unintentional)
        ]

        # Useful Varables for logging
        self.logging_log_entrop = np.zeros((self.num_env_steps_per_epoch,
                                            self._n_unintentional + 1))
        self.logging_log_policy_target = np.zeros((self.num_env_steps_per_epoch,
                                                   self._n_unintentional + 1))
        self.logging_policy_log_std = np.zeros((self.num_env_steps_per_epoch,
                                                self._n_unintentional + 1))
        self.logging_policy_mean = np.zeros((self.num_env_steps_per_epoch,
                                             self._n_unintentional + 1))
        self.logging_qf_loss = np.zeros((self.num_env_steps_per_epoch,
                                         self._n_unintentional + 1))
        self.logging_vf_loss = np.zeros((self.num_env_steps_per_epoch,
                                         self._n_unintentional + 1))
        self.logging_rewards = np.zeros((self.num_env_steps_per_epoch,
                                         self._n_unintentional + 1))
        self.logging_mixing_coeff = np.zeros((self.num_env_steps_per_epoch,
                                              self._n_unintentional))

    def pretrain(self):
        # We do not require any pretrain (I think...)
        pass

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
        #     self.logging_log_entrop[step_idx, uu] = \
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
        # self.logging_log_entrop[step_idx, -1] = \
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
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

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
        vf = self._u_vf
        target_vf = self._u_target_vf

        # ########### #
        # Critic Step #
        # ########### #
        accum_u_qf_loss = 0
        u_q_preds = qf(obs, actions)[0]
        u_target_v_values = target_vf(next_obs)[0]
        for uu in range(self._n_unintentional):
            rewards = batch['reward_vectors'][:, uu].unsqueeze(-1) \
                      * self._u_reward_scales[uu]
            terminals = batch['terminal_vectors'][:, uu].unsqueeze(-1)

            # Calculate QF Loss
            target_v_value = u_target_v_values[uu]  # (From Next obs)
            q_target = rewards + (1. - terminals) * self.discount * target_v_value

            q_pred = u_q_preds[uu]
            u_qf_loss = self._qf_criterion(q_pred, q_target.detach())

            accum_u_qf_loss += u_qf_loss

            # Log data
            self.logging_qf_loss[step_idx, uu] = ptu.get_numpy(u_qf_loss)
            self.logging_rewards[step_idx, uu] = \
                ptu.get_numpy(rewards.mean(dim=0))

        # Update Unintentional Q-values
        self._u_qf_optimizer.zero_grad()
        accum_u_qf_loss.backward()
        self._u_qf_optimizer.step()

        # ############### #
        # Actor & Vf Step #
        # ############### #
        accum_u_vf_loss = 0
        accum_u_policy_loss = 0
        u_v_preds = vf(obs)[0]
        for uu in range(self._n_unintentional):
            # Get Actions and Info from Unintentional Policy
            new_actions, policy_info = policy(obs, deterministic=False,
                                              return_log_prob=True,
                                              pol_idx=uu,
                                              optimize_policies=True)
            log_pi = policy_info['log_prob']
            policy_mean = policy_info['mean']
            policy_log_std = policy_info['log_std']
            pre_tanh_value = policy_info['pre_tanh_value']

            q_new_actions = qf(obs, new_actions)[0][uu]
            v_pred = u_v_preds[uu]
            log_policy_target = q_new_actions - v_pred

            # KL loss
            policy_kl_loss = (
                    log_pi * (log_pi - log_policy_target).detach()
            ).mean()
            # Regularization loss
            mean_reg_loss = self._u_policy_mean_reg_weight[uu] * \
                (policy_mean ** 2).mean()
            std_reg_loss = self._u_policy_std_reg_weight[uu] * \
                (policy_log_std ** 2).mean()
            pre_activation_reg_loss = self._u_policy_pre_activation_weight[uu] * \
                (pre_tanh_value**2).sum(dim=-1).mean()
            policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss

            accum_u_policy_loss += (policy_kl_loss + policy_reg_loss)

            # # Calculate Intentional Vf Loss
            v_target = q_new_actions - log_pi
            u_vf_loss = self._vf_criterion(v_pred, v_target.detach())
            # u_vf_loss = (v_pred * (v_pred - q_new_actions + log_pi).detach()).mean()

            accum_u_vf_loss += u_vf_loss


            # ############### #
            # LOG Useful Data #
            # ############### #
            self.logging_log_entrop[step_idx, uu] = \
                ptu.get_numpy(-log_pi.mean(dim=0))
            self.logging_log_policy_target[step_idx, uu] = \
                ptu.get_numpy(log_policy_target.mean(dim=0))
            self.logging_policy_log_std[step_idx, uu] = \
                ptu.get_numpy(policy_log_std.mean())
            self.logging_policy_mean[step_idx, uu] = \
                ptu.get_numpy(policy_mean.mean())
            self.logging_vf_loss[step_idx, uu] = \
                ptu.get_numpy(u_vf_loss)

        # Update Unintentional (Composable) Policies
        self._policy_optimizer.zero_grad()
        self._policies_optimizer.zero_grad()
        accum_u_policy_loss.backward()
        self._policies_optimizer.step()

        # Update Unintentional V-value
        self._u_vf_optimizer.zero_grad()
        accum_u_vf_loss.backward()
        self._u_vf_optimizer.step()

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
        vf = self._i_vf
        target_vf = self._i_target_vf

        # ########### #
        # Critic Step #
        # ########### #
        # Calculate Q-value loss
        q_pred = qf(obs, actions)[0]
        target_v_value = target_vf(next_obs)[0]
        q_target = rewards + (1. - terminals) * self.discount * target_v_value
        i_qf_loss = self._qf_criterion(q_pred, q_target.detach())



        # Update Intentional Q-value
        self._i_qf_optimizer.zero_grad()
        i_qf_loss.backward()
        self._i_qf_optimizer.step()

        # ########## #
        # Actor Step #
        # ########## #
        # Calculate Intentional Policy Loss
        new_actions, policy_info = policy(obs, deterministic=False,
                                          return_log_prob=True,
                                          pol_idx=None,
                                          optimize_policies=False)
                                          # optimize_policies=True)

        log_pi = policy_info['log_prob']
        policy_mean = policy_info['mean']
        policy_log_std = policy_info['log_std']
        pre_tanh_value = policy_info['pre_tanh_value']
        mixing_coeff = policy_info['mixing_coeff']

        v_pred = vf(obs)[0]
        q_new_actions = qf(obs, new_actions)[0]
        log_policy_target = q_new_actions - v_pred

        # KL loss
        policy_kl_loss = (
                log_pi * (log_pi - log_policy_target).detach()
        ).mean()
        # Regularization loss
        mean_reg_loss = self._i_policy_mean_reg_weight * \
            (policy_mean ** 2).mean()
        std_reg_loss = self._i_policy_std_reg_weight * \
            (policy_log_std ** 2).mean()
        pre_activation_reg_loss = self._i_policy_pre_activation_weight * \
            (pre_tanh_value**2).sum(dim=-1).mean()
        mixing_coeff_loss = self._i_policy_mixing_coeff_weight * \
            (mixing_coeff ** 2).sum(dim=-1).mean()

        policy_reg_loss = mean_reg_loss + std_reg_loss + \
                          pre_activation_reg_loss + mixing_coeff_loss

        i_policy_loss = (policy_kl_loss + policy_reg_loss)

        # Update Intentional Policy
        self._policy_optimizer.zero_grad()
        self._mixing_optimizer.zero_grad()
        i_policy_loss.backward()
        self._mixing_optimizer.step()

        # ############### #
        # V-function Step #
        # ############### #
        # Calculate Intentional Vf Loss
        v_target = q_new_actions - log_pi
        i_vf_loss = self._vf_criterion(v_pred, v_target.detach())
        # i_vf_loss = (v_pred * (v_pred - q_new_actions + log_pi).detach()).mean()

        # Update Intentional V-value
        self._i_vf_optimizer.zero_grad()
        i_vf_loss.backward()
        self._i_vf_optimizer.step()

        # Update Intentional V Target Network
        self._update_v_target_network(vf=self._i_vf,
                                      target_vf=self._i_target_vf,
                                      soft_target_tau=self._i_soft_target_tau)

        # ########################### #
        # LOG Useful Intentional Data #
        # ########################### #
        self.logging_log_entrop[step_idx, -1] = \
            ptu.get_numpy(-log_pi.mean(dim=0))
        self.logging_log_policy_target[step_idx, -1] = \
            ptu.get_numpy(log_policy_target.mean(dim=0))
        self.logging_policy_log_std[step_idx, -1] = \
            ptu.get_numpy(policy_log_std.mean())
        self.logging_policy_mean[step_idx, -1] = \
            ptu.get_numpy(policy_mean.mean())
        self.logging_qf_loss[step_idx, -1] = ptu.get_numpy(i_qf_loss)
        self.logging_vf_loss[step_idx, -1] = ptu.get_numpy(i_vf_loss)
        self.logging_rewards[step_idx, -1] = \
            ptu.get_numpy(rewards.mean(dim=0))
        self.logging_mixing_coeff[step_idx, :] = \
            ptu.get_numpy(mixing_coeff.mean(dim=0))

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
        if self._i_target_vf is None:
            target_i_vf = []
        else:
            target_i_vf = [self._i_target_vf]

        return [self._policy] + \
               [self._i_qf] + [self._u_qf] + \
               [self._i_vf] + [self._u_vf] + \
               target_i_vf + [self._u_target_vf]

    def _update_v_target_network(self, vf, target_vf, soft_target_tau):
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
            vf=self._i_vf,
            target_vf=self._i_target_vf,
            u_qf=self._u_qf,
            u_vf=self._u_vf,
            target_u_vf=self._u_target_vf,
        )

        if self.env.online_normalization:
            snapshot.update(
                obs_mean=self.env.obs_mean,
                obs_std=self.env.obs_std,
            )

        if self.save_replay_buffer:
            snapshot.update(
                replay_buffer=self.replay_buffer,
            )

        return snapshot

    def _update_logging_data(self):
        max_step = self._n_epoch_train_steps

        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
        # Unintentional info
        for uu in range(self._n_unintentional):
            self.eval_statistics['[U-%02d] Policy Entropy' % uu] = \
                np.nan_to_num(np.mean(self.logging_log_entrop[:max_step, uu]))
            self.eval_statistics['[U-%02d] Log Policy Target' % uu] = \
                np.nan_to_num(np.mean(self.logging_log_policy_target[:max_step, uu]))
            self.eval_statistics['[U-%02d] Policy Std' % uu] = \
                np.nan_to_num(np.mean(np.exp(self.logging_policy_log_std[:max_step, uu])))
            self.eval_statistics['[U-%02d] Policy Mean' % uu] = \
                np.nan_to_num(np.mean(self.logging_policy_mean[:max_step, uu]))
            self.eval_statistics['[U-%02d] Qf Loss' % uu] = \
                np.nan_to_num(np.mean(self.logging_qf_loss[:max_step, uu]))
            self.eval_statistics['[U-%02d] Vf Loss' % uu] = \
                np.nan_to_num(np.mean(self.logging_vf_loss[:max_step, uu]))
            self.eval_statistics['[U-%02d] Rewards' % uu] = \
                np.nan_to_num(np.mean(self.logging_rewards[:max_step, uu]))
            self.eval_statistics['[U-%02d] Mixing Weights' % uu] = \
                np.nan_to_num(np.mean(self.logging_mixing_coeff[:max_step, uu]))

        # Intentional info
        self.eval_statistics['[I] Policy Entropy'] = \
            np.nan_to_num(np.mean(self.logging_log_entrop[:max_step, -1]))
        self.eval_statistics['[I] Log Policy Target'] = \
            np.nan_to_num(np.mean(self.logging_log_policy_target[:max_step, -1]))
        self.eval_statistics['[I] Policy Std'] = \
            np.nan_to_num(np.mean(np.exp(self.logging_policy_log_std[:max_step, -1])))
        self.eval_statistics['[I] Policy Mean'] = \
            np.nan_to_num(np.mean(self.logging_policy_mean[:max_step, -1]))
        self.eval_statistics['[I] Qf Loss'] = \
            np.nan_to_num(np.mean(self.logging_qf_loss[:max_step, -1]))
        self.eval_statistics['[I] Vf Loss'] = \
            np.nan_to_num(np.mean(self.logging_vf_loss[:max_step, -1]))
        self.eval_statistics['[I] Rewards'] = \
            np.nan_to_num(np.mean(self.logging_rewards[:max_step, -1]))

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
            average_returns = \
                eval_util.get_average_multigoal_returns(test_paths[unint_idx],
                                                        unint_idx)
            avg_txt = '[U-%02d] Test AverageReturn' % unint_idx
            statistics[avg_txt] = average_returns

        logger.log("[I] Collecting samples for evaluation")
        i_test_path = self.eval_sampler.obtain_samples()
        statistics.update(eval_util.get_generic_path_information(
            i_test_path, stat_prefix="[I] Test",
        ))
        average_return = eval_util.get_average_returns(i_test_path)
        statistics['[I] Test AverageReturn'] = average_return

        if self._exploration_paths:
            statistics.update(eval_util.get_generic_path_information(
                self._exploration_paths, stat_prefix="Exploration",
            ))
        else:
            statistics.update(eval_util.get_generic_path_information(
                i_test_path, stat_prefix="Exploration",
            ))

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
        self.logging_log_entrop = np.zeros((self.num_env_steps_per_epoch,
                                            self._n_unintentional + 1))
        self.logging_log_policy_target = np.zeros((self.num_env_steps_per_epoch,
                                                   self._n_unintentional + 1))
        self.logging_policy_log_std = np.zeros((self.num_env_steps_per_epoch,
                                                self._n_unintentional + 1))
        self.logging_policy_mean = np.zeros((self.num_env_steps_per_epoch,
                                             self._n_unintentional + 1))
        self.logging_qf_loss = np.zeros((self.num_env_steps_per_epoch,
                                         self._n_unintentional + 1))
        self.logging_vf_loss = np.zeros((self.num_env_steps_per_epoch,
                                         self._n_unintentional + 1))
        self.logging_rewards = np.zeros((self.num_env_steps_per_epoch,
                                         self._n_unintentional + 1))
        self.logging_mixing_coeff = np.zeros((self.num_env_steps_per_epoch,
                                              self._n_unintentional))

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


