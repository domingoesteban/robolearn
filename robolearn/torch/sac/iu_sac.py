"""
Based on Pong's SAC implementation

https://github.com/vitchyr/rlkit
"""

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

from collections import OrderedDict

import robolearn.torch.pytorch_util as ptu
from robolearn.core import logger, eval_util
from robolearn.core.eval_util import create_stats_ordered_dict
from robolearn.utils.samplers import InPlacePathSampler
from robolearn.torch.torch_rl_algorithm import TorchRLAlgorithm
from robolearn.policies import MakeDeterministic


class IUSAC(TorchRLAlgorithm):
    """Intentional-Unintentional Soft Actor Critic (IU-SAC).

    """
    def __init__(
            self,
            env,
            u_policies,
            u_qfs,
            u_vfs,
            i_policy=None,
            i_qf=None,
            i_vf=None,
            exploration_pol_id=0,
            iu_mode='composition',

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            epoch_plotter=None,
            render_eval_paths=False,
            eval_deterministic=True,
            **kwargs
    ):
        # ######## #
        # Networks #
        # ######## #

        # Intentional Task
        if i_policy is None:
            self._i_policy = u_policies[exploration_pol_id]
        else:
            self._i_policy = i_policy

        if eval_deterministic:
            eval_policy = MakeDeterministic(i_policy)
        else:
            eval_policy = i_policy

        if i_qf is None:
            self._i_qf = u_qfs[exploration_pol_id]
        else:
            self._i_qf = i_qf

        if i_vf is None:
            self._i_vf = u_vfs[exploration_pol_id]
        else:
            self._i_vf = i_vf

        self._i_target_vf = self._i_vf.copy()

        self._iu_mode = iu_mode

        super(IUSAC, self).__init__(
            env=env,
            exploration_policy=self._i_policy,
            eval_policy=eval_policy,
            **kwargs
        )

        # Unintentional Tasks
        self._n_unintentional = len(u_qfs)

        self._u_policies = u_policies
        self._u_qfs = u_qfs
        self._u_vfs = u_vfs
        self._u_target_vfs = [u_vf.copy() for u_vf in self._u_vfs]

        self.soft_target_tau = soft_target_tau
        self._policy_mean_reg_weight = policy_mean_reg_weight
        self._policy_std_reg_weight = policy_std_reg_weight
        self._policy_pre_activation_weight = policy_pre_activation_weight

        # ########## #
        # Optimizers #
        # ########## #
        self._qf_criterion = nn.MSELoss()
        self._vf_criterion = nn.MSELoss()

        # Policy optimizers
        self._u_policy_optimizers = [optimizer_class(u_policy.parameters(),
                                                     lr=policy_lr)
                                     for u_policy in self._u_policies]
        if iu_mode == 'composition':
            self._i_policy_optimizer = \
                optimizer_class(self._i_policy.parameters(),
                                lr=policy_lr,)
        elif iu_mode == 'intentional':
            self._i_policy_optimizer = \
                optimizer_class(self._i_policy.parameters(),
                                lr=policy_lr,)
        elif iu_mode == 'random':
            self._i_policy_optimizer = None
        else:
            self._i_policy_optimizer = \
                self._u_policy_optimizers[exploration_pol_id]

        # Qf optimizers
        self._u_qf_optimizers = [optimizer_class(u_qf.parameters(),
                                                 lr=policy_lr)
                                 for u_qf in self._u_qfs]
        if iu_mode == 'composition':
            self._i_qf_optimizer = None
        elif iu_mode == 'intentional':
            self._i_qf_optimizer = optimizer_class(
                self._i_qf.parameters(),
                lr=qf_lr,
            )
        elif iu_mode == 'random':
            self._i_qf_optimizer = None
        else:
            self._i_qf_optimizer = \
                self._u_qf_optimizers[exploration_pol_id]

        # Vf optimizers
        self._u_vf_optimizers = [optimizer_class(u_vf.parameters(),
                                                 lr=policy_lr)
                                 for u_vf in self._u_vfs]
        if iu_mode == 'composition':
            self._i_vf_optimizer = None
        elif iu_mode == 'intentional':
            self._i_vf_optimizer = optimizer_class(
                self._i_vf.parameters(),
                lr=vf_lr,
            )
        elif iu_mode == 'random':
            self._i_vf_optimizer = None
        else:
            self._i_vf_optimizer = \
                self._u_vf_optimizers[exploration_pol_id]

        # ########### #
        # Other Stuff
        # ########### #
        self.eval_statistics = None
        self._epoch_plotter = epoch_plotter
        self.render_eval_paths = render_eval_paths
        # Evaluation Sampler (One for each unintentional
        self.eval_samplers = [
            InPlacePathSampler(env=env, policy=eval_policy,
                               max_samples=self.num_steps_per_eval + self.max_path_length,
                               max_path_length=self.max_path_length,)
            for eval_policy in self._u_policies
        ]

    def pretrain(self):
        pass

    def _do_training(self):
        batch = self.get_batch()

        # Update Unintentional Networks
        for unint_idx in range(self._n_unintentional):
            update_outputs = self._update_soft_networks(batch, unint_idx)
            self._print_soft_statistics(update_outputs, unint_idx)

        # Update Intentional Networks
        if self._iu_mode == 'composition':
            update_outputs = self._update_soft_networks(batch, unint_idx=None)
            self._print_soft_statistics(update_outputs, unint_idx=None)
        elif self._iu_mode == 'random':
            pass
        else:
            update_outputs = self._update_soft_networks(batch, unint_idx=None)
            self._print_soft_statistics(update_outputs, unint_idx=None)

    def _update_soft_networks(self, batch, unint_idx=None):
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        if unint_idx is None:
            rewards = batch['rewards']
        else:
            rewards = batch['reward_vectors'][:, unint_idx].unsqueeze(-1) \
                      * self.reward_scale

        if unint_idx is None:
            policy = self._i_policy
            qf = self._i_qf
            vf = self._i_vf
            target_vf = self._i_target_vf
            policy_optimizer = self._i_policy_optimizer
            qf_optimizer = self._i_qf_optimizer
            vf_optimizer = self._i_vf_optimizer
        else:
            policy = self._u_policies[unint_idx]
            qf = self._u_qfs[unint_idx]
            vf = self._u_vfs[unint_idx]
            target_vf = self._u_target_vfs[unint_idx]
            policy_optimizer = self._u_policy_optimizers[unint_idx]
            qf_optimizer = self._u_qf_optimizers[unint_idx]
            vf_optimizer = self._u_vf_optimizers[unint_idx]

        q_pred = qf(obs, actions)
        v_pred = vf(obs)

        # Make sure _i_policy accounts for squashing functions like tanh correctly!
        policy_outputs = policy(obs, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        """
        QF Loss
        """
        target_v_values = target_vf(next_obs)
        q_target = rewards + (1. - terminals) * self.discount * target_v_values
        qf_loss = self._qf_criterion(q_pred, q_target.detach())

        """
        VF Loss
        """
        q_new_actions = qf(obs, new_actions)
        v_target = q_new_actions - log_pi
        vf_loss = self._vf_criterion(v_pred, v_target.detach())

        """
        Policy Loss
        """
        log_policy_target = q_new_actions - v_pred
        policy_loss = (
                log_pi * (log_pi - log_policy_target).detach()
        ).mean()
        mean_reg_loss = self._policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self._policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self._policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        """
        Update networks
        """
        if qf_optimizer is not None:
            qf_optimizer.zero_grad()
            qf_loss.backward()
            qf_optimizer.step()

        if vf_optimizer is not None:
            vf_optimizer.zero_grad()
            vf_loss.backward()
            vf_optimizer.step()

        if policy_optimizer is not None:
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

        self._update_v_target_network(vf, target_vf)

        return policy_loss, qf_loss, vf_loss, \
               policy_mean, log_pi, policy_log_std, q_pred, v_pred


    def _print_soft_statistics(self, update_outputs, unint_idx=None):
        policy_loss = update_outputs[0]
        qf_loss = update_outputs[1]
        vf_loss = update_outputs[2]
        policy_mean = update_outputs[3]
        log_pi = update_outputs[4]
        policy_log_std = update_outputs[5]
        q_pred = update_outputs[6]
        v_pred = update_outputs[7]

        if unint_idx is None:
            stats_string = 'I'
        else:
            stats_string = 'U-%02d' % unint_idx

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()

        self.eval_statistics['[%s] QF Loss' % stats_string] = \
            np.mean(ptu.get_numpy(qf_loss))
        self.eval_statistics['[%s] VF Loss' % stats_string] = \
            np.mean(ptu.get_numpy(vf_loss))
        self.eval_statistics['[%s] Policy Loss' % stats_string] = \
            np.mean(ptu.get_numpy(policy_loss))

        self.eval_statistics.update(create_stats_ordered_dict(
            '[%s] Q Predictions' % stats_string,
            ptu.get_numpy(q_pred),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            '[%s] V Predictions' % stats_string,
            ptu.get_numpy(v_pred),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            '[%s] Log Pis' % stats_string,
            ptu.get_numpy(log_pi),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            '[%s] Policy mu' % stats_string,
            ptu.get_numpy(policy_mean),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            '[%s] Policy log stds' % stats_string,
            ptu.get_numpy(policy_log_std),
        ))

    @property
    def networks(self):
        if self._i_target_vf is None:
            target_i_vf = []
        else:
            target_i_vf = [self._i_target_vf]

        return [self._i_policy] + self._u_policies + \
               [self._i_qf] + self._u_qfs + \
               [self._i_vf] + self._u_vfs + \
               target_i_vf + self._u_target_vfs

    def _update_v_target_network(self, vf, target_vf):
        ptu.soft_update_from_to(vf, target_vf, self.soft_target_tau)

    def get_epoch_snapshot(self, epoch):
        if self._epoch_plotter is not None:
            self._epoch_plotter.draw()
            self._epoch_plotter.save_figure(epoch)

        snapshot = super(IUSAC, self).get_epoch_snapshot(epoch)
        snapshot.update(
            policy=self._i_policy,
            u_policies=self._u_policies,
            qf=self._i_qf,
            # u_qfs=self._u_qf,
            vf=self._i_vf,
            # u_vfs=self._u_vf,
            target_vf=self._i_target_vf,
        )
        return snapshot

    def evaluate(self, epoch):
        statistics = OrderedDict()
        statistics.update(self.eval_statistics)
        self.eval_statistics = None

        test_paths = [None for _ in range(self._n_unintentional)]
        for unint_idx in range(self._n_unintentional):
            logger.log("[U-%02d] Collecting samples for evaluation" % unint_idx)
            test_paths[unint_idx] = self.eval_samplers[unint_idx].obtain_samples()

            statistics.update(eval_util.get_generic_path_information(
                test_paths[unint_idx], stat_prefix="[U-%02d] Test" % unint_idx,
            ))
            average_returns = eval_util.get_average_returns(test_paths[unint_idx])
            statistics['[U-%02d] AverageReturn' % unint_idx] = average_returns

        logger.log("[I] Collecting samples for evaluation")
        i_test_path = self.eval_sampler.obtain_samples()
        statistics.update(eval_util.get_generic_path_information(
            i_test_path, stat_prefix="[I] Test",
        ))
        average_return = eval_util.get_average_returns(i_test_path)
        statistics['[I] AverageReturn'] = average_return

        statistics.update(eval_util.get_generic_path_information(
            self._exploration_paths, stat_prefix="Exploration",
        ))
        if hasattr(self.env, "log_diagnostics"):
            # TODO: CHECK ENV LOG_DIAGNOSTICS
            print('TODO: WE NEED LOG_DIAGNOSTICS IN ENV')
            # self.env.log_diagnostics(test_paths[demon])

        # Record the data
        for key, value in statistics.items():
            logger.record_tabular(key, value)

        for unint_idx in range(self._n_unintentional):
            if self.render_eval_paths:
                # TODO: CHECK ENV RENDER_PATHS
                print('TODO: RENDER_PATHS')
                pass
                # self.env.render_paths(test_paths[demon])

        if self._epoch_plotter is not None:
            self._epoch_plotter.draw()
