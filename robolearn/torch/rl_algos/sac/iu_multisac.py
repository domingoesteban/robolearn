"""
Based on Pong's SAC implementation

https://github.com/vitchyr/rlkit
"""

import numpy as np
import torch.optim as optim
from torch import nn as nn

from collections import OrderedDict

import robolearn.torch.pytorch_util as ptu
from robolearn.core import logger, eval_util
from robolearn.utils.samplers import InPlacePathSampler
from robolearn.torch.torch_incremental_rl_algorithm import TorchIncrementalRLAlgorithm
from robolearn.policies import MakeDeterministic
from robolearn.torch.policies import MultiPolicySelector


class IUMultiSAC(TorchIncrementalRLAlgorithm):
    """Intentional-Unintentional Soft Actor Critic (IU-SAC)
    with MultiHead Networks.

    """
    def __init__(
            self,
            env,
            u_policy,
            u_qf,
            u_vf,
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
            self._i_policy = u_policy
        else:
            self._i_policy = i_policy

        if eval_deterministic:
            eval_policy = MakeDeterministic(i_policy)
        else:
            eval_policy = i_policy

        if i_qf is None:
            self._i_qf = u_qf
        else:
            self._i_qf = i_qf

        if i_vf is None:
            self._i_vf = u_vf
        else:
            self._i_vf = i_vf

        if iu_mode == 'composition':
            # self._i_target_vf = None
            self._i_target_vf = self._i_vf.copy()
        else:
            self._i_target_vf = self._i_vf.copy()

        self._iu_mode = iu_mode

        super(IUMultiSAC, self).__init__(
            env=env,
            exploration_policy=self._i_policy,
            eval_policy=eval_policy,
            off_policy=True,
            **kwargs
        )

        # Unintentional Tasks
        self._n_unintentional = u_policy.n_heads

        self._u_policy = u_policy
        self._u_qf = u_qf
        self._u_vf = u_vf
        self._u_target_vf = self._u_vf.copy()

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
        self._u_policy_optimizer = optimizer_class(self._u_policy.parameters(),
                                                   lr=policy_lr)
        if iu_mode == 'composition':
            # self._i_policy_optimizer = None
            self._i_policy_optimizer = \
                optimizer_class(self._i_policy.parameters(),
                                lr=policy_lr,)
        elif iu_mode == 'intentional':
            self._i_policy_optimizer = None
            # self._i_policy_optimizer = \
            #     optimizer_class(self._i_policy.parameters(),
            #                     lr=policy_lr,)
        elif iu_mode == 'random':
            self._i_policy_optimizer = None
        else:
            self._i_policy_optimizer = self._u_policy_optimizer

        # Qf optimizers
        self._u_qf_optimizer = optimizer_class(self._u_qf.parameters(),
                                               lr=qf_lr)
        if iu_mode == 'composition':
            # self._i_qf_optimizer = None
            self._i_qf_optimizer = optimizer_class(
                self._i_qf.parameters(),
                lr=qf_lr,
            )
        elif iu_mode == 'intentional':
            self._i_qf_optimizer = None
            # self._i_qf_optimizer = optimizer_class(
            #     self._i_qf.parameters(),
            #     lr=qf_lr,
            # )
        elif iu_mode == 'random':
            self._i_qf_optimizer = None
        else:
            self._i_qf_optimizer = self._u_qf_optimizer

        # Vf optimizers
        self._u_vf_optimizer = optimizer_class(self._u_vf.parameters(),
                                               lr=vf_lr)
        if iu_mode == 'composition':
            # self._i_vf_optimizer = None
            self._i_vf_optimizer = optimizer_class(
                self._i_vf.parameters(),
                lr=vf_lr,
            )
        elif iu_mode == 'intentional':
            self._i_vf_optimizer = None
            # self._i_vf_optimizer = optimizer_class(
            #     self._i_vf.parameters(),
            #     lr=vf_lr,
            # )
        elif iu_mode == 'random':
            self._i_vf_optimizer = None
        else:
            self._i_vf_optimizer = self._u_vf_optimizer

        # ########### #
        # Other Stuff
        # ########### #
        self.eval_statistics = None
        self._epoch_plotter = epoch_plotter
        self.render_eval_paths = render_eval_paths
        # Evaluation Sampler (One for each unintentional)
        self.eval_samplers = [
            InPlacePathSampler(env=env,
                               policy=MultiPolicySelector(self._u_policy, idx),
                               max_samples=self.num_steps_per_eval,
                               max_path_length=self.max_path_length,)
            for idx in range(self._n_unintentional)
        ]

        # TODO: REMOVE THIS TEMPO LATER
        self.tempo_entrop = [0, 0]
        self.tempo_log_policy_target = [0, 0]
        self.tempo_policy_log_std = [0, 0]
        self.tempo_policy_mean = [0, 0]
        self.tempo_qf_loss = [0, 0]
        self.tempo_vf_loss = [0, 0]
        self.tempo_rewards = [0, 0]

    def pretrain(self):
        pass

    def _do_training(self):
        # Get batch of samples
        batch = self.get_batch()

        # Update Unintentional Networks
        update_outputs = self._update_unintentional_soft_networks(batch)
        self._print_soft_statistics(update_outputs)

        # Update Intentional Network
        update_outputs = self._update_intentional_soft_networks(batch)

    def _update_intentional_soft_networks(self, batch):
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        rewards = batch['rewards']
        terminals = batch['terminals']

        policy = self._i_policy
        qf = self._i_qf
        vf = self._i_vf
        target_vf = self._i_target_vf
        policy_optimizer = self._i_policy_optimizer
        qf_optimizer = self._i_qf_optimizer
        vf_optimizer = self._i_vf_optimizer

        q_pred = qf(obs, actions)
        v_pred = vf(obs)

        new_actions, pol_dict = policy(obs, return_log_prob=True)

        log_pi = pol_dict['log_action']

        target_v_values = target_vf(next_obs)

        """
        QF Loss
        """
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
        # q_new_actions = qf(obs, new_actions)
        log_policy_target = q_new_actions - v_pred
        policy_loss = (
                log_pi * (log_pi - log_policy_target).detach()
        ).mean()
        # log_policy_target = q_new_actions - v_pred.detach()
        # policy_loss = -log_policy_target.mean()

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

        self._update_v_target_network(vf=vf, target_vf=target_vf)

        return policy_loss, qf_loss, vf_loss

    def _update_unintentional_soft_networks(self, batch):
        # terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        policy = self._u_policy
        qf = self._u_qf
        vf = self._u_vf
        target_vf = self._u_target_vf
        policy_optimizer = self._u_policy_optimizer
        qf_optimizer = self._u_qf_optimizer
        vf_optimizer = self._u_vf_optimizer

        all_q_pred = qf(obs, actions)[0]
        all_v_pred = vf(obs)[0]

        all_new_actions, all_policy_infos = policy(obs, return_log_prob=True)
        all_policy_mean = all_policy_infos['mean']
        all_policy_log_std = all_policy_infos['log_std']
        all_log_pi = all_policy_infos['log_prob']
        all_pre_tanh_value = all_policy_infos['pre_tanh_value']

        all_target_v_values = target_vf(next_obs)[0]

        qf_loss = 0
        vf_loss = 0
        policy_loss = 0

        for uu in range(self._n_unintentional):
            rewards = batch['reward_vectors'][:, uu].unsqueeze(-1) \
                         * self.reward_scale
            terminals = batch['terminal_vectors'][:, uu].unsqueeze(-1)

            target_v_values = all_target_v_values[uu]
            q_pred = all_q_pred[uu]
            new_actions = all_new_actions[uu]
            v_pred = all_v_pred[uu]
            policy_mean = all_policy_mean[uu]
            log_pi = all_log_pi[uu]
            policy_log_std = all_policy_log_std[uu]
            pre_tanh_value = all_pre_tanh_value[uu]

            """
            QF Loss
            """
            q_target = rewards + (1. - terminals) * self.discount * target_v_values
            qf_loss += self._qf_criterion(q_pred, q_target.detach())

            """
            VF Loss
            """
            q_new_actions = qf(obs, new_actions)[0][uu]
            v_target = q_new_actions - log_pi
            vf_loss += self._vf_criterion(v_pred, v_target.detach())

            """
            Policy Loss
            """
            log_policy_target = q_new_actions - v_pred
            policy_loss += (
                    log_pi * (log_pi - log_policy_target).detach()
            ).mean()

            mean_reg_loss = self._policy_mean_reg_weight * (policy_mean ** 2).mean()
            std_reg_loss = self._policy_std_reg_weight * (policy_log_std ** 2).mean()
            pre_activation_reg_loss = self._policy_pre_activation_weight * (
                (pre_tanh_value**2).sum(dim=1).mean()
            )
            policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss

            policy_loss += policy_reg_loss

            self.tempo_entrop[uu] = -log_pi.mean()
            self.tempo_log_policy_target[uu] = log_policy_target.mean()
            self.tempo_policy_log_std[uu] = policy_log_std.mean()
            self.tempo_policy_mean[uu] = policy_mean.mean()
            self.tempo_qf_loss[uu] = self._qf_criterion(q_pred, q_target.detach())
            self.tempo_vf_loss[uu] = self._vf_criterion(v_pred, v_target.detach())
            self.tempo_rewards[uu] = rewards.mean()

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

        self._update_v_target_network(vf=vf, target_vf=target_vf)

        return policy_loss, qf_loss, vf_loss

    def _print_soft_statistics(self, update_outputs, unint_idx=None):
        policy_loss = update_outputs[0]
        qf_loss = update_outputs[1]
        vf_loss = update_outputs[2]

        stats_string = 'ACCUM'

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

        # TODO: REMOVE THIS LATER
        for uu in range(self._n_unintentional):
            self.eval_statistics['[U-%02d] Policy Entropy' % uu] = \
                np.mean(ptu.get_numpy(self.tempo_entrop[uu]))
            self.eval_statistics['[U-%02d] Log Policy Target' % uu] = \
                np.mean(ptu.get_numpy(self.tempo_log_policy_target[uu]))
            self.eval_statistics['[U-%02d] Policy Log Std' % uu] = \
                np.mean(ptu.get_numpy(self.tempo_policy_log_std[uu]))
            self.eval_statistics['[U-%02d] Policy Mean' % uu] = \
                np.mean(ptu.get_numpy(self.tempo_policy_mean[uu]))
            self.eval_statistics['[U-%02d] Qf Loss' % uu] = \
                np.mean(ptu.get_numpy(self.tempo_qf_loss[uu]))
            self.eval_statistics['[U-%02d] Vf Loss' % uu] = \
                np.mean(ptu.get_numpy(self.tempo_vf_loss[uu]))
            self.eval_statistics['[U-%02d] Rewards' % uu] = \
                np.mean(ptu.get_numpy(self.tempo_rewards[uu]))

    @property
    def networks(self):
        if self._i_target_vf is None:
            target_i_vf = []
        else:
            target_i_vf = [self._i_target_vf]

        return [self._i_policy] + [self._u_policy] + \
               [self._i_qf] + [self._u_qf] + \
               [self._i_vf] + [self._u_vf] + \
               target_i_vf + [self._u_target_vf]

    def _update_v_target_network(self, vf, target_vf):
        # print('YEEEE', self.soft_target_tau)
        ptu.soft_update_from_to(vf, target_vf, self.soft_target_tau)

    def get_epoch_snapshot(self, epoch):
        if self._epoch_plotter is not None:
            self._epoch_plotter.draw()
            self._epoch_plotter.save_figure(epoch)

        snapshot = super(IUMultiSAC, self).get_epoch_snapshot(epoch)
        snapshot.update(
            policy=self._i_policy,
            u_policy=self._u_policy,
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



