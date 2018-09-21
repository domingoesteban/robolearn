"""
This has been adapted from Vitchyr Pong's Soft Actor Critic implementation.
https://github.com/vitchyr/rlkit
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import robolearn.torch.pytorch_util as ptu
from robolearn.core import logger, eval_util
from robolearn.core.eval_util import create_stats_ordered_dict

from robolearn.torch.rl_algos.torch_incremental_rl_algorithm \
    import TorchIncrementalRLAlgorithm
from robolearn.policies import MakeDeterministic
from torch.autograd import Variable

from tensorboardX import SummaryWriter
from robolearn.core import logger


class SoftActorCritic(TorchIncrementalRLAlgorithm):
    def __init__(
            self,
            env,
            policy,
            qf,
            vf,

            replay_buffer,
            batch_size=1024,
            eval_env=None,

            qf2=None,
            reparameterize=True,
            action_prior='uniform',
            entropy_scale=1.,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,

            policy_mean_regu_weight=1e-3,
            policy_std_regu_weight=1e-3,
            policy_pre_activation_weight=0.,

            policy_weight_decay=0.,
            q_weight_decay=0.,
            v_weight_decay=0.,

            optimizer_class=optim.Adam,
            # optimizer_class=optim.SGD,
            amsgrad=True,

            soft_target_tau=1e-2,
            target_update_interval=1,

            save_replay_buffer=False,
            epoch_plotter=None,
            render_eval_paths=False,
            eval_deterministic=True,
            **kwargs
    ):

        # Exploration Policy
        self._policy = policy

        # # TODO: TEMPORALLY REDUCING VALUES
        # for name, param in self._policy.named_parameters():
        #     # print(name, param.mean())
        #     if not name.startswith('last_fc_log_std'):
        #         param.data.mul_(0.5)
        #     # print(name, param.mean())

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

        self._reparameterize = reparameterize
        assert self._reparameterize == self._policy.reparameterize
        self._action_prior = action_prior
        self._entropy_scale = entropy_scale

        # Q-function and V-function
        self._qf = qf
        self._qf2 = qf2
        self._vf = vf
        self._target_vf = vf.copy()

        # Soft-update rate for target Vf
        self.soft_target_tau = soft_target_tau
        self._target_update_interval = target_update_interval

        # Replay Buffer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.save_replay_buffer = save_replay_buffer

        # Q-function and V-funciton Optimization Criteria
        self._qf_criterion = nn.MSELoss()
        self._vf_criterion = nn.MSELoss()

        # Q-function(s) optimizer(s)
        self._qf_optimizer = optimizer_class(
            self._qf.parameters(),
            lr=qf_lr,
            amsgrad=amsgrad,
            weight_decay=q_weight_decay,
        )
        if self._qf2 is None:
            self._qf2_optimizer = None
        else:
            self._qf2_optimizer = optimizer_class(
                self._qf2.parameters(),
                lr=qf_lr,
                amsgrad=amsgrad,
                weight_decay=q_weight_decay,
            )

        # V-function optimizer
        self._vf_optimizer = optimizer_class(
            self._vf.parameters(),
            lr=vf_lr,
            amsgrad=amsgrad,
            weight_decay=v_weight_decay,
        )

        # Policy optimizer
        self._policy_optimizer = optimizer_class(
            self._policy.parameters(),
            lr=policy_lr,
            weight_decay=policy_weight_decay,
        )

        # Policy regularization coefficients (weights)
        self.policy_mean_reg_weight = policy_mean_regu_weight
        self.policy_std_reg_weight = policy_std_regu_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight

        # Other Stuff
        self.eval_statistics = None
        self._epoch_plotter = epoch_plotter
        self.render_eval_paths = render_eval_paths

        # Useful Variables for logging
        self.logging_pol_kl_loss = np.zeros(self.num_env_steps_per_epoch)
        self.logging_qf_loss = np.zeros(self.num_env_steps_per_epoch)
        self.logging_qf2_loss = np.zeros(self.num_env_steps_per_epoch)
        self.logging_vf_loss = np.zeros(self.num_env_steps_per_epoch)
        self.logging_rewards = np.zeros(self.num_env_steps_per_epoch)
        self.logging_policy_entropy = np.zeros(self.num_env_steps_per_epoch)
        self.logging_policy_log_std = np.zeros(self.num_env_steps_per_epoch)
        self.logging_policy_mean = np.zeros(self.num_env_steps_per_epoch)

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

            if terminal:
                self.env.reset()

    def _do_training(self):
        # Get batch of samples
        batch = self.get_batch()

        # Get data from batch
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        rewards = batch['rewards']
        terminals = batch['terminals']

        # Get the idx for logging
        step_idx = self._n_epoch_train_steps

        # ########### #
        # Critic Step #
        # ########### #
        # Calculate QF Loss (Soft Bellman Eq.)
        v_value_next = self._target_vf(next_obs)[0]
        q_pred = self._qf(obs, actions)[0]
        q_target = rewards + (1. - terminals) * self.discount * v_value_next
        qf_loss = 0.5*self._qf_criterion(q_pred, q_target.detach())

        # Update Intentional Q-value
        self._qf_optimizer.zero_grad()
        qf_loss.backward()
        self._qf_optimizer.step()

        if self._qf2 is not None:
            q2_pred = self._qf2(obs, actions)[0]
            q2_target = rewards + (1. - terminals) * self.discount * v_value_next
            qf2_loss = 0.5*self._qf_criterion(q2_pred, q2_target.detach())

            # Update Intentional Q-value
            self._qf2_optimizer.zero_grad()
            qf2_loss.backward()
            self._qf2_optimizer.step()

        # ########## #
        # Actor Step #
        # ########## #
        # Calculate Intentional Policy Loss
        new_actions, policy_info = self._policy(obs, return_log_prob=True)
        log_pi = policy_info['log_prob'] * self._entropy_scale
        policy_mean = policy_info['mean']
        policy_log_std = policy_info['log_std']
        pre_tanh_value = policy_info['pre_tanh_value']

        if self._action_prior == 'normal':
            raise NotImplementedError
        else:
            policy_prior_log_probs = 0.0

        v_pred = self._vf(obs)[0]
        q_new_actions = self._qf(obs, new_actions)[0]

        if self._qf2 is not None:
            q2_new_actions = self._qf2(obs, new_actions)[0]
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
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_regu_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss

        policy_loss = policy_kl_loss + policy_regu_loss

        # Update Policy
        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # ############### #
        # V-function Step #
        # ############### #
        # Calculate Intentional Vf Loss
        v_target = q_new_actions - log_pi + policy_prior_log_probs
        vf_loss = 0.5*self._vf_criterion(v_pred, v_target.detach())

        # Update V-value
        self._vf_optimizer.zero_grad()
        vf_loss.backward()
        self._vf_optimizer.step()

        # Update V Target Network
        if self._n_train_steps_total % self._target_update_interval == 0:
            self._update_target_network()

        # ########################### #
        # LOG Useful Intentional Data #
        # ########################### #
        self.logging_policy_entropy[step_idx] = \
            ptu.get_numpy(-log_pi.mean(dim=0))
        self.logging_policy_log_std[step_idx] = ptu.get_numpy(policy_log_std.mean())
        self.logging_policy_mean[step_idx] = ptu.get_numpy(policy_mean.mean())
        self.logging_qf_loss[step_idx] = ptu.get_numpy(qf_loss)
        self.logging_vf_loss[step_idx] = ptu.get_numpy(vf_loss)
        self.logging_pol_kl_loss[step_idx] = ptu.get_numpy(policy_kl_loss)
        self.logging_rewards[step_idx] = ptu.get_numpy(rewards.mean(dim=0))

        self._summary_writer.add_scalar('Training/qf_loss',
                                        ptu.get_numpy(qf_loss),
                                        self._n_env_steps_total)
        if self._qf2 is not None:
            self._summary_writer.add_scalar('Training/qf2_loss',
                                            ptu.get_numpy(qf2_loss),
                                            self._n_env_steps_total)

        self._summary_writer.add_scalar('Training/vf_loss',
                                        ptu.get_numpy(vf_loss),
                                        self._n_env_steps_total)
        self._summary_writer.add_scalar('Training/avg_reward',
                                        ptu.get_numpy(rewards.mean()),
                                        self._n_env_steps_total)
        self._summary_writer.add_scalar('Training/avg_advantage',
                                        ptu.get_numpy(advantages_new_actions.mean()),
                                        self._n_env_steps_total)
        self._summary_writer.add_scalar('Training/policy_loss',
                                        ptu.get_numpy(policy_loss),
                                        self._n_env_steps_total)
        self._summary_writer.add_scalar('Training/policy_entropy',
                                        ptu.get_numpy(-log_pi.mean()),
                                        self._n_env_steps_total)
        self._summary_writer.add_scalar('Training/policy_mean',
                                        ptu.get_numpy(policy_mean.mean()),
                                        self._n_env_steps_total)
        self._summary_writer.add_scalar('Training/policy_std',
                                        np.exp(ptu.get_numpy(policy_log_std.mean())),
                                        self._n_env_steps_total)
        self._summary_writer.add_scalar('Training/q_vals',
                                        ptu.get_numpy(q_new_actions.mean()),
                                        self._n_env_steps_total)

        if self._n_env_steps_total % 500 == 0:
            for name, param in self._policy.named_parameters():
                self._summary_writer.add_histogram('policy/'+name,
                                                   param.data.cpu().numpy(),
                                                   self._n_env_steps_total)
                self._summary_writer.add_histogram('policy_grad/'+name,
                                                   param.grad.data.cpu().numpy(),
                                                   self._n_env_steps_total)

            for name, param in self._qf.named_parameters():
                self._summary_writer.add_histogram('qf/'+name,
                                                   param.data.cpu().numpy(),
                                                   self._n_env_steps_total)
                self._summary_writer.add_histogram('qf_grad/'+name,
                                                   param.grad.data.cpu().numpy(),
                                                   self._n_env_steps_total)
            if self._qf2 is not None:
                for name, param in self._qf2.named_parameters():
                    self._summary_writer.add_histogram('qf2/'+name,
                                                       param.data.cpu().numpy(),
                                                       self._n_env_steps_total)
                    self._summary_writer.add_histogram('qf2_grad/'+name,
                                                       param.grad.data.cpu().numpy(),
                                                       self._n_env_steps_total)

            for name, param in self._vf.named_parameters():
                self._summary_writer.add_histogram('vf/'+name,
                                                   param.data.cpu().numpy(),
                                                   self._n_env_steps_total)
                self._summary_writer.add_histogram('vf_grad/'+name,
                                                   param.grad.data.cpu().numpy(),
                                                   self._n_env_steps_total)

            for name, param in self._target_vf.named_parameters():
                self._summary_writer.add_histogram('vf_target/'+name,
                                                   param.cpu().data.numpy(),
                                                   self._n_env_steps_total)




        # """
        # Save some statistics for eval
        # """
        # if self.eval_statistics is None:
        #     """
        #     Eval should set this to None.
        #     This way, these statistics are only computed for one batch.
        #     """
        #     self.eval_statistics = OrderedDict()
        #     self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
        #     self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
        #     self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
        #         policy_loss
        #     ))
        #     self.eval_statistics.update(create_stats_ordered_dict(
        #         'Q Predictions',
        #         ptu.get_numpy(q_pred),
        #     ))
        #     self.eval_statistics.update(create_stats_ordered_dict(
        #         'V Predictions',
        #         ptu.get_numpy(v_pred),
        #     ))
        #     self.eval_statistics.update(create_stats_ordered_dict(
        #         'Log Pis',
        #         ptu.get_numpy(log_pi),
        #     ))
        #     self.eval_statistics.update(create_stats_ordered_dict(
        #         'Policy mu',
        #         ptu.get_numpy(policy_mean),
        #     ))
        #     self.eval_statistics.update(create_stats_ordered_dict(
        #         'Policy log std',
        #         ptu.get_numpy(policy_log_std),
        #     ))

    def _do_not_training(self):
        return

    @property
    def networks(self):
        networks_list = [
            self._policy,
            self._qf,
            self._vf,
            self._target_vf,
        ]
        if self._qf2 is not None:
            networks_list.append(self._qf2)

        return networks_list

    def _update_target_network(self):
        ptu.soft_update_from_to(self._vf, self._target_vf, self.soft_target_tau)

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

        snapshot = super(SoftActorCritic, self).get_epoch_snapshot(epoch)

        snapshot.update(
            policy=self._policy,
            qf=self._qf,
            vf=self._vf,
            target_vf=self._target_vf,
            qf2=self._qf2,  # It could be None
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

        # Intentional info
        self.eval_statistics['[I] Policy Entropy'] = \
            np.nan_to_num(np.mean(self.logging_policy_entropy[:max_step]))
        self.eval_statistics['[I] Policy Std'] = \
            np.nan_to_num(np.mean(np.exp(self.logging_policy_log_std[:max_step])))
        self.eval_statistics['[I] Policy Mean'] = \
            np.nan_to_num(np.mean(self.logging_policy_mean[:max_step]))
        self.eval_statistics['[I] Qf Loss'] = \
            np.nan_to_num(np.mean(self.logging_qf_loss[:max_step]))
        self.eval_statistics['[I] Vf Loss'] = \
            np.nan_to_num(np.mean(self.logging_vf_loss[:max_step]))
        self.eval_statistics['[I] Pol KL Loss'] = \
            np.nan_to_num(np.mean(self.logging_pol_kl_loss[:max_step]))
        self.eval_statistics['[I] Rewards'] = \
            np.nan_to_num(np.mean(self.logging_rewards[:max_step]))

    def evaluate(self, epoch):
        statistics = OrderedDict()
        self._update_logging_data()
        statistics.update(self.eval_statistics)
        self.eval_statistics = None

        logger.log("Collecting samples for evaluation")
        test_paths = self.eval_sampler.obtain_samples()
        statistics.update(eval_util.get_generic_path_information(
            test_paths, stat_prefix="[I] Test",
        ))
        average_return = eval_util.get_average_returns(test_paths)
        statistics['[I] Test AverageReturn'] = average_return * self.reward_scale

        statistics.update(eval_util.get_generic_path_information(
            test_paths, stat_prefix="Exploration",
        ))

        self._summary_writer.add_scalar('Evaluation/avg_return',
                                        statistics['[I] Test AverageReturn'],
                                        self._n_epochs)

        self._summary_writer.add_scalar('Evaluation/avg_reward',
                                        statistics['[I] Test Rewards Mean'] * self.reward_scale,
                                        self._n_epochs)

        if hasattr(self.env, "log_diagnostics"):
            # TODO: CHECK ENV LOG_DIAGNOSTICS
            print('TODO: WE NEED LOG_DIAGNOSTICS IN ENV')

        # Record the data
        for key, value in statistics.items():
            logger.record_tabular(key, value)

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
