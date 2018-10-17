"""
This has been adapted from Vitchyr Pong's Deep Deterministic Policy Gradient
https://github.com/vitchyr/rlkit
"""

from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
import torch.optim as optim

import robolearn.torch.pytorch_util as ptu
from robolearn.core.eval_util import create_stats_ordered_dict
from robolearn.policies.random_policy import RandomPolicy
from robolearn.utils.samplers import rollout
from robolearn.torch.rl_algos.torch_incremental_rl_algorithm \
    import TorchIncrementalRLAlgorithm
from robolearn.torch.data_management import TorchFixedNormalizer


class DDPG(TorchIncrementalRLAlgorithm):
    """
    Deep Deterministic Policy Gradient
    """

    def __init__(
            self,
            env,
            qf,
            policy,
            exploration_policy,

            replay_buffer,
            batch_size=1024,

            policy_learning_rate=1e-4,
            qf_learning_rate=1e-3,
            qf_weight_decay=0,
            target_hard_update_period=1000,
            tau=1e-2,
            use_soft_update=False,
            qf_criterion=None,
            residual_gradient_weight=0,
            epoch_discount_schedule=None,
            eval_with_target_policy=False,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,

            obs_normalizer: TorchFixedNormalizer=None,
            action_normalizer: TorchFixedNormalizer=None,
            num_paths_for_normalization=0,

            min_q_value=-np.inf,
            max_q_value=np.inf,

            save_replay_buffer=False,
            **kwargs
    ):
        """

        :param env:
        :param qf:
        :param policy:
        :param exploration_policy:
        :param policy_learning_rate:
        :param qf_learning_rate:
        :param qf_weight_decay:
        :param target_hard_update_period:
        :param tau:
        :param use_soft_update:
        :param qf_criterion: Loss function to use for the q function. Should
        be a function that takes in two inputs (y_predicted, y_target).
        :param residual_gradient_weight: c, float between 0 and 1. The gradient
        used for training the Q function is then
            (1-c) * normal td gradient + c * residual gradient
        :param epoch_discount_schedule: A schedule for the discount factor
        that varies with the epoch.
        :param kwargs:
        """
        self.target_policy = policy.copy()
        if eval_with_target_policy:
            eval_policy = self.target_policy
        else:
            eval_policy = policy
        super(DDPG, self).__init__(
            env,
            exploration_policy,
            eval_policy=eval_policy,
            **kwargs
        )
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.qf = qf
        self.policy = policy
        self.policy_learning_rate = policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.qf_weight_decay = qf_weight_decay
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update
        self.residual_gradient_weight = residual_gradient_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.qf_criterion = qf_criterion
        self.epoch_discount_schedule = epoch_discount_schedule
        self.obs_normalizer = obs_normalizer
        self.action_normalizer = action_normalizer
        self.num_paths_for_normalization = num_paths_for_normalization
        self.min_q_value = min_q_value
        self.max_q_value = max_q_value

        self.target_qf = self.qf.copy()
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=self.qf_learning_rate,
        )

        # Replay Buffer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.save_replay_buffer = save_replay_buffer

        # Optimizer
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=self.policy_learning_rate,
        )
        self.eval_statistics = None

        # Useful Variables for logging
        self.logging_qf_loss = np.zeros(self.num_train_steps_per_epoch)
        self.logging_policy_loss = np.zeros(self.num_train_steps_per_epoch)
        self.logging_raw_policy_loss = np.zeros(self.num_train_steps_per_epoch)
        self.logging_q_pred = np.zeros(
            (self.num_train_steps_per_epoch, batch_size)
        )
        self.logging_q_target = np.zeros(
            (self.num_train_steps_per_epoch, batch_size)
        )
        self.logging_bellman_errors = np.zeros(
            (self.num_train_steps_per_epoch, batch_size)
        )
        self.logging_policy_actions = np.zeros(
            (self.num_train_steps_per_epoch, batch_size, self.env.action_dim)
        )

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy operations.
        """
        if self.policy_pre_activation_weight > 0:
            policy_actions, policy_info = self.policy(
                obs, return_preactivations=True,
            )
            pre_tanh_value = policy_info['pre_tanh_value']
            pre_activation_policy_loss = (
                (pre_tanh_value**2).sum(dim=1).mean()
            )
            q_output = self.qf(obs, policy_actions)[0]
            raw_policy_loss = - q_output.mean()
            policy_loss = (
                raw_policy_loss +
                pre_activation_policy_loss * self.policy_pre_activation_weight
            )
        else:
            policy_actions = self.policy(obs)[0]
            q_output = self.qf(obs, policy_actions)[0]
            raw_policy_loss = policy_loss = - q_output.mean()

        """
        Critic operations.
        """

        next_actions = self.target_policy(next_obs)[0]
        # speed up computation by not backpropping these gradients
        next_actions.detach()
        target_q_values = self.target_qf(
            next_obs,
            next_actions,
        )[0]
        q_target = rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()
        q_target = torch.clamp(q_target, self.min_q_value, self.max_q_value)
        # Hack for ICLR rebuttal
        if hasattr(self, 'reward_type') and self.reward_type == 'indicator':
            q_target = \
                torch.clamp(q_target, -self.reward_scale/(1-self.discount), 0)
        q_pred = self.qf(obs, actions)[0]
        bellman_errors = (q_pred - q_target) ** 2
        raw_qf_loss = self.qf_criterion(q_pred, q_target)

        if self.residual_gradient_weight > 0:
            residual_next_actions = self.policy(next_obs)
            # speed up computation by not backpropping these gradients
            residual_next_actions.detach()
            residual_target_q_values = self.qf(
                next_obs,
                residual_next_actions,
            )[0]
            residual_q_target = (
                rewards
                + (1. - terminals) * self.discount * residual_target_q_values
            )
            residual_bellman_errors = (q_pred - residual_q_target) ** 2
            # noinspection PyUnresolvedReferences
            residual_qf_loss = residual_bellman_errors.mean()
            raw_qf_loss = (
                self.residual_gradient_weight * residual_qf_loss
                + (1 - self.residual_gradient_weight) * raw_qf_loss
            )

        if self.qf_weight_decay > 0:
            reg_loss = self.qf_weight_decay * sum(
                torch.sum(param ** 2)
                for param in self.qf.regularizable_parameters()
            )
            qf_loss = raw_qf_loss + reg_loss
        else:
            qf_loss = raw_qf_loss

        """
        Update Networks
        """

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self._update_target_networks()

        # Get the idx for logging
        step_idx = self._n_epoch_train_steps

        # Log data
        self.logging_qf_loss[step_idx] = ptu.get_numpy(qf_loss)
        self.logging_policy_loss[step_idx] = ptu.get_numpy(policy_loss)
        self.logging_raw_policy_loss[step_idx] = ptu.get_numpy(raw_policy_loss)
        self.logging_q_pred[step_idx] = ptu.get_numpy(q_pred).squeeze(-1)
        self.logging_q_target[step_idx] = ptu.get_numpy(q_target).squeeze(-1)
        self.logging_bellman_errors[step_idx] = \
            ptu.get_numpy(bellman_errors).squeeze(-1)
        self.logging_policy_actions[step_idx] = ptu.get_numpy(policy_actions)

    def _update_target_networks(self):
        if self.use_soft_update:
            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
            ptu.soft_update_from_to(self.qf, self.target_qf, self.tau)
        else:
            if self._n_env_steps_total % self.target_hard_update_period == 0:
                ptu.copy_model_params_from_to(self.qf, self.target_qf)
                ptu.copy_model_params_from_to(self.policy, self.target_policy)

    def get_epoch_snapshot(self, epoch):
        snapshot = super(DDPG, self).get_epoch_snapshot(epoch)
        snapshot.update(
            qf=self.qf,
            policy=self.eval_policy,
            trained_policy=self.policy,
            target_policy=self.target_policy,
            exploration_policy=self.exploration_policy,
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
            self.eval_statistics = OrderedDict()
            self.eval_statistics['QF Loss'] = \
                np.nan_to_num(np.mean(self.logging_qf_loss[:max_step]))
            self.eval_statistics['Policy Loss'] = \
                np.nan_to_num(np.mean(self.logging_policy_loss[:max_step]))
            self.eval_statistics['Raw Policy Loss'] = \
                np.nan_to_num(np.mean(self.logging_raw_policy_loss[:max_step]))
            self.eval_statistics['Preactivation Policy Loss'] = (
                    self.eval_statistics['Policy Loss'] -
                    self.eval_statistics['Raw Policy Loss']
            )
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                np.nan_to_num(np.mean(self.logging_q_pred[:max_step]))
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                np.nan_to_num(np.mean(self.logging_q_target[:max_step]))
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors',
                np.nan_to_num(np.mean(self.logging_bellman_errors[:max_step]))
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                np.nan_to_num(np.mean(self.logging_policy_actions[:max_step]))
            ))

    def evaluate(self, epoch):
        self._update_logging_data()
        TorchIncrementalRLAlgorithm.evaluate(self, epoch)

        # Reset log values
        self.logging_qf_loss.fill(0)
        self.logging_policy_loss.fill(0)
        self.logging_raw_policy_loss.fill(0)
        self.logging_q_pred.fill(0)
        self.logging_q_target.fill(0)
        self.logging_bellman_errors.fill(0)
        self.logging_policy_actions.fill(0)

    @property
    def torch_models(self):
        return [
            self.policy,
            self.qf,
            self.target_policy,
            self.target_qf,
        ]

    def pretrain(self):
        if (
            self.num_paths_for_normalization == 0
            or (self.obs_normalizer is None and self.action_normalizer is None)
        ):
            return

        pretrain_paths = []
        random_policy = RandomPolicy(self.env.action_space)
        while len(pretrain_paths) < self.num_paths_for_normalization:
            path = rollout(self.env, random_policy, self.max_path_length)
            pretrain_paths.append(path)
        ob_mean, ob_std, ac_mean, ac_std = (
            compute_normalization(pretrain_paths)
        )
        if self.obs_normalizer is not None:
            self.obs_normalizer.set_mean(ob_mean)
            self.obs_normalizer.set_std(ob_std)
            self.target_qf.obs_normalizer = self.obs_normalizer
            self.target_policy.obs_normalizer = self.obs_normalizer
        if self.action_normalizer is not None:
            self.action_normalizer.set_mean(ac_mean)
            self.action_normalizer.set_std(ac_std)
            self.target_qf.action_normalizer = self.action_normalizer
            self.target_policy.action_normalizer = self.action_normalizer

    def get_batch(self):
        batch = self.replay_buffer.random_batch(self.batch_size)

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


def compute_normalization(paths):
    obs = np.vstack([path["observations"] for path in paths])
    ob_mean = np.mean(obs, axis=0)
    ob_std = np.std(obs, axis=0)
    actions = np.vstack([path["actions"] for path in paths])
    ac_mean = np.mean(actions, axis=0)
    ac_std = np.std(actions, axis=0)
    return ob_mean, ob_std, ac_mean, ac_std
