"""
This has been adapted from Vitchyr Pong's Deep Deterministic Policy Gradient
https://github.com/vitchyr/rlkit
"""

import numpy as np
import torch
from torch import nn as nn
import torch.optim as optim

from collections import OrderedDict

import robolearn.torch.utils.pytorch_util as ptu
from robolearn.utils.eval_util import create_stats_ordered_dict
from robolearn.models.policies import RandomPolicy
from robolearn.utils.samplers import rollout

from robolearn.algorithms.rl_algos import RLAlgorithm
from robolearn.torch.algorithms.torch_algorithm import TorchAlgorithm

from robolearn.torch.utils.data_management import TorchFixedNormalizer


class DDPG(RLAlgorithm, TorchAlgorithm):
    """
    Deep Deterministic Policy Gradient (DDPG)
    """
    def __init__(
            self,
            explo_env,
            qf,
            policy,
            explo_policy,

            replay_buffer,
            batch_size=1024,
            eval_env=None,

            target_hard_update_period=1000,
            tau=1e-2,
            use_soft_update=False,
            qf_criterion=None,
            residual_gradient_weight=0,
            epoch_discount_schedule=None,
            eval_with_target_policy=False,

            policy_pre_activation_weight=0.,

            policy_lr=1e-4,
            qf_lr=1e-3,

            policy_weight_decay=0.,
            qf_weight_decay=0,

            optimizer='adam',
            # optimizer='rmsprop',
            # optimizer='sgd',
            optimizer_kwargs=None,

            obs_normalizer: TorchFixedNormalizer=None,
            action_normalizer: TorchFixedNormalizer=None,
            num_paths_for_normalization=0,

            reward_scale=1.,

            min_q_value=-np.inf,
            max_q_value=np.inf,

            save_replay_buffer=False,
            **kwargs
    ):
        """

        :param explo_env:
        :param qf:
        :param policy:
        :param explo_policy:
        :param policy_lr:
        :param qf_lr:
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
        self._target_policy = policy.copy()
        if eval_with_target_policy:
            eval_policy = self._target_policy
        else:
            eval_policy = policy
        RLAlgorithm.__init__(
            self,
            explo_env=explo_env,
            explo_policy=explo_policy,
            eval_env=eval_env,
            eval_policy=eval_policy,
            **kwargs
        )
        self.policy = policy
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update
        self.residual_gradient_weight = residual_gradient_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.epoch_discount_schedule = epoch_discount_schedule
        self.obs_normalizer = obs_normalizer
        self.action_normalizer = action_normalizer
        self.num_paths_for_normalization = num_paths_for_normalization
        self.reward_scale = reward_scale

        # Q-function
        self._qf = qf
        self._target_qf = self._qf.copy()
        self.min_q_value = min_q_value
        self.max_q_value = max_q_value
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.qf_criterion = qf_criterion

        # Replay Buffer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.save_replay_buffer = save_replay_buffer

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
        self._qf_lr = qf_lr
        self._policy_lr = policy_lr
        self._qf_weight_decay = qf_weight_decay
        self._policy_weight_decay = qf_weight_decay

        # Q-function optimizer
        self._qf_optimizer = optimizer_class(
            self._qf.parameters(),
            lr=qf_lr,
            weight_decay=qf_weight_decay,
            **optimizer_kwargs
        )

        # Policy optimizer
        self._policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
            weight_decay=policy_weight_decay,
            **optimizer_kwargs
        )

        # Useful Variables for logging
        self.log_data = dict()
        self.log_data['Raw Pol Loss'] = np.zeros(self.num_train_steps_per_epoch)
        self.log_data['Pol Loss'] = np.zeros(self.num_train_steps_per_epoch)
        self.log_data['Qf Loss'] = np.zeros(self.num_train_steps_per_epoch)
        self.log_data['Q pred'] = np.zeros(
            (self.num_train_steps_per_epoch, batch_size)
        )
        self.log_data['Q target'] = np.zeros(
            (self.num_train_steps_per_epoch, batch_size)
        )
        self.log_data['Bellman Error'] = np.zeros(
            (self.num_train_steps_per_epoch, batch_size)
        )
        self.log_data['Policy Actions'] = np.zeros(
            (self.num_train_steps_per_epoch, batch_size, self.explo_env.action_dim)
        )

    def pretrain(self, n_pretrain_samples):
        if (
                self.num_paths_for_normalization == 0
                or (self.obs_normalizer is None and self.action_normalizer is None)
        ):
            observation = self.explo_env.reset()
            for ii in range(n_pretrain_samples):
                action = self.explo_env.action_space.sample()
                # Interact with environment
                next_ob, reward, terminal, env_info = (
                    self.explo_env.step(action)
                )
                agent_info = None

                # Increase counter
                self._n_env_steps_total += 1
                # Create np.array of obtained terminal and reward
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
                    self.explo_env.reset()
        else:
            pretrain_paths = []
            random_policy = RandomPolicy(self.explo_env.action_space)
            while len(pretrain_paths) < self.num_paths_for_normalization:
                path = rollout(self.explo_env, random_policy, self.max_path_length)
                pretrain_paths.append(path)
            ob_mean, ob_std, ac_mean, ac_std = (
                compute_normalization(pretrain_paths)
            )
            if self.obs_normalizer is not None:
                self.obs_normalizer.set_mean(ob_mean)
                self.obs_normalizer.set_std(ob_std)
                self._target_qf.obs_normalizer = self.obs_normalizer
                self._target_policy.obs_normalizer = self.obs_normalizer
            if self.action_normalizer is not None:
                self.action_normalizer.set_mean(ac_mean)
                self.action_normalizer.set_std(ac_std)
                self._target_qf.action_normalizer = self.action_normalizer
                self._target_policy.action_normalizer = self.action_normalizer

    def _do_training(self):
        # Get batch of samples
        batch = self.get_batch()

        # Get common data from batch
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        rewards = batch['rewards'] * self.reward_scale
        terminals = batch['terminals']

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
            q_output = self._qf(obs, policy_actions)[0]
            raw_policy_loss = - q_output.mean()
            policy_loss = (
                raw_policy_loss +
                pre_activation_policy_loss * self.policy_pre_activation_weight
            )
        else:
            policy_actions = self.policy(obs)[0]
            q_output = self._qf(obs, policy_actions)[0]
            raw_policy_loss = policy_loss = - q_output.mean()

        """
        Critic operations.
        """

        next_actions = self._target_policy(next_obs)[0]
        # speed up computation by not backpropping these gradients
        next_actions.detach()
        target_q_values = self._target_qf(
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
        q_pred = self._qf(obs, actions)[0]
        bellman_errors = (q_pred - q_target) ** 2
        qf_loss = self.qf_criterion(q_pred, q_target)

        if self.residual_gradient_weight > 0:
            residual_next_actions = self.policy(next_obs)
            # speed up computation by not backpropping these gradients
            residual_next_actions.detach()
            residual_target_q_values = self._qf(
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
            qf_loss = (
                self.residual_gradient_weight * residual_qf_loss
                + (1 - self.residual_gradient_weight) * qf_loss
            )

        """
        Update Networks
        """

        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        self._qf_optimizer.zero_grad()
        qf_loss.backward()
        self._qf_optimizer.step()

        # ###################### #
        # Update Target Networks #
        # ###################### #
        if self.use_soft_update:
            ptu.soft_update_from_to(self.policy, self._target_policy, self.tau)
            ptu.soft_update_from_to(self._qf, self._target_qf, self.tau)
        else:
            if self._n_env_steps_total % self.target_hard_update_period == 0:
                ptu.copy_model_params_from_to(self._qf, self._target_qf)
                ptu.copy_model_params_from_to(self.policy, self._target_policy)


        # ############### #
        # LOG Useful Data #
        # ############### #
        step_idx = self._n_epoch_train_steps
        self.log_data['Qf Loss'][step_idx] = ptu.get_numpy(qf_loss)
        self.log_data['Pol Loss'][step_idx] = ptu.get_numpy(policy_loss)
        self.log_data['Raw Pol Loss'][step_idx] = ptu.get_numpy(raw_policy_loss)
        self.log_data['Q pred'][step_idx] = ptu.get_numpy(q_pred).squeeze(-1)
        self.log_data['Q target'][step_idx] = ptu.get_numpy(q_target).squeeze(-1)
        self.log_data['Bellman Error'][step_idx] = \
            ptu.get_numpy(bellman_errors).squeeze(-1)
        self.log_data['Policy Actions'][step_idx] = ptu.get_numpy(policy_actions)

    def _not_do_training(self):
        return

    @property
    def torch_models(self):
        networks_list = [
            self.policy,
            self._qf,
            self._target_policy,
            self._target_qf,
        ]

        return networks_list

    def get_epoch_snapshot(self, epoch):
        """
        Stuff to save in file.
        Args:
            epoch:

        Returns:

        """
        snapshot = RLAlgorithm.get_epoch_snapshot(self, epoch)

        snapshot.update(
            policy=self.eval_policy,
            trained_policy=self.policy,
            target_policy=self._target_policy,
            exploration_policy=self.explo_policy,
            qf=self._qf,
            target_qf=self._target_qf,
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
            np.nan_to_num(np.mean(
               self.log_data['Qf Loss'][:max_step]
            ))
        self.eval_statistics['Policy Loss'] = \
            np.nan_to_num(np.mean(
                self.log_data['Pol Loss'][:max_step]
            ))
        self.eval_statistics['Raw Policy Loss'] = \
            np.nan_to_num(np.mean(
                self.log_data['Raw Pol Loss'][:max_step]
            ))
        self.eval_statistics['Preactivation Policy Loss'] = (
                self.eval_statistics['Policy Loss'] -
                self.eval_statistics['Raw Policy Loss']
        )
        self.eval_statistics.update(create_stats_ordered_dict(
            'Q Predictions',
            np.nan_to_num(np.mean(self.log_data['Q pred'][:max_step]))
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'Q Targets',
            np.nan_to_num(np.mean(
                self.log_data['Q target'][:max_step]
            ))
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'Bellman Errors',
            np.nan_to_num(np.mean(
                self.log_data['Bellman Error'][:max_step]
            ))
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'Policy Action',
            np.nan_to_num(np.mean(
                self.log_data['Policy Actions'][:max_step]
            ))
        ))

    def evaluate(self, epoch):
        self._update_logging_data()
        RLAlgorithm.evaluate(self, epoch)

        # Reset log_data
        for key in self.log_data.keys():
            self.log_data[key].fill(0)

    def get_batch(self):
        batch = self.replay_buffer.random_batch(self.batch_size)

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

        RLAlgorithm._handle_step(
            self,
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            agent_info=agent_info,
            env_info=env_info,
        )

    def _end_rollout(self):
        """
        Implement anything that needs to happen after every rollout.
        """

        self.replay_buffer.terminate_episode()

        RLAlgorithm._end_rollout(self)


def compute_normalization(paths):
    obs = np.vstack([path["observations"] for path in paths])
    ob_mean = np.mean(obs, axis=0)
    ob_std = np.std(obs, axis=0)
    actions = np.vstack([path["actions"] for path in paths])
    ac_mean = np.mean(actions, axis=0)
    ac_std = np.std(actions, axis=0)
    return ob_mean, ob_std, ac_mean, ac_std
