"""
Based on ...
"""

import numpy as np
import torch
import torch.optim as optim

from collections import OrderedDict
from itertools import chain

import robolearn.torch.utils.pytorch_util as ptu
from robolearn.utils.logging import logger
from robolearn.utils import eval_util

from robolearn.algorithms.rl_algos import RLAlgorithm
from robolearn.torch.algorithms.torch_algorithm import TorchAlgorithm

from robolearn.models.policies import MakeDeterministic
from robolearn.utils.data_management.normalizer import RunningNormalizer

import tensorboardX


class PPO(RLAlgorithm, TorchAlgorithm):
    """
    Proximal Policy Optimization
    """

    def __init__(
            self,
            env,
            policy,
            qf,

            replay_buffer,
            normalize_obs=False,
            eval_env=None,

            action_prior='uniform',

            entropy_scale=1.,

            policy_lr=1e-4,
            qf_lr=1e-3,

            policy_weight_decay=0,
            qf_weight_decay=0,

            residual_gradient_weight=0,
            epoch_discount_schedule=None,
            policy_mean_regu_weight=1e-3,
            policy_std_regu_weight=1e-3,
            policy_pre_activation_weight=0.,

            optimizer='adam',
            # optimizer='rmsprop',
            # optimizer='sgd',
            optimizer_kwargs=None,

            target_hard_update_period=1000,
            tau=1e-2,
            use_soft_update=False,

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

        RLAlgorithm.__init__(
            self,
            env=env,
            exploration_policy=self._policy,
            eval_env=eval_env,
            eval_policy=eval_policy,
            obs_normalizer=self._obs_normalizer,
            **kwargs
        )

        # Important algorithm hyperparameters
        self._action_prior = action_prior
        self._entropy_scale = entropy_scale

        # Q-function
        self._qf = qf

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

        # Q-function(s) optimizer(s)
        self._qf_optimizer = optimizer_class(
            self._qf.parameters(),
            lr=qf_lr,
            weight_decay=0,
            **optimizer_kwargs
        )

        # Policy optimizer
        self._policy_optimizer = optimizer_class(
            self._policy.parameters(),
            lr=policy_lr,
            weight_decay=0,
            **optimizer_kwargs
        )

        # Policy regularization coefficients (weights)
        self._policy_mean_regu_weight = policy_mean_regu_weight
        self._policy_std_regu_weight = policy_std_regu_weight
        self._policy_pre_activation_weight = policy_pre_activation_weight

        # Useful Variables for logging
        self.logging_pol_kl_loss = np.zeros(self.num_train_steps_per_epoch)
        self.logging_qf_loss = np.zeros(self.num_train_steps_per_epoch)
        self.logging_rewards = np.zeros(self.num_train_steps_per_epoch)
        self.logging_policy_entropy = np.zeros(self.num_train_steps_per_epoch)
        self.logging_policy_log_std = np.zeros((self.num_train_steps_per_epoch,
                                                self.explo_env.action_dim))
        self.logging_policy_mean = np.zeros((self.num_train_steps_per_epoch,
                                             self.explo_env.action_dim))

        self._log_tensorboard = log_tensorboard
        self._summary_writer = tensorboardX.SummaryWriter(log_dir=logger.get_snapshot_dir())

    def pretrain(self, n_pretrain_samples):
        # We do not require any pretrain (I think...)
        pass

    def _do_training(self):
        # Get batch of samples
        # batch = self.get_batch()
        cosa = self.get_exploration_paths()

        # # Get common data from batch
        # rewards = batch['rewards']
        # terminals = batch['terminals']
        # obs = batch['observations']
        # actions = batch['actions']
        # next_obs = batch['next_observations']

        # ########################### #
        # LOG Useful Intentional Data #
        # ########################### #

        if self._log_tensorboard:
            pass

    def _not_do_training(self):
        return

    @property
    def torch_models(self):
        networks_list = list()
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

        snapshot = RLAlgorithm.get_epoch_snapshot(self, epoch)

        snapshot.update(
            policy=self._policy,
            qf=self._qf,
        )

        if self.explo_env.online_normalization or self.explo_env.normalize_obs:
            snapshot.update(
                obs_mean=self.explo_env.obs_mean,
                obs_var=self.explo_env.obs_var,
            )

        # Observation Normalizer
        snapshot.update(
            obs_normalizer=self._obs_normalizer,
        )

        # # Replay Buffer
        # if self.save_replay_buffer:
        #     snapshot.update(
        #         replay_buffer=self.replay_buffer,
        #     )

        return snapshot

    def _update_logging_data(self):
        max_step = max(self._n_epoch_train_steps, 1)

        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()

    def evaluate(self, epoch):
        RLAlgorithm.evaluate(self, epoch)

    def get_batch(self):
        pass

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
        # # Add to replay buffer
        # self.replay_buffer.add_sample(
        #     observation=observation,
        #     action=action,
        #     reward=reward,
        #     terminal=terminal,
        #     next_observation=next_observation,
        #     agent_info=agent_info,
        #     env_info=env_info,
        # )

        # Update observation normalizer (if applicable)
        if self._obs_normalizer is not None:
            self._obs_normalizer.update(np.array([observation]))

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

        # self.replay_buffer.terminate_episode()

        RLAlgorithm._end_rollout(self)

