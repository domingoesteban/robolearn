"""
Based on ...
"""

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

from collections import OrderedDict

import robolearn.torch.pytorch_util as ptu
from robolearn.core import logger
from robolearn.core import eval_util
from robolearn.torch.rl_algos.torch_iterative_rl_algorithm \
    import TorchIterativeRLAlgorithm
from robolearn.policies.make_deterministic import MakeDeterministic
from robolearn.utils.data_management.normalizer import RunningNormalizer

from tensorboardX import SummaryWriter


class PPO(TorchIterativeRLAlgorithm):
    """
    Proximal Policy Optimization
    """

    def __init__(
            self,
            env,
            policy,
            qf,

            normalize_obs=False,
            eval_env=None,

            reparameterize=True,

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

            optimizer_class=optim.Adam,
            # optimizer_class=optim.SGD,
            amsgrad=True,

            target_hard_update_period=1000,
            tau=1e-2,
            use_soft_update=False,

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

        TorchIterativeRLAlgorithm.__init__(
            self,
            env=env,
            exploration_policy=self._policy,
            eval_env=eval_env,
            eval_policy=eval_policy,
            obs_normalizer=self._obs_normalizer,
            **kwargs
        )

        # Important algorithm hyperparameters
        self._reparameterize = reparameterize
        assert self._reparameterize == self._policy.reparameterize
        self._entropy_scale = entropy_scale

        # Q-function
        self._qf = qf

        # ########## #
        # Optimizers #
        # ########## #
        # Q-function  Optimization Criteria
        self._qf_criterion = nn.MSELoss()

        # Q-function optimizer
        self._qf_optimizer = optimizer_class(
            self._qf.parameters(),
            lr=qf_lr,
            amsgrad=amsgrad,
            weight_decay=qf_weight_decay
        )

        # Policy optimizer
        self._policy_optimizer = optimizer_class(
            self._policy.parameters(),
            lr=policy_lr,
            amsgrad=amsgrad,
            weight_decay=policy_weight_decay,
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
                                                self.env.action_dim))
        self.logging_policy_mean = np.zeros((self.num_train_steps_per_epoch,
                                             self.env.action_dim))

        self._log_tensorboard = log_tensorboard
        self._summary_writer = SummaryWriter(log_dir=logger.get_snapshot_dir())

    def pretrain(self):
        # We do not require any pretrain (I think...)
        pass

    def _do_training(self):
        # Get batch of samples
        batch = self.get_batch()
        self.get_exploration_paths()

        # Get common data from batch
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        # ########################### #
        # LOG Useful Intentional Data #
        # ########################### #

        if self._log_tensorboard:
            pass

    def _do_not_training(self):
        return

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

        snapshot = TorchIterativeRLAlgorithm.get_epoch_snapshot(self, epoch)

        snapshot.update(
            policy=self._policy,
            qf=self._qf,
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
        statistics = OrderedDict()
        self._update_logging_data()
        statistics.update(self.eval_statistics)
        self.eval_statistics = None

    def get_batch(self):
        pass
        # batch = self.replay_buffer.random_batch(self.batch_size)
        #
        # if self._obs_normalizer is not None:
        #     batch['observations'] = \
        #         self._obs_normalizer.normalize(batch['observations'])
        #     batch['next_observations'] = \
        #         self._obs_normalizer.normalize(batch['next_observations'])
        #
        # return ptu.np_to_pytorch_batch(batch)

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

        TorchIterativeRLAlgorithm._handle_step(
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

        # self.replay_buffer.terminate_episode()

        TorchIterativeRLAlgorithm._handle_rollout_ending(self)

