"""
This has been adapted from Vitchyr Pong's SAC implementation.
https://github.com/vitchyr/rlkit
"""

import numpy as np
import torch
import torch.optim as optim
from itertools import chain

from collections import OrderedDict
from itertools import chain

import robolearn.torch.utils.pytorch_util as ptu
from robolearn.utils.logging import logger
from robolearn.utils import eval_util

from robolearn.algorithms.rl_algos import IncrementalRLAlgorithm
from robolearn.torch.algorithms.torch_algorithm import TorchAlgorithm

from robolearn.models.policies import MakeDeterministic
from robolearn.utils.data_management.normalizer import RunningNormalizer

import tensorboardX

MAX_LOG_ALPHA = 9.21034037  # Alpha=10000


class SAC(IncrementalRLAlgorithm, TorchAlgorithm):
    """
    Soft Actor Critic (SAC)
    """
    def __init__(
            self,
            env,
            policy,
            qf,

            replay_buffer,
            batch_size=1024,
            normalize_obs=False,
            eval_env=None,

            vf=None,
            qf2=None,
            reparameterize=True,
            action_prior='uniform',

            entropy_scale=1.,
            auto_alpha=True,
            tgt_entro=None,

            policy_lr=3e-4,
            qf_lr=3e-4,
            vf_lr=3e-4,

            policy_mean_regu_weight=1e-3,
            policy_std_regu_weight=1e-3,
            policy_pre_activation_weight=0.,

            policy_weight_decay=0.,
            q_weight_decay=0.,
            v_weight_decay=0.,

            optimizer='adam',
            # optimizer='rmsprop',
            # optimizer='sgd',
            optimizer_kwargs=None,

            soft_target_tau=5e-3,
            target_update_interval=1,

            save_replay_buffer=False,
            eval_deterministic=True,
            log_tensorboard=False,
            **kwargs
    ):
        """

        Args:
            env:

            policy:

            qf:

            qf2:

            vf:

            replay_buffer:

            batch_size:

            normalize_obs:

            eval_env:

            reparameterize:

            action_prior:

            entropy_scale:

            policy_lr:

            qf_lr:

            vf_lr:

            policy_mean_regu_weight:

            policy_std_regu_weight:

            policy_pre_activation_weight:

            policy_weight_decay:

            q_weight_decay:

            v_weight_decay:

            optimizer (string): Optimizer name

            optimizer_kwargs: Optimizer parameters

            soft_target_tau: Interpolation factor in polyak averaging for
                target networks.

            target_update_interval:

            save_replay_buffer:

            eval_deterministic:

            log_tensorboard:

            **kwargs:
        """

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

        # Q-function(s) and V-function
        self._qf = qf
        self._qf2 = qf2

        if vf is None:
            self._vf = None
            self._target_vf = None
            self._target_qf1 = qf.copy()
            self._target_qf2 = None if qf2 is None else qf2.copy()
        else:
            self._vf = vf
            self._target_vf = vf.copy()
            self._target_qf1 = None
            self._target_qf2 = None

        # Replay Buffer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.save_replay_buffer = save_replay_buffer

        # Soft-update rate for target V-function
        self._soft_target_tau = soft_target_tau
        self._target_update_interval = target_update_interval

        # Important algorithm hyperparameters
        self._reparameterize = reparameterize
        assert self._reparameterize == self._policy.reparameterize
        self._action_prior = action_prior
        self._entropy_scale = entropy_scale

        # Desired Alpha
        self._auto_alpha = auto_alpha
        if tgt_entro is None:
            tgt_entro = -env.action_dim
        self._tgt_entro = ptu.FloatTensor([tgt_entro])
        self._log_alpha = ptu.zeros(1, requires_grad=True, device=ptu.device)

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
        self._qf1_optimizer = optimizer_class(
            self._qf.parameters(),
            lr=qf_lr,
            weight_decay=q_weight_decay,
            **optimizer_kwargs
        )
        values_parameters = self._qf.parameters()
        if self._qf2 is None:
            self._qf2_optimizer = None
        else:
            self._qf2_optimizer = optimizer_class(
                self._qf2.parameters(),
                lr=qf_lr,
                weight_decay=q_weight_decay,
                **optimizer_kwargs
            )
            values_parameters = chain(values_parameters, self._qf2.parameters())

        # V-function optimizer
        if self._vf is None:
            self._vf_optimizer = None
        else:
            self._vf_optimizer = optimizer_class(
                self._vf.parameters(),
                lr=vf_lr,
                weight_decay=v_weight_decay,
                **optimizer_kwargs
            )
            values_parameters = chain(values_parameters, self._vf.parameters())
        self._values_optimizer = optimizer_class(
            values_parameters,
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

        # Alpha optimizer
        self._alpha_optimizer = optimizer_class(
            [self._log_alpha],
            lr=policy_lr,
            **optimizer_kwargs
        )

        # Weights for policy regularization coefficients
        self.pol_mean_regu_weight = policy_mean_regu_weight
        self.pol_std_regu_weight = policy_std_regu_weight
        self.pol_pre_activation_weight = policy_pre_activation_weight

        # Useful Variables for logging
        self.log_data = dict()
        self.log_data['Pol KL Loss'] = np.zeros(self.num_train_steps_per_epoch)
        self.log_data['Qf Loss'] = np.zeros(self.num_train_steps_per_epoch)
        self.log_data['Qf2 Loss'] = np.zeros(self.num_train_steps_per_epoch)
        self.log_data['Vf Loss'] = np.zeros(self.num_train_steps_per_epoch)
        self.log_data['Rewards'] = np.zeros(self.num_train_steps_per_epoch)
        self.log_data['Pol Entropy'] = np.zeros(
            self.num_train_steps_per_epoch
        )
        self.log_data['Pol Log Std'] = np.zeros((
            self.num_train_steps_per_epoch,
            self.env.action_dim,
        ))
        self.log_data['Policy Mean'] = np.zeros((
            self.num_train_steps_per_epoch,
            self.env.action_dim,
        ))
        self.log_data['Alphas'] = np.zeros(self.num_train_steps_per_epoch)

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

        # Get data from batch
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        # Get the idx for logging
        step_idx = self._n_epoch_train_steps

        # Alpha (Entropy weight in Maximum Entropy objective)
        alpha = self._entropy_scale*torch.clamp(self._log_alpha,
                                                max=MAX_LOG_ALPHA).exp()

        # ############ #
        # Critics Step #
        # ############ #
        rewards = batch['rewards']
        terminals = batch['terminals']

        if self._target_vf is None:
            # Get next actions
            next_actions, next_policy_info = self._policy(
                next_obs, return_log_prob=True
            )
            # Get next log pol
            next_log_pi = next_policy_info['log_prob']
            # Intentional Q1(s', a')
            next_q1 = self._target_qf1(next_obs, next_actions)[0]
            if self._qf2 is not None:
                # Intentional Q2(s', a')
                next_q2 = self._target_qf2(next_obs, next_actions)[0]
                # Minimum Unintentional Double-Q
                next_q = torch.min(next_q1, next_q2)
            else:
                next_q = next_q1
            # Vtarget(s')
            v_value_next = next_q - alpha*next_log_pi
        else:
            # Vtarget(s')
            v_value_next = self._target_vf(next_obs)[0]

        # Calculate Bellman Backup for Q-value
        q_backup = rewards + (1. - terminals) * self.discount * v_value_next
        q_backup = q_backup.detach()  # Detach gradient computations

        # Q1(s, a)
        q1_pred = self._qf(obs, actions)[0]

        # QF1 Loss: Mean Squared Bellman Equation (MSBE)
        qf1_loss = 0.5 * torch.mean((q_backup - q1_pred)**2)

        # # Update Q1-value function
        # self._qf1_optimizer.zero_grad()
        # qf1_loss.backward()
        # self._qf1_optimizer.step()

        if self._qf2 is not None:
            # Q2(s, a)
            q2_pred = self._qf2(obs, actions)[0]

            # QF2 Loss: Mean Squared Bellman Equation (MSBE)
            qf2_loss = 0.5 * torch.mean((q_backup - q2_pred)**2)

            # # Update Q2-value function
            # self._qf2_optimizer.zero_grad()
            # qf2_loss.backward()
            # self._qf2_optimizer.step()
        else:
            qf2_loss = 0

        # ########## #
        # Actor Step #
        # ########## #
        # Calculate Policy Loss
        new_actions, policy_info = self._policy(
            obs, return_log_prob=True
        )
        log_pi = policy_info['log_prob']
        policy_mean = policy_info['mean']
        policy_log_std = policy_info['log_std']
        pre_tanh_value = policy_info['pre_tanh_value']

        if self._action_prior == 'normal':
            raise NotImplementedError
        else:
            policy_prior_log_probs = 0.0

        # Q1(s, a)
        q1_new_actions = self._qf(obs, new_actions)[0]

        if self._qf2 is not None:
            # Q2(s, a)
            q2_new_actions = self._qf2(obs, new_actions)[0]
            # Minimum Double-Q
            q_new_actions = torch.min(q1_new_actions, q2_new_actions)
        else:
            q_new_actions = q1_new_actions

        # Policy KL loss
        if self._reparameterize:
            # policy_kl_losses = (
            #     alpha*log_pi - q_new_actions - policy_prior_log_probs
            # )
            policy_kl_losses = -(
                q_new_actions - alpha*log_pi + policy_prior_log_probs
            )
        else:
            raise NotImplementedError
            # policy_kl_losses = (
            #         alpha*log_pi * (alpha*log_pi - q_new_actions + v_pred
            #                         - policy_prior_log_probs).detach()
            # )
        policy_kl_loss = torch.mean(policy_kl_losses)

        # Policy regularization loss
        mean_reg_loss = self.pol_mean_regu_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.pol_std_regu_weight * (policy_log_std ** 2).mean()
        pre_activ_reg_loss = self.pol_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_regu_loss = mean_reg_loss + std_reg_loss + pre_activ_reg_loss

        policy_loss = policy_kl_loss + policy_regu_loss

        # Update Policy
        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # ############### #
        # V-function Step #
        # ############### #
        # V(s)
        if self._vf is None:
            v_pred = q_new_actions - alpha*log_pi
            vf_loss = 0
        else:
            v_pred = self._vf(obs)[0]
            # Calculate Bellman Backup for V-value
            v_backup = q_new_actions - alpha*log_pi + policy_prior_log_probs
            v_backup = v_backup.detach()
            # Calculate Intentional Vf Loss
            vf_loss = 0.5*torch.mean((v_backup - v_pred)**2)

            # # Update V-value function
            # self._vf_optimizer.zero_grad()
            # vf_loss.backward()
            # self._vf_optimizer.step()

        # TODO: New all values optimizer
        values_loss = qf1_loss + qf2_loss + vf_loss
        self._values_optimizer.zero_grad()
        values_loss.backward()
        self._values_optimizer.step()

        # ###################### #
        # Update Target Networks #
        # ###################### #

        if self._n_train_steps_total % self._target_update_interval == 0:
            if self._target_vf is None:
                # Update Q-value Target Network(s)
                ptu.soft_update_from_to(
                    self._qf,
                    self._target_qf1,
                    self._soft_target_tau
                )
                if self._target_qf2 is not None:
                    ptu.soft_update_from_to(
                        self._qf2,
                        self._target_qf2,
                        self._soft_target_tau
                    )
            else:
                # Update V-value Target Network
                ptu.soft_update_from_to(
                    self._vf,
                    self._target_vf,
                    self._soft_target_tau
                )

        advantages_new_actions = q_new_actions - v_pred.detach()

        # ##### #
        # Alpha #
        # ##### #
        if self._auto_alpha:
            log_alpha = self._log_alpha.clamp(max=MAX_LOG_ALPHA)
            alpha_loss = -torch.mean(log_alpha *
                                     (log_pi + self._tgt_entro).detach()
                                     )
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        # ############### #
        # LOG Useful Data #
        # ############### #
        self.log_data['Pol Entropy'][step_idx] = \
            ptu.get_numpy(-log_pi.mean(dim=0))
        self.log_data['Pol Log Std'][step_idx] = \
            ptu.get_numpy(policy_log_std.mean())
        self.log_data['Policy Mean'][step_idx] = \
            ptu.get_numpy(policy_mean.mean())
        self.log_data['Pol KL Loss'][step_idx] = \
            ptu.get_numpy(policy_kl_loss)
        self.log_data['Qf Loss'][step_idx] = ptu.get_numpy(qf1_loss)
        if self._qf2 is not None:
            self.log_data['Qf2 Loss'][step_idx] = ptu.get_numpy(qf2_loss)
        self.log_data['Vf Loss'][step_idx] = ptu.get_numpy(vf_loss)
        self.log_data['Rewards'][step_idx] = ptu.get_numpy(rewards.mean(dim=0))
        self.log_data['Alphas'][step_idx] = ptu.get_numpy(alpha)

        if self._log_tensorboard:
            self._summary_writer.add_scalar(
                'Training/qf_loss',
                ptu.get_numpy(qf1_loss),
                self._n_env_steps_total
            )
            if self._qf2 is not None:
                self._summary_writer.add_scalar(
                    'Training/qf2_loss',
                    ptu.get_numpy(qf2_loss),
                    self._n_env_steps_total
                )
            self._summary_writer.add_scalar(
                'Training/vf_loss',
                ptu.get_numpy(vf_loss),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'Training/avg_reward',
                ptu.get_numpy(rewards.mean()),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'Training/avg_advantage',
                ptu.get_numpy(advantages_new_actions.mean()),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'Training/policy_loss',
                ptu.get_numpy(policy_loss),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'Training/policy_entropy',
                ptu.get_numpy(-log_pi.mean()),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'Training/policy_mean',
                ptu.get_numpy(policy_mean.mean()),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'Training/policy_std',
                np.exp(ptu.get_numpy(policy_log_std.mean())),
                self._n_env_steps_total
            )
            self._summary_writer.add_scalar(
                'Training/q_vals',
                ptu.get_numpy(q_new_actions.mean()),
                self._n_env_steps_total
            )

            if self._n_env_steps_total % 500 == 0:
                for name, param in self._policy.named_parameters():
                    self._summary_writer.add_histogram(
                        'policy/'+name,
                        param.data.cpu().numpy(),
                        self._n_env_steps_total
                    )
                    self._summary_writer.add_histogram(
                        'policy_grad/'+name,
                        param.grad.data.cpu().numpy(),
                        self._n_env_steps_total
                    )

                for name, param in self._qf.named_parameters():
                    self._summary_writer.add_histogram(
                        'qf/'+name,
                        param.data.cpu().numpy(),
                        self._n_env_steps_total
                    )
                    self._summary_writer.add_histogram(
                        'qf_grad/'+name,
                        param.grad.data.cpu().numpy(),
                        self._n_env_steps_total
                    )
                if self._qf2 is not None:
                    for name, param in self._qf2.named_parameters():
                        self._summary_writer.add_histogram(
                            'qf2/'+name,
                            param.data.cpu().numpy(),
                            self._n_env_steps_total
                        )
                        self._summary_writer.add_histogram(
                            'qf2_grad/'+name,
                            param.grad.data.cpu().numpy(),
                            self._n_env_steps_total
                        )

                for name, param in self._vf.named_parameters():
                    self._summary_writer.add_histogram(
                        'vf/'+name,
                        param.data.cpu().numpy(),
                        self._n_env_steps_total
                    )
                    self._summary_writer.add_histogram(
                        'vf_grad/'+name,
                        param.grad.data.cpu().numpy(),
                        self._n_env_steps_total
                    )

                for name, param in self._target_vf.named_parameters():
                    self._summary_writer.add_histogram(
                        'vf_target/'+name,
                        param.cpu().data.numpy(),
                        self._n_env_steps_total
                    )

    def _do_not_training(self):
        return

    @property
    def torch_models(self):
        networks_list = [
            self._policy,
            self._qf,
        ]
        if self._qf2 is not None:
            networks_list.append(self._qf2)

        if self._vf is not None:
            networks_list.append(self._vf)
            networks_list.append(self._target_vf)
        else:
            networks_list.append(self._target_qf1)
            if self._qf2 is not None:
                networks_list.append(self._target_qf2)

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
            qf=self._qf,
            qf2=self._qf2,  # It could be None
            vf=self._vf,
            target_vf=self._target_vf,
            target_qf1=self._target_qf1,
            target_qf2=self._target_qf2,
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

        # Intentional info
        self.eval_statistics['[I] Policy Entropy'] = \
            np.nan_to_num(np.mean(
                self.log_data['Pol Entropy'][:max_step]
            ))
        self.eval_statistics['[I] Qf Loss'] = \
            np.nan_to_num(np.mean(
                self.log_data['Qf Loss'][:max_step]
            ))
        if self._qf2 is not None:
            self.eval_statistics['[I] Qf2 Loss'] = \
                np.nan_to_num(np.mean(
                    self.log_data['Qf2 Loss'][:max_step]
                ))
        self.eval_statistics['[I] Vf Loss'] = \
            np.nan_to_num(np.mean(
                self.log_data['Vf Loss'][:max_step]
            ))
        self.eval_statistics['[I] Pol KL Loss'] = \
            np.nan_to_num(np.mean(
                self.log_data['Pol KL Loss'][:max_step]
            ))
        self.eval_statistics['[I] Rewards'] = \
            np.nan_to_num(np.mean(
                self.log_data['Rewards'][:max_step]
            ))
        self.eval_statistics['[I] Policy Std'] = \
            np.nan_to_num(np.mean(
                np.exp(self.log_data['Pol Log Std'][:max_step])
            ))
        self.eval_statistics['[I] Policy Mean'] = \
            np.nan_to_num(np.mean(
                self.log_data['Policy Mean'][:max_step]
            ))
        self.eval_statistics['[I] Alphas'] = \
            np.nan_to_num(np.mean(
                self.log_data['Alphas'][:max_step]
            ))

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

        if self._exploration_paths:
            statistics.update(eval_util.get_generic_path_information(
                self._exploration_paths, stat_prefix="Exploration",
            ))
        else:
            statistics.update(eval_util.get_generic_path_information(
                test_paths, stat_prefix="Exploration",
            ))

        if self._log_tensorboard:
            self._summary_writer.add_scalar(
                'Evaluation/avg_return',
                statistics['[I] Test AverageReturn'],
                self._n_epochs
            )

            self._summary_writer.add_scalar(
                'Evaluation/avg_reward',
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

        # return ptu.np_to_pytorch_batch(batch)
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
