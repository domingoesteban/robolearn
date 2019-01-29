import abc
from future.utils import with_metaclass
import pickle
import time

import gtimer as gt
import numpy as np

from robolearn.utils.logging import logger
from robolearn.utils.data_management.path_builder import PathBuilder
from robolearn.utils.samplers.in_place_path_sampler import InPlacePathSampler
from robolearn.utils.samplers.finite_path_sampler import FinitePathSampler

from collections import OrderedDict
from robolearn.utils import eval_util


class RLAlgorithm(with_metaclass(abc.ABCMeta, object)):
    """
    Base Reinforcement Learning algorithm interface.
    """

    def __init__(
            self,
            explo_env,
            explo_policy,
            eval_env=None,
            eval_policy=None,
            finite_horizon_eval=False,
            obs_normalizer=None,

            algo_mode='online',
            num_epochs=100,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            max_path_length=1000,

            min_steps_start_train=0,
            min_start_eval=0,

            num_updates_per_train_call=1,

            discount=0.99,

            render=False,

            save_algorithm=False,
            save_environment=True,

            epoch_plotter=None,
    ):
        # type: # (gym.Env, ExplorationPolicy) -> None
        """
        Base class for RL Algorithms
        :param explo_env: Environment used for training.
        :param explo_policy: Policy used to explore during training
        (Behavior policy).
        :param eval_env: Environment used for evaluation. By default, a
        copy of `env` will be made.
        :param eval_policy: Policy to evaluate with.
        :param num_epochs: Number of training episodes.
        :param num_steps_per_epoch: Number of timesteps per epoch.
        :param num_steps_per_eval: Number of timesteps per evaluation
        :param num_updates_per_train_call: Used by online training mode.
        :param num_updates_per_epoch: Used by batch training mode.
        :param max_path_length: Max length of sampled path (rollout) from env.
        :param min_steps_start_train: Min steps to start training.
        :param min_start_eval: Min steps to start evaluating.
        :param discount: discount factor (gamma).
        :param render: Visualize or not the environment.
        :param save_algorithm: Save or not the algorithm  after iterations.
        :param save_environment: Save or not the environment after iterations
        """
        # Training environment, policy and state-action spaces
        self.explo_env = explo_env
        self.explo_policy = explo_policy
        self.action_space = explo_env.action_space
        self.obs_space = explo_env.observation_space
        self._obs_normalizer = obs_normalizer

        # Evaluation environment, policy and sampler
        self.eval_env = eval_env or pickle.loads(pickle.dumps(explo_env))
        if eval_policy is None:
            eval_policy = explo_policy
        self.eval_policy = eval_policy

        if finite_horizon_eval:
            self.eval_sampler = FinitePathSampler(
                env=eval_env,
                policy=eval_policy,
                total_paths=int(num_steps_per_eval/max_path_length),
                max_path_length=max_path_length,
                obs_normalizer=self._obs_normalizer,
            )
        else:
            self.eval_sampler = InPlacePathSampler(
                env=eval_env,
                policy=eval_policy,
                total_samples=num_steps_per_eval,
                max_path_length=max_path_length,
                obs_normalizer=self._obs_normalizer,
            )

        # RL algorithm hyperparameters
        self.num_epochs = num_epochs
        self.num_train_steps_per_epoch = num_steps_per_epoch
        self.max_path_length = max_path_length
        self.num_updates_per_train_call = num_updates_per_train_call
        self.num_steps_per_eval = num_steps_per_eval

        self._min_steps_start_train = min_steps_start_train
        self._min_steps_start_eval = min_start_eval

        # Reward related
        self.discount = discount

        # Flag to render while sampling
        self._render = render

        # Save flags
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment

        # Internal variables
        self._n_epochs = 0
        self._n_env_steps_total = 0  # Accumulated interactions with the env
        self._n_total_train_steps = 0  # Accumulated total training steps
        self._n_epoch_train_steps = 0  # Accumulated epoch's training steps
        self._n_rollouts_total = 0  # Accumulated rollouts
        self._epoch_start_time = None  # Wall time
        self._current_path = PathBuilder()  # Current path
        self._exploration_paths = []  # All paths in current epoch
        self.post_epoch_funcs = []  # Fcns to call at the end of episode

        if algo_mode not in ['online', 'episode']:
            raise TypeError("Invalid algo_mode: %s" % self.algo_mode)
        self._algo_mode = algo_mode

        # Logger variables
        self.eval_statistics = None  # Dict of stats from evaluation
        self._old_table_keys = None  # Previous table keys of the logger
        self._epoch_plotter = epoch_plotter

    """
    ############################
    ############################
    Methods related to Training.
    ############################
    ############################
    """
    def pretrain(self, *args, **kwargs):
        """
        Do anything before the training phase.
        """
        pass

    def train(self, start_epoch=0):
        # Get snapshot of initial algo state
        if start_epoch == 0:
            self._log_initial_data()

        self.training_mode(False)
        self._n_env_steps_total = start_epoch * self.num_train_steps_per_epoch

        gt.reset()
        gt.set_def_unique(False)

        self._current_path = PathBuilder()
        for epoch in gt.timed_for(
                range(start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self._start_epoch(epoch)
            obs = self._start_new_rollout()
            for ss in range(self.num_train_steps_per_epoch):
                obs = self._take_step_in_env(obs)
                gt.stamp('sample')

                if self._algo_mode == 'online':
                    self._try_to_train()
                    gt.stamp('train')

            if self._algo_mode == 'episode':
                self._try_to_train()
                gt.stamp('train')

            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch(epoch)

    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    def _take_step_in_env(self, observation):
        # Get policy action
        if self._obs_normalizer is None:
            policy_input = observation
        else:
            policy_input = self._obs_normalizer.normalize(observation)
        action, pol_info = self.explo_policy.get_action(
            policy_input,
        )

        if self._render:
            self.explo_env.render()

        # Interact with environment
        next_obs, reward, terminal, env_info = (
            self.explo_env.step(action)
        )

        # Increase counter
        self._n_env_steps_total += 1

        terminal = np.array([terminal])
        reward = np.array([reward])

        self._handle_step(
            observation,
            action,
            reward,
            next_obs,
            terminal,
            agent_info=pol_info,
            env_info=env_info,
        )

        # Check it we need to start a new rollout
        if terminal or (len(self._current_path) >=
                        self.max_path_length):
            self._end_rollout()
            next_obs = self._start_new_rollout()

        return next_obs

    def _try_to_train(self):
        """
        Check if the requirements are fulfilled to start or not training.
        Returns:

        """
        if self._can_train():
            self.training_mode(True)
            for i in range(self.num_updates_per_train_call):
                # Call algorithm-specific training method
                self._do_training()
                self._n_epoch_train_steps += 1
                self._n_total_train_steps += 1
            self.training_mode(False)
        else:
            self._not_do_training()

    def _can_train(self):
        """
        Training requirements are fulfilled or not.

        Train only if you have more data than the batch size in the
        Replay Buffer.

        :return (bool):
        """
        # Bigger than, because n_env_steps_total updated after sampling
        return self._n_env_steps_total > self._min_steps_start_train

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some computations, e.g. perform one gradient step.
        :return:
        """
        pass

    @abc.abstractmethod
    def _not_do_training(self):
        """
        Perform something when it is not possible to do training.
        :return:
        """
        pass

    def _start_epoch(self, epoch):
        """
        Computations at the beginning of an epoch.
        Args:
            epoch:

        Returns:

        """
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._n_epochs = epoch + 1
        self._n_epoch_train_steps = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self, epoch):
        """
        Computations at the end of an epoch.
        Returns:

        """
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _start_new_rollout(self):
        """
        Computations at the beginning of every rollout.
        Returns:

        """
        self.explo_policy.reset()
        return self.explo_env.reset()

    def _end_rollout(self):
        """
        Computations at the end of every rollout.
        """
        self._n_rollouts_total += 1
        if len(self._current_path) > 0:
            self._exploration_paths.append(
                self._current_path.get_all_stacked()
            )
            self._current_path = PathBuilder()

    def _handle_path(self, path):
        """
        Computations for a path.
        :param path:
        :return:
        """
        for (
                ob,
                action,
                reward,
                next_ob,
                terminal,
                agent_info,
                env_info
        ) in zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        ):
            self._handle_step(
                ob,
                action,
                reward,
                next_ob,
                terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self._end_rollout()

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
        Computations at every step.
        :return:
        """
        # Add data to current path builder
        self._current_path.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )

    """
    ##############################
    ##############################
    Methods related to Evaluation.
    ##############################
    ##############################
    """
    def _try_to_eval(self, epoch, eval_paths=None):
        """
        Check if the requirements are fulfilled to start or not an evaluation.
        Args:
            epoch (int): Epoch

        Returns:

        """
        logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            # Call algorithm-specific evaluate method
            self.evaluate(epoch)

            self._log_data(epoch)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        Evaluation requirements are fulfilled or not.

        Evaluate only if you have non-zero exploration paths AND you have
        more steps than _min_steps_start_eval. This value can be the minimum
        buffer size in the Replay Buffer.

        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        :return:
        """
        return (
                len(self._exploration_paths) > 0
                and self._n_epoch_train_steps >= self._min_steps_start_eval
        )

    def evaluate(self, epoch):
        """
        Evaluate the policy, e.g. save/print progress.
        :param epoch:
        :return:
        """
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        statistics = OrderedDict()
        statistics.update(self.eval_statistics)
        self.eval_statistics = None

        logger.log("Collecting samples for evaluation")
        test_paths = self.eval_sampler.obtain_samples()

        statistics.update(eval_util.get_generic_path_information(
            test_paths, stat_prefix="Test",
        ))

        if self._exploration_paths:
            statistics.update(eval_util.get_generic_path_information(
                self._exploration_paths, stat_prefix="Exploration",
            ))
        else:
            statistics.update(eval_util.get_generic_path_information(
                test_paths, stat_prefix="Exploration",
            ))

        if hasattr(self.explo_env, "log_diagnostics"):
            self.explo_env.log_diagnostics(test_paths)

        for key, value in statistics.items():
            logger.record_tabular(key, value)

        if self._epoch_plotter is not None:
            self._epoch_plotter.draw()
            self._epoch_plotter.save_figure(epoch)

    def get_epoch_snapshot(self, epoch):
        """
        Data to save in file.
        :param epoch:
        :return:
        """
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.explo_policy,
        )

        if self.save_environment:
            data_to_save['env'] = self.explo_env

        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.explo_env
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def _log_initial_data(self):
        self.training_mode(False)

        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)

        self.evaluate(-1)
        logger.record_tabular("Number of train steps total", 0)
        logger.record_tabular("Number of env steps total", 0)
        logger.record_tabular("Number of rollouts total", 0)
        logger.record_tabular('Train Time (s)', 0)
        logger.record_tabular('(Previous) Eval Time (s)', 0)
        logger.record_tabular('Sample Time (s)', 0)
        logger.record_tabular('Epoch Time (s)', 0)
        logger.record_tabular('Total Train Time (s)', 0)
        logger.record_tabular("Epoch", 0)

        logger.dump_tabular(with_prefix=False, with_timestamp=False,
                            write_header=True)

    def _log_data(self, epoch, write_header=False):
        # Update logger parameters with algorithm-specific variables
        params = self.get_epoch_snapshot(epoch)
        logger.save_itr_params(epoch, params)

        # Check that logger parameters (table keys) did not change.
        table_keys = logger.get_table_key_set()
        if self._old_table_keys is not None:
            if not table_keys == self._old_table_keys:
                error_text = "Table keys cannot change from iteration " \
                             "to iteration.\n"
                error_text += 'table_keys: '
                error_text += str(table_keys)
                error_text += '\n'
                error_text += 'old_table_keys: '
                error_text += str(self._old_table_keys)
                error_text += 'not in new: '
                error_text += str(np.setdiff1d(list(table_keys),
                                               list(self._old_table_keys))
                                  )
                error_text += 'not in old:'
                error_text += str(np.setdiff1d(list(self._old_table_keys),
                                               list(table_keys))
                                  )
                raise AttributeError(error_text)
        self._old_table_keys = table_keys

        # Add the number of steps to the logger
        logger.record_tabular(
            "Number of train steps total",
            self._n_total_train_steps,
        )
        logger.record_tabular(
            "Number of env steps total",
            self._n_env_steps_total,
        )
        logger.record_tabular(
            "Number of rollouts total",
            self._n_rollouts_total,
        )

        # Get useful times
        times_itrs = gt.get_times().stamps.itrs
        train_time = times_itrs['train'][-1]
        sample_time = times_itrs['sample'][-1]
        # eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
        eval_time = times_itrs['eval'][-1] if 'eval' in times_itrs else 0
        epoch_time = train_time + sample_time + eval_time
        total_time = gt.get_times().total

        # Add the previous times to the logger
        logger.record_tabular('Train Time (s)', train_time)
        logger.record_tabular('(Previous) Eval Time (s)', eval_time)
        logger.record_tabular('Sample Time (s)', sample_time)
        logger.record_tabular('Epoch Time (s)', epoch_time)
        logger.record_tabular('Total Train Time (s)', total_time)

        # Add the number of epoch to the logger
        logger.record_tabular("Epoch", epoch)

        # Dump the logger data
        logger.dump_tabular(with_prefix=False, with_timestamp=False,
                            write_header=write_header)

