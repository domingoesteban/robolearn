import numpy as np
import gtimer as gt

from collections import OrderedDict
from robolearn.core import eval_util

from robolearn.core.rl_algorithm import RLAlgorithm
from robolearn.utils.data_management import PathBuilder
from robolearn.utils.samplers.exploration_rollout import exploration_rollout
from robolearn.core import logger


class IterativeRLAlgorithm(RLAlgorithm):
    def __init__(self, *args, **kwargs):
        """
        Base class for Iterative RL Algorithms
        """
        self.rollouts_per_epoch = kwargs.pop('rollouts_per_epoch', 1)
        RLAlgorithm.__init__(self, *args, **kwargs)

    def train(self, start_epoch=0):
        # Get snapshot of initial stuff
        if start_epoch == 0:
            self.training_mode(False)
            params = self.get_epoch_snapshot(-1)
            logger.save_itr_params(-1, params)

        self._n_env_steps_total = start_epoch * self.num_train_steps_per_epoch

        gt.reset()
        gt.set_def_unique(False)

        for epoch in gt.timed_for(
                range(start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self._start_epoch(epoch)

            while len(self._exploration_paths) < self.rollouts_per_epoch:
                self._current_path_builder = PathBuilder()
                path = exploration_rollout(self.env,
                                           self.exploration_policy,
                                           max_path_length=self.max_path_length,
                                           animated=self._render,
                                           deterministic=None)
                self._handle_path(path)
                self._n_env_steps_total += len(path['observations'])

            # Iterative learning step
            gt.stamp('sample')
            self._try_to_train()
            gt.stamp('train')

            # Evaluate if requirements are met
            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch()

    def get_exploration_paths(self):
        return self._exploration_paths

    def evaluate(self, epoch):
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

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths)

        average_returns = eval_util.get_average_returns(test_paths)
        statistics['AverageReturn'] = average_returns
        for key, value in statistics.items():
            logger.record_tabular(key, value)

        if self.render_eval_paths:
            self.env.render_paths(test_paths)

        if self._epoch_plotter is not None:
            self._epoch_plotter.draw()
            self._epoch_plotter.save_figure(epoch)
