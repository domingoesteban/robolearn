import gtimer as gt

from robolearn.algorithms.rl_algos.rl_algorithm import RLAlgorithm
from robolearn.utils.logging import logger
from robolearn.utils.data_management import PathBuilder
from robolearn.utils.samplers.exploration_rollout import exploration_rollout


class IterativeRLAlgorithm(RLAlgorithm):
    def __init__(self, *args, **kwargs):
        """
        Base class for Iterative(Episodic) RL Algorithms
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
