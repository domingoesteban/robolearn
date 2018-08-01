import abc
from collections import OrderedDict
from typing import Iterable

import numpy as np

from robolearn.core.incremental_rl_algorithm import IncrementalRLAlgorithm
from robolearn.torch import pytorch_util as ptu
from robolearn.torch.core import PyTorchModule
from robolearn.core import logger, eval_util


class TorchIncrementalRLAlgorithm(IncrementalRLAlgorithm):
    def __init__(self, *args, **kwargs):
        render_eval_paths = kwargs.pop('render_eval_paths', False)
        plotter = kwargs.pop('plotter', None)
        IncrementalRLAlgorithm.__init__(self, *args, **kwargs)
        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self._epoch_plotter = plotter

    def get_paths(self):
        paths = self._exploration_paths
        # batch = self.replay_buffer.random_batch(self.batch_size)
        return [dict(observations=ptu.np_to_var(path['observations']),
                     actions=ptu.np_to_var(path['actions']),
                     rewards=ptu.np_to_var(path['rewards']),
                     next_observations=ptu.np_to_var(path["next_observations"]),
                     terminals=ptu.np_to_var(path["terminals"]),
                     agent_infos=path['agent_infos']
                     )
                for path in paths]

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[PyTorchModule]:
        pass

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def cuda(self):
        for net in self.networks:
            net.cuda()

    def cpu(self):
        for net in self.networks:
            net.cpu()

    def evaluate(self, epoch):
        statistics = OrderedDict()
        statistics.update(self.eval_statistics)
        self.eval_statistics = None

        logger.log("Collecting samples for evaluation")
        test_paths = self.eval_sampler.obtain_samples()

        statistics.update(eval_util.get_generic_path_information(
            test_paths, stat_prefix="Test",
        ))
        statistics.update(eval_util.get_generic_path_information(
            self._exploration_paths, stat_prefix="Exploration",
        ))
        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths)

        average_returns = eval_util.get_average_returns(test_paths)
        statistics['AverageReturn'] = average_returns
        for key, value in statistics.items():
            logger.record_tabular(key, value)

        if self.render_eval_paths:
            self.env.render_paths(test_paths)

        if self._epoch_plotter:
            self._epoch_plotter.draw()
