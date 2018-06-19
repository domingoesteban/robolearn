import abc
from collections import OrderedDict
from typing import Iterable

import numpy as np
from torch.autograd import Variable

from robolearn.core.incremental_rl_algorithm import IncrementalRLAlgorithm
from robolearn.torch import pytorch_util as ptu
from robolearn.torch.core import PyTorchModule
from robolearn.core import logger, eval_util


class TorchIncrementalRLAlgorithm(IncrementalRLAlgorithm):
    def __init__(self, *args, render_eval_paths=False, plotter=None, **kwargs):
        super(TorchIncrementalRLAlgorithm, self).__init__(*args, **kwargs)
        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self._epoch_plotter = plotter

    def get_batch(self):
        batch = self.replay_buffer.random_batch(self.batch_size)
        return np_to_pytorch_batch(batch)

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

    # def _get_action_and_info(self, observation):
    #     """
    #     Get an action to take in the environment.
    #     :param observation:
    #     :return:
    #     """
    #     if self._algo_interface == 'np':
    #         return super(IncrementalRLAlgorithm,
    #                      self)._get_action_and_info(observation,)
    #     else:
    #         action = self.exploration_policy(ptu.np_to_var(observation))
    #
    #         if isinstance(action, tuple):
    #             action = action[0]
    #
    #         if self._algo_off_policy:
    #             return ptu.get_numpy(action), dict()
    #         else:
    #             return ptu.get_numpy(action), \
    #                    dict(zip(self.exploration_policy.get_output_labels(),
    #                             action))

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


def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }