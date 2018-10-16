import abc
from collections import OrderedDict
from typing import Iterable

import numpy as np
from torch.autograd import Variable

from robolearn.core.iterative_rl_algorithm import IterativeRLAlgorithm
from robolearn.torch.rl_algos.torch_algorithm import TorchAlgorithm
from robolearn.torch import pytorch_util as ptu
from robolearn.torch.core import PyTorchModule
from robolearn.core import logger, eval_util


class TorchIterativeRLAlgorithm(IterativeRLAlgorithm, TorchAlgorithm):
    def __init__(self, *args, **kwargs):
        super(TorchIterativeRLAlgorithm, self).__init__(*args, **kwargs)
        self.eval_statistics = None

    def get_exploration_paths(self):
        """
        Get the current exploration paths.
        Returns:

        """
        paths = IterativeRLAlgorithm.get_exploration_paths(self)

        return [dict(observations=ptu.np_to_var(path['observations']),
                     actions=ptu.np_to_var(path['actions']),
                     rewards=ptu.np_to_var(path['rewards']),
                     next_observations=ptu.np_to_var(path["next_observations"]),
                     terminals=ptu.np_to_var(path["terminals"]),
                     agent_infos=path['agent_infos']
                     )
                for path in paths]

def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return ptu.Variable(ptu.from_numpy(elem_or_tuple).float(),
                        requires_grad=False)


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
