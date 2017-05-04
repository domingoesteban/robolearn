import numpy as np
import copy

class Algorithm(object):
    def __init__(self, default_hyperparams, hyperparams):
        config = copy.deepcopy(default_hyperparams)
        config.update(hyperparams)
        self._hyperparams = config


class RLAlgorithm(Algorithm):
    def __init__(self, default_hyperparams, hyperparams):
        super(RLAlgorithm, self).__init__(default_hyperparams, hyperparams)

    def explore(self):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError

    def run(self, **kwargs):
        raise NotImplementedError


class ILAlgorithm(Algorithm):
    def __init__(self, default_hyperparams, hyperparams):
        super(ILAlgorithm, self).__init__(default_hyperparams, hyperparams)

    def demonstrate(self):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError
