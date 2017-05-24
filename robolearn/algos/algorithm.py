import copy


class Algorithm(object):
    def __init__(self, default_hyperparams, hyperparams):
        config = copy.deepcopy(default_hyperparams)
        config.update(hyperparams)
        assert isinstance(config, dict)
        self._hyperparams = config
