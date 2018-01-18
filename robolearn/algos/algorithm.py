import copy


class Algorithm(object):
    def __init__(self, default_hyperparams, hyperparams):
        """
        Algorithm base class constructor. It will set _hyperparams attribute
        with default values and replace only the specific object hyperparameters
        :param default_hyperparams: Default algorithm hyperparameters.
        :param hyperparams: Particular object hyperparameters.
        """
        config = copy.deepcopy(default_hyperparams)
        config.update(hyperparams)
        assert isinstance(config, dict)
        self._hyperparams = config
