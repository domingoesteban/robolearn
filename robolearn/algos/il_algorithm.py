from robolearn.algos.algorithm import Algorithm


class ILAlgorithm(Algorithm):
    def __init__(self, default_hyperparams, hyperparams):
        """
        Class constructor.
        :param default_hyperparams: Default algorithm hyperparameters.
        :param hyperparams: Particular object hyperparameters.
        """
        super(ILAlgorithm, self).__init__(default_hyperparams, hyperparams)

    def demonstrate(self):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError
