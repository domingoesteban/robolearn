import abc


class VFunction(object, metaclass=abc.ABCMeta):
    """
    General state value function (V-function) interface.
    """

    def __init__(self, obs_dim):
        self._obs_dim = obs_dim

    @abc.abstractmethod
    def get_value(self, observation):
        pass

    def get_values(self, observations):
        pass

    @property
    def obs_dim(self):
        return self._obs_dim
