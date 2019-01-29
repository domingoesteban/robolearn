import abc
from future.utils import with_metaclass


class VFunction(with_metaclass(abc.ABCMeta, object)):
    """
    Base state value function (V-function) interface.
    :math:`V(s_t)`
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
