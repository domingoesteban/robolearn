import abc
from future.utils import with_metaclass


class QFunction(with_metaclass(abc.ABCMeta, object)):
    """
    General state-action value function (Q-function) interface.
    """

    def __init__(self, obs_dim, action_dim):
        self._obs_dim = obs_dim
        self._action_dim = action_dim

    @abc.abstractmethod
    def get_value(self, observation, action):
        pass

    def get_values(self, observations, actions):
        pass

    @property
    def obs_dim(self):
        return self._obs_dim

    @property
    def action_dim(self):
        return self._action_dim
