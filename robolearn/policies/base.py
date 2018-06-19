import abc


class Policy(object):
    """
    General policy superclass.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, action_dim):
        self._action_dim = action_dim

    @abc.abstractmethod
    def get_action(self, observation):
        """

        :param observation:
        :return: action, debug_dictionary
        """
        pass

    def reset(self):
        pass

    @property
    def action_dim(self):
        return self._action_dim


class ExplorationPolicy(Policy):
    """
    Exploration Policy
    """
    def set_num_steps_total(self, t):
        pass


class SerializablePolicy(Policy):
    """
    Policy that can be serialized.
    """
    def get_param_values(self):
        return None

    def set_param_values(self, values):
        pass

    """
    Parameters should be passed as np arrays in the two functions below.
    """
    def get_param_values_np(self):
        return None

    def set_param_values_np(self, values):
        pass
