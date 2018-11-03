"""
Base cost class.
Based on: C. Finn
"""
import abc
from future.utils import with_metaclass


class Reward(with_metaclass(abc.ABCMeta, object)):
    # def __init__(self, hyperparams):
    #     self._hyperparams = hyperparams

    @abc.abstractmethod
    def eval(self, states, actions, gradients=False):
        """
        Evaluate cost function and derivatives.
        Args:
            states:
            actions:
            gradients(Bool):

        Returns:

        """
        raise NotImplementedError("Must be implemented in subclass.")

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)
