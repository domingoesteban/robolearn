"""
Base cost class.
Based on: C. Finn
"""
import abc


class Cost(object):
    """ Cost superclass. """
    __metaclass__ = abc.ABCMeta

    # def __init__(self, hyperparams):
    #     self._hyperparams = hyperparams

    @abc.abstractmethod
    def eval(self, states, actions):
        """
        Evaluate cost function and derivatives.
        Args:
            sample:  A single sample.
        """
        raise NotImplementedError("Must be implemented in subclass.")
