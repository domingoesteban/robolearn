"""
This file defines the base policy optimization class.
Author: C. Finn et al. Original code in: https://github.com/cbfinn/gps
"""
import abc


class PolicyOpt(object):
    """ Policy optimization superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, dO, dU):
        self._hyperparams = hyperparams
        self._dO = dO
        self._dU = dU

    @abc.abstractmethod
    def update(self, *args):
        """ Update policy. """
        raise NotImplementedError("Must be implemented in subclass.")
