"""
This file defines the base trajectory optimization class.
Based on Finn-GPS

"""
import abc
from future.utils import with_metaclass


class TrajOpt(object):
    """ Trajectory optimization superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        """ Update trajectory distribution. """
        raise NotImplementedError("Must be implemented in subclass.")

    def set_logger(self, logger):
        self.logger = logger
