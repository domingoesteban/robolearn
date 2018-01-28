""" This file defines the base trajectory optimization class. """
import abc


class TrajOpt(object):
    """ Trajectory optimization superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        """ Update trajectory distributions. """
        raise NotImplementedError("Must be implemented in subclass.")

    def set_logger(self, logger):
        self.logger = logger



# TODO - Interface with C++ traj opt?
