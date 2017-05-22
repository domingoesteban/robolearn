""" This file defines the torque (action) cost. """
import copy

import numpy as np

from robolearn.costs.config import COST_ACTION
from robolearn.costs.cost import Cost


class CostAction(Cost):
    """ Computes torque penalties. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_ACTION)
        config.update(hyperparams)
        Cost.__init__(self, config)

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample: A single sample
        """
        sample_u = sample.get_acts()
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        # Code from superball_gps
        if self._hyperparams['target'] is None:
            target = 0
        else:
            target = np.tile(self._hyperparams['target'], (T, 1))
            #target = np.tile(self._hyperparams['target'], (Du, 1)) #TODO: Remove this

        #l = 0.5 * np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1)
        l = 0.5 * np.sum(self._hyperparams['wu'] * ((sample_u - target) ** 2), axis=1)  # Code from superball_gps
        #lu = self._hyperparams['wu'] * sample_u
        lu = self._hyperparams['wu'] * (sample_u - target)  # Code from superball_gps
        lx = np.zeros((T, Dx))
        luu = np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1])
        lxx = np.zeros((T, Dx, Dx))
        lux = np.zeros((T, Du, Dx))
        return l, lx, lu, lxx, luu, lux
