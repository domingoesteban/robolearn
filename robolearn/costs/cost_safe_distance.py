""" This file defines the binary region cost. """
import copy

import numpy as np

from robolearn.costs.config import COST_SAFE_DISTANCE
from robolearn.costs.cost import Cost
from robolearn.costs.cost_utils import evall1l2term, get_ramp_multiplier


class CostSafeDistance(Cost):
    """ Computes binary cost that determines if the object 
    is inside the given region around the target state.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_SAFE_DISTANCE)
        config.update(hyperparams)
        Cost.__init__(self, config)

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        l = np.zeros(T)
        lu = np.zeros((T, Du))
        lx = np.zeros((T, Dx))
        luu = np.zeros((T, Du, Du))
        lxx = np.zeros((T, Dx, Dx))
        lux = np.zeros((T, Du, Dx))

        for data_type in self._hyperparams['data_types']:
            config = self._hyperparams['data_types'][data_type]
            wp = config['wp']
            safe_distance = config['safe_distance']
            outside_cost = config['outside_cost']
            inside_cost = config['inside_cost']
            x = sample.get_states(data_type)[:, config['idx_to_use']]
            _, dim_sensor = x.shape

            wpm = get_ramp_multiplier(
                self._hyperparams['ramp_option'], T,
                wp_final_multiplier=self._hyperparams['wp_final_multiplier']
            )
            wp = wp * np.expand_dims(wpm, axis=-1)

            # Compute binary region penalty.
            dist = safe_distance - np.abs(x)

            dist_violation = dist > 0

            l += np.sum(dist*(dist_violation*inside_cost
                        + ~dist_violation*outside_cost), axis=1)

            # l += np.sum(dist * temp_cost, axis=1)

            # Cost derivative of c*max(0, d - |x|) --> c*I(d-|x|)*-1*x/|x|
            idx = np.array(config['data_idx'])
            lx[:, idx] += wp*inside_cost*dist_violation*-1*x/np.abs(x) \
                          + wp*outside_cost*~dist_violation*-1*x/np.abs(x)

        return l, lx, lu, lxx, luu, lux
