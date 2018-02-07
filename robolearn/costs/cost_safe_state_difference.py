""" This file defines the binary region cost. """
import copy

import numpy as np

from robolearn.costs.config import COST_SAFE_DISTANCE
from robolearn.costs.cost import Cost
from robolearn.costs.cost_utils import evall1l2term, get_ramp_multiplier


class CostSafeStateDifference(Cost):
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
            tgt = sample.get_states(config['target_state'])[:, config['idx_to_use']]
            _, dim_sensor = x.shape

            wpm = get_ramp_multiplier(
                self._hyperparams['ramp_option'], T,
                wp_final_multiplier=self._hyperparams['wp_final_multiplier']
            )
            wp = wp * np.expand_dims(wpm, axis=-1)

            # # Compute binary region penalty.
            # difference = x-tgt
            # abs_diff = np.abs(difference)
            #
            # dist = safe_distance - abs_diff
            # norm_dist = np.linalg.norm(dist, axis=1, keepdims=True)
            #
            # dist_violation = dist > 0
            # # dist_violation = (norm_dist - np.linalg.norm(safe_distance)) < 0
            # l += np.sum(wp*dist*(dist_violation*inside_cost
            #                      + ~dist_violation*outside_cost), axis=1)
            # # Cost derivative of c*max(0, d - |x|) --> c*I(d-|x|)*-1*x/||x||
            # # http://math.mit.edu/classes/18.086/2006/am57.pdf
            # jacob = wp*inside_cost*dist_violation*-1*dist/norm_dist \
            #         + wp*outside_cost*~dist_violation*-1*dist/norm_dist
            # # Tgt
            # idx = np.array(config['tgt_idx'])[config['idx_to_use']]
            # lx[:, idx] += jacob
            # # State
            # idx = np.array(config['data_idx'])[config['idx_to_use']]
            # lx[:, idx] -= jacob

            difference = x-tgt
            diff_abs = abs(difference)
            distance = diff_abs - safe_distance
            distance_norm = np.tile(np.linalg.norm(distance, axis=1, keepdims=True), [1, 2])

            measure = np.linalg.norm(difference, axis=1, keepdims=True) \
                      - np.linalg.norm(safe_distance)
            violation = np.tile(measure, [1, 2]) < 0

            l += np.sum(wp*(violation * inside_cost * distance_norm
                        + ~violation * outside_cost * distance_norm), axis=1)

            # Cost derivative of c*max(0, d - |x|) --> c*I(d-|x|)*-1*x/||x||
            # http://math.mit.edu/classes/18.086/2006/am57.pdf
            jacob = wp*(violation*inside_cost + ~violation*outside_cost) \
                    * difference/(diff_abs * distance_norm + 1e-10)

            # l += np.sum(dist * temp_cost, axis=1)

            # Tgt
            idx = np.array(config['tgt_idx'])[config['idx_to_use']]
            lx[:, idx] -= jacob
            # State
            idx = np.array(config['data_idx'])[config['idx_to_use']]
            lx[:, idx] += jacob

        return l, lx, lu, lxx, luu, lux
