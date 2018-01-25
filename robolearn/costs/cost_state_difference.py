""" This file defines the state target cost. """
import copy

import numpy as np

from robolearn.costs.config import COST_STATE
from robolearn.costs.cost import Cost
from robolearn.costs.cost_utils import evall1l2term, get_ramp_multiplier


class CostStateDifference(Cost):
    """ Computes l1/l2 distance to a fixed target state. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_STATE)
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

        final_l = np.zeros(T)
        final_lu = np.zeros((T, Du))
        final_lx = np.zeros((T, Dx))
        final_luu = np.zeros((T, Du, Du))
        final_lxx = np.zeros((T, Dx, Dx))
        final_lux = np.zeros((T, Du, Dx))

        for data_type in self._hyperparams['data_types']:
            config = self._hyperparams['data_types'][data_type]
            wp = config['wp']
            tgt = sample.get_states(config['target_state'])
            x = sample.get_states(data_type)

            if config['average'] is not None:  # From superball_gps
                raise NotImplementedError('Not implemented for average')
                x = np.mean(x.reshape((T,) + config['average']), axis=1)
                _, dim_sensor = x.shape
                num_sensor = config['average'][0]
                l = x.dot(np.array(wp).T)
                ls = np.tile(np.array(wp), [T, num_sensor]) / num_sensor
                lss = np.zeros((T, dim_sensor * num_sensor, dim_sensor * num_sensor))
            else:
                _, dim_sensor = x.shape

                wpm = get_ramp_multiplier(
                    self._hyperparams['ramp_option'], T,
                    wp_final_multiplier=self._hyperparams['wp_final_multiplier']
                )
                wp = wp * np.expand_dims(wpm, axis=-1)
                # Compute state penalty.
                dist = (tgt - x)

                Jd = np.c_[np.eye(dim_sensor), -np.eye(dim_sensor)]

                # Evaluate penalty term.
                l, ls, lss = evall1l2term(
                    wp, dist, np.tile(Jd, [T, 1, 1]),
                    np.zeros((T, dim_sensor, 2*dim_sensor, 2*dim_sensor)),
                    self._hyperparams['l1'], self._hyperparams['l2'],
                    self._hyperparams['alpha']
                )

            final_l += l

            # Tgt idx
            final_lx[:, config['tgt_idx']] = ls[:, :dim_sensor]
            temp_idx = np.ix_(config['tgt_idx'], config['tgt_idx'])
            final_lxx[:, temp_idx[0], temp_idx[1]] = lss[:, :dim_sensor, :dim_sensor]

            # Data idx
            final_lx[:, config['data_idx']] = ls[:, dim_sensor:]
            temp_idx = np.ix_(config['data_idx'], config['data_idx'])
            final_lxx[:, temp_idx[0], temp_idx[1]] = lss[:, dim_sensor:, dim_sensor:]

        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux
