import numpy as np
from robolearn.algos.gps.costs.cost_utils import evall1l2term
from robolearn.algos.gps.costs.cost_utils import evallogl2term
from robolearn.algos.gps.costs.cost_utils import get_ramp_multiplier

from robolearn.algos.gps.costs.cost_utils import RAMP_CONSTANT


class CostInitialState(object):
    def __init__(self, state_idxs, target_states=None, wps=None,
                 ramp_option=RAMP_CONSTANT, wp_final_multiplier=1.0,
                 cost_type='logl2', l1_weight=0., l2_weight=1., alpha=1e-2,
                 ):
        self._state_idxs = state_idxs

        if target_states is None:
            target_states = [np.zeros(state_idx)
                             for state_idx in state_idxs]
        self._target_states = [np.array(tgt) for tgt in target_states]
        self._ramp_option = ramp_option
        self._wp_final_multiplier = wp_final_multiplier

        if wps is None:
            wps = [np.ones(state_idx) for state_idx in state_idxs]
        self._wps = [np.array(wp) for wp in wps]

        if cost_type == 'logl2':
            self._cost_type = evallogl2term
        elif cost_type == 'l1l2':
            self._cost_type = evall1l2term
        else:
            raise AttributeError("Wrong cost_type option")

        self._l1_weight = l1_weight
        self._l2_weight = l2_weight
        self._alpha = alpha

    def eval(self, path):
        observations = path['observations']
        T = len(path['observations'])
        Du = path['actions'][-1].shape[0]
        Dx = path['observations'][-1].shape[0]

        final_l = np.zeros(T)
        final_lu = np.zeros((T, Du))
        final_lx = np.zeros((T, Dx))
        final_luu = np.zeros((T, Du, Du))
        final_lxx = np.zeros((T, Dx, Dx))
        final_lux = np.zeros((T, Du, Dx))

        for state_idx, tgt, wp in zip(self._state_idxs, self._target_states,
                                      self._wps):
            x = observations[:, state_idx]
            dim_sensor = x.shape[-1]
            wpm = get_ramp_multiplier(
                self._ramp_option, T,
                wp_final_multiplier=self._wp_final_multiplier
            )
            wp = wp * np.expand_dims(wpm, axis=-1)

            # Compute state penalty.
            dist = (x - x[0, :]) - tgt

            jx = np.tile(np.eye(dim_sensor), [T, 1, 1])
            jxx = np.zeros((T, dim_sensor, dim_sensor, dim_sensor))

            # Evaluate penalty term.
            l, ls, lss = self._cost_type(wp, dist, jx, jxx,
                                         self._l1_weight,
                                         self._l2_weight,
                                         self._alpha,
                                         )
            final_l += l

            final_lx[:, state_idx] += ls
            temp_idx = np.ix_(state_idx, state_idx)
            final_lxx[:, temp_idx[0], temp_idx[1]] += lss

        # print('**************')
        # print('**************')
        # print('STATE_COST 0', final_lx[0])
        # print('STATE_COST -1', final_lx[-1])
        # print('**************')
        # print('**************')

        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux

