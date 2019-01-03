import numpy as np
from robolearn.algorithms.rl_algos import get_ramp_multiplier
from robolearn.algorithms.rl_algos import RAMP_CONSTANT


class CostSafeDistance(object):
    def __init__(self, state_idxs, safe_distances, wps=None,
                 inside_costs=None, outside_costs=None,
                 ramp_option=RAMP_CONSTANT, wp_final_multiplier=1.0,
                 ):

        self._state_idxs = state_idxs

        self._safe_distances = [np.array(dist) for dist in safe_distances]

        if wps is None:
            wps = [np.ones(state_idx) for state_idx in state_idxs]
        self._wps = [np.array(wp) for wp in wps]

        if inside_costs is None:
            inside_costs = [np.ones(state_idx) for state_idx in state_idxs]
        self._inside_costs = inside_costs

        if outside_costs is None:
            outside_costs = [np.zeros(state_idx) for state_idx in state_idxs]
        self._outside_costs = outside_costs

        self._ramp_option = ramp_option
        self._wp_final_multiplier = wp_final_multiplier

    def eval(self, path):
        observations = path['observations']
        T = len(path['observations'])
        Du = path['actions'][-1].shape[0]
        Dx = path['observations'][-1].shape[0]

        l = np.zeros(T)
        lu = np.zeros((T, Du))
        lx = np.zeros((T, Dx))
        luu = np.zeros((T, Du, Du))
        lxx = np.zeros((T, Dx, Dx))
        lux = np.zeros((T, Du, Dx))

        for state_idx, safe_dist, wp, inside_cost, outside_cost in zip(
                self._state_idxs, self._safe_distances, self._wps,
                self._inside_costs, self._outside_costs
        ):
            x = observations[:, state_idx]
            dim_sensor = x.shape[-1]
            wpm = get_ramp_multiplier(
                self._ramp_option, T,
                wp_final_multiplier=self._wp_final_multiplier
            )
            wp = wp * np.expand_dims(wpm, axis=-1)

            # Compute binary region penalty.
            dist = safe_dist - np.abs(x)

            dist_violation = dist > 0
            is_penetrating = np.all(dist_violation, axis=1, keepdims=True)

            l += np.sum(dist * (dist_violation * is_penetrating * inside_cost
                                + ~dist_violation * ~is_penetrating * outside_cost),
                        axis=1)

            # Cost derivative of c*max(0, d - |x|) --> c*I(d-|x|)*-1*x/|x|
            idx = np.array(state_idx)
            lx[:, idx] += wp*(
                    inside_cost * dist_violation * is_penetrating - 1 * x / np.abs(x) +
                    outside_cost * ~dist_violation * ~is_penetrating - 1 * x / np.abs(x)
            )

        return l, lx, lu, lxx, luu, lux
