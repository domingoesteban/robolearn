import numpy as np
import matplotlib.pyplot as plt

from robolearn.costs.cost_action import CostAction
from robolearn.costs.cost_state import CostState
from robolearn.costs.cost_sum import CostSum
from robolearn.costs.cost_utils import RAMP_QUADRATIC

class Sample(object):
    def __init__(self, T, dX, dU, dO):
        #self.agent = agent

        self.T = T
        self.dX = dX  # State
        self.dU = dU # Action
        self.dO = dO    # Observation
        self._X = np.empty((self.T, self.dX))
        self._X.fill(np.nan)
        self._obs = np.empty((self.T, self.dO))
        self._obs.fill(np.nan)
        self._act = np.empty((self.T, self.dU))
        self._act.fill(np.nan)

    def set_acts(self, act_data, t=None):
        #TODO: Check the len of act_data
        if t is None:
            self._act = act_data
        else:
            self._act[t, :] = act_data

    def set_obs(self, obs_data, obs_name=None, t=None):
        if obs_name is None:
            obs_idx = range(self.dO)
        else:
            if obs_name not in self._info['obs']['names']:
                raise AttributeError("There is not any observation with name %s in sample." % obs_name)

            obs_idx = self._info['obs']['idx'][self._info['obs']['names'].index(obs_name)]

        if t is None:
            self._obs[:, obs_idx] = obs_data
        else:
            self._obs[t, obs_idx] = obs_data

    def set_states(self, state_data, state_name=None, t=None):
        if state_name is None:
            state_idx = range(self.dX)
        else:
            if state_name not in self._info['state']['names']:
                raise AttributeError("There is not any state with name %s in sample." % state_name)

            state_idx = self._info['state']['idx'][self._info['state']['names'].index(state_name)]

        if t is None:
            self._X[:, state_idx] = state_data
        else:
            self._X[t, state_idx] = state_data

    def set(self, act_data=None, obs_data=None, state_data=None, t=None):
        if act_data is not None:
            self.set_acts(act_data, t=t)
        if obs_data is not None:
            self.set_obs(obs_data, t=t)
        if state_data is not None:
            self.set_states(state_data, t=t)


    def get_acts(self, t=None):
        return self._act if t is None else self._act[t, :]

    def get_states(self, state_name=None, t=None):
        """ Get the observation. Put it together if not precomputed. """
        state = self._X if t is None else self._X[t, :]
        if state_name is not None:
            if state_name not in self._info['state']['names']:
                raise AttributeError("There is not state with name %s in sample." % state_name)

            state_idx = self._info['state']['idx'][self._info['state']['names'].index(state_name)]
            state = state[:, state_idx]
        return state

    def get_obs(self, obs_name=None, t=None):
        obs = self._obs if t is None else self._obs[t, :]
        if obs_name is not None:
            if obs_name not in self._info['obs']['names']:
                raise AttributeError("There is not observation with name %s in sample." % obs_name)

            obs_idx = self._info['obs']['idx'][self._info['obs']['names'].index(obs_name)]
            obs = obs[:, obs_idx]

        return obs


# ########## #
# ########## #

T = 100
dX = 2
dU = 1
dO = 1


# ###### #
# Sample #
# ###### #
sample = Sample(T, dX, dU, dO)

all_actions = np.ones((T, dU))
all_obs = np.ones((T, dO))
all_states = np.ones((T, dX))

sample.set_acts(all_actions)  # Set all actions at the same time
sample.set_obs(all_obs)  # Set all obs at the same time
sample.set_states(all_states)  # Set all states at the same time



# #### #
# Cost #
# #### #
cost_act = {
    'type': CostAction,
    'wu': np.ones(dU) * 1e-4,
    'target': None,   # Target action value
}

#cost_state = {
#    'type': CostState,
#    'ramp_option': RAMP_QUADRATIC,  # How target cost ramps over time. RAMP_* :CONSTANT,LINEAR, QUADRATIC, FINAL_ONLY
#    'l1': 0.0,
#    'l2': 1.0,
#    'wp_final_multiplier': 5.0,  # Weight multiplier on final time step.
#    'data_types': {
#        'optitrack': {
#            'wp': np.ones_like(target_state),  # State weights - must be set.
#            'target_state': target_state,  # Target state - must be set.
#            'average': None,  #(12, 3),
#            'data_idx': bigman_env.get_state_info(name='optitrack')['idx']
#        }
#    },
#}
#
#cost_sum = {
#    'type': CostSum,
#    'costs': [cost_act, cost_state],
#    'weights': [0.1, 5.0],
#}

cost = cost_act['type'](cost_act)
#cost = cost_sum['type'](cost_sum)

print("Evaluating sample's cost...")
l, lx, lu, lxx, luu, lux = cost.eval(sample)

print("l %s" % str(l.shape))
print("lx %s" % str(lx.shape))
print("lu %s" % str(lu.shape))
print("lxx %s" % str(lxx.shape))
print("luu %s" % str(luu.shape))
print("lux %s" % str(lux.shape))
#plt.plot(l)
#plt.plot(lx)
#plt.plot(lu)
#plt.plot(lxx)
#plt.plot(lux)
#plt.show()

