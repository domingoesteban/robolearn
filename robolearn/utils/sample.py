
import numpy as np

#from gps.proto.gps_pb2 import ACTION


class Sample(object):
    """
    Class that handles the representation of a trajectory and stores a
    single trajectory.
    Inspired by code in github.com:cbfinn/gps.git
    """
    def __init__(self, env, T):
        #self.agent = agent

        self.T = T
        self.dX = env.get_state_dim()  # State
        self.dU = env.get_action_dim() # Action
        self.dO = env.get_obs_dim()    # Observation
        #self.dM = env.get_env_info  # Meta data?


        self._X = np.empty((self.T, self.dX))
        self._X.fill(np.nan)
        self._obs = np.empty((self.T, self.dO))
        self._obs.fill(np.nan)
        self._act = np.empty((self.T, self.dU))
        self._act.fill(np.nan)
        #self._meta = np.empty(self.dM)
        #self._meta.fill(np.nan)

        self._info = env.get_env_info()

    def set_acts(self, act_data, t=None):
        #TODO: Check the len of act_data
        if t is None:
            self._act = act_data
        else:
            self._act[t, :] = act_data

    def set_obs(self, obs_data, obs_name=None, t=None):
        #TODO: Check the len of obs_data
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
        #TODO: Check the len of state_data
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
        """ Set trajectory data for a particular sensor. """
        if act_data is not None:
            self.set_acts(act_data, t=t)
        if obs_data is not None:
            self.set_obs(obs_data, t=t)
        if state_data is not None:
            self.set_states(state_data, t=t)


    def get_acts(self, t=None):
        """ Get the action(s). """
        return self._act if t is None else self._act[t, :]

    def get_states(self, state_name=None, t=None):
        """ Get the observation. Put it together if not precomputed. """
        state = self._X if t is None else self._X[t, :]
        if state_name is not None:
            if state_name not in self._info['state']['names']:
                raise AttributeError("There is not state with name %s in sample." % state_name)

            state_idx = self._info['state']['idx'][self._info['state']['names'].index(state_name)]
            print(state_idx)
            state = state[:, state_idx]
        return state

    def get_obs(self, obs_name=None, t=None):
        """ Get the observation. Put it together if not precomputed. """
        obs = self._obs if t is None else self._obs[t, :]
        if obs_name is not None:
            if obs_name not in self._info['obs']['names']:
                raise AttributeError("There is not observation with name %s in sample." % obs_name)

            obs_idx = self._info['obs']['idx'][self._info['obs']['names'].index(obs_name)]
            obs = obs[:, obs_idx]

        return obs

    def get_info(self):
        """ Get info data."""
        return self._info()

