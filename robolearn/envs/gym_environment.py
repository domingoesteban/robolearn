from __future__ import print_function

import numpy as np
import gym

from robolearn.envs.environment import Environment


class GymEnv(Environment):
    # TODO BigmanEnv is generic. Check if it is better to crete one Environment for each task
    # TODO Check if the evaluation of the commands should be done here!

    def __init__(self, name=None, render=False, observation_active=None, state_active=None, cmd_freq=100, seed=None):
        if name is None:
            name = 'CartPole-v0'
        self.name = name

        self.env = gym.make(self.name)
        self.render = render

        self.seed = seed
        if self.seed is not None:
            self.env.seed(self.seed)

        self.last_state = None
        self.last_obs = self.last_state
        self.last_reward = 0
        self.total_reward = 0
        self.total_return = 0
        self.done = False
        self.conditions = []  # Necessary for GPS
        self.seed_cond = []

        if issubclass(type(self.env.action_space), gym.spaces.discrete.Discrete):
            self.dU = self.env.action_space.n
        elif issubclass(type(self.env.action_space), gym.spaces.box.Box):
            self.dU = self.env.action_space.shape[0]
        else:
            raise NotImplementedError("Action no implemented for %s" % type(self.env.action_space))

        if issubclass(type(self.env.observation_space), gym.spaces.discrete.Discrete):
            self.dO = self.env.observation_space.n
            self.dX = self.dO
        elif issubclass(type(self.env.observation_space), gym.spaces.box.Box):
            self.dO = self.env.observation_space.shape[0]
            self.dX = self.dO
        else:
            raise NotImplementedError("Action no implemented for %s" % type(self.env.action_space))

    def send_action(self, action):
        self.last_state, self.last_reward, self.done, _ = self.env.step(action)
        self.last_obs = self.last_state

        self.total_reward += self.last_reward

        if self.render:
            self.env.render()

    def reset(self, time=None, cond=None):
        self.last_reward = 0
        self.total_return = 0
        condition = self.seed_cond[cond]
        self.env.seed(condition)
        self.last_state = self.env.reset()
        self.last_obs = self.last_state

    def get_action_dim(self):
        return self.dU

    def get_obs_dim(self):
        return self.dO

    def get_state_dim(self):
        return self.dX

    def get_x0(self):
        pass

    def set_x0(self, x0):
        pass

    def get_observation(self):
        return self.last_obs.copy()

    def get_state(self):
        return self.last_state.copy()

    def get_obs_info(self, **kwargs):
        obs_info = {'names': ['gym_observation'],
                    'dimensions': [self.get_obs_dim()],
                    'idx': [0]}
        return obs_info

    def get_state_info(self, **kwargs):
        state_info = {'names': ['gym_state'],
                    'dimensions': [self.get_state_dim()],
                    'idx': [0]}
        return state_info

    def get_env_info(self):
        env_info = {'name': self.name,
                    'obs': self.get_obs_info(),
                    'state': self.get_state_info()}
        return env_info

    def set_conditions(self, conditions):
        pass

    def add_condition(self, condition):
        self.seed_cond.append(condition)
        self.env.seed(condition)
        self.conditions.append(self.env.reset())

    def remove_condition(self, cond_idx):
        self.conditions.pop(cond_idx)

    def get_conditions(self, condition=None):
        return self.conditions[condition] if condition is not None else self.conditions

    def get_target_pose(self):
        pass

    def get_robot_pose(self):
        pass

    def stop(self):
        pass
