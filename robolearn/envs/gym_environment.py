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

        self.env = gym.make(name)
        self.render = render

        if seed is not None:
            self.env.seed(seed)

        self.last_state = None
        self.last_obs = self.last_state
        self.last_reward = 0
        self.total_reward = 0
        self.done = False

    def send_action(self, action):
        self.last_state, self.last_reward, self.done, _ = self.env.step(action)

        self.total_reward += self.last_reward

        if self.render:
            self.env.render()

    def reset(self, **kwargs):
        self.last_reward = 0
        self.total_return = 0
        self.last_state = self.env.reset()

    def get_action_dim(self):
        pass

    def get_obs_dim(self):
        pass

    def get_state_dim(self):
        pass

    def get_x0(self):
        pass

    def set_x0(self, x0):
        pass

    def get_observation(self):
        return self.last_obs.copy()

    def get_state(self):
        return self.last_state.copy()

    def get_obs_info(self, **kwargs):
        pass

    def get_state_info(self, **kwargs):
        pass

    def get_env_info(self):
        pass

    def set_conditions(self, conditions):
        pass

    def add_condition(self, condition):
        pass

    def remove_condition(self, cond_idx):
        pass

    def get_conditions(self, cond=None):
        pass

    def add_q0(self, q0):
        pass

    def get_q0_idx(self, q0):
        pass

    def stop(self):
        pass

    def get_target_pose(self):
        pass

    def get_robot_pose(self):
        pass

