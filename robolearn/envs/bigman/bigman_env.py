from __future__ import print_function

import numpy as np

from robolearn.envs.base import Env
from bigman_ros_env_interface import BigmanROSEnvInterface


class BigmanEnv(Env):
    # TODO BigmanEnv is generic. Check if it is better to crete one Env for each task
    # TODO Check if the evaluation of the commands should be done here!

    def __init__(self, interface='ros', mode='simulation', joints_active='left_arm', command_type='torque'):

        if interface == 'ros':
            self.interface = BigmanROSEnvInterface(mode=mode, joints_active=joints_active, command_type=command_type)
        else:
            raise NotImplementedError("Only ROS interface has been implemented")

    def send_action(self, action):
        self.interface.send_action(action=action)

    def read_observation(self):
        return self.interface.read_observation(option=['all'])

    def reset(self):
        self.interface.reset()

    def get_reward(self):
        return np.random.uniform(0, 2)

    def get_action_dim(self):
        return self.interface.get_action_dim()

    def get_obs_dim(self):
        return self.interface.get_obs_dim()
