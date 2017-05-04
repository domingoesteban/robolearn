from __future__ import print_function

import numpy as np

from robolearn.envs.environment import Environment
from bigman_ros_env_interface import BigmanROSEnvInterface


class BigmanEnv(Environment):
    # TODO BigmanEnv is generic. Check if it is better to crete one Environment for each task
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
        #desired_configuration = [0, 0, 0, 0, 0, 0,
        #                              0, 0, 0, 0, 0, 0,
        #                              0, 0, 0,
        #                              0, 1.57079630, 0, 0, 0, 0, 0,
        #                              0, 0,
        #                              0, -1.5707963, 0, 0, 0, 0, 0]
        desired_config = np.zeros(31)
        desired_config[8] = 1.57079630
        desired_config[19] = -1.57079630

        # Get current obs
        current_config = self.interface.last_obs[0].position
        r1 = -sum(np.power(np.abs((desired_config + np.pi - current_config) % (2*np.pi) - np.pi), 2))
        return r1

    def get_action_dim(self):
        return self.interface.get_action_dim()

    def get_obs_dim(self):
        return self.interface.get_obs_dim()
