from __future__ import print_function

import numpy as np

from robolearn.envs.environment import Environment
from robolearn.envs.robot_ros_env_interface import RobotROSEnvInterface


class BigmanEnv(Environment):
    # TODO BigmanEnv is generic. Check if it is better to crete one Environment for each task
    # TODO Check if the evaluation of the commands should be done here!

    def __init__(self, interface='ros', mode='simulation', body_part_active='LA', command_type='torque',
                 observation_active=None, state_active=None):

        if interface == 'ros':
            self.interface = RobotROSEnvInterface(robot_name='bigman',
                                                  mode=mode,
                                                  body_part_active=body_part_active, cmd_type=command_type,
                                                  observation_active=observation_active,
                                                  state_active=state_active)
        else:
            raise NotImplementedError("Only ROS interface has been implemented")

    def send_action(self, action):
        self.interface.send_action(action=action)

    def read_observation(self):
        return self.interface.read_observation(option=['all'])

    def reset(self, **kwargs):
        self.interface.reset(**kwargs)

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

    def get_state_dim(self):
        return self.interface.get_state_dim()

    def get_x0(self):
        return self.interface.get_x0()

    def set_x0(self, x0):
        self.interface.set_x0(x0)

    def get_observation(self):
        return self.interface.get_observation()

    def get_state(self):
        return self.interface.get_state()

    def get_obs_info(self, **kwargs):
        return self.interface.get_obs_info(**kwargs)

    def get_state_info(self, **kwargs):
        return self.interface.get_state_info(**kwargs)

    def get_env_info(self):
        return self.interface.get_env_info()

