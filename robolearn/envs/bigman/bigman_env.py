from __future__ import print_function

import numpy as np

from robolearn.envs.environment import Environment
from robolearn.envs.robot_ros_env_interface import RobotROSEnvInterface


class BigmanEnv(Environment):
    # TODO BigmanEnv is generic. Check if it is better to crete one Environment for each task
    # TODO Check if the evaluation of the commands should be done here!

    def __init__(self, interface='ros', mode='simulation', body_part_active='LA', command_type='torque',
                 observation_active=None, state_active=None, cmd_freq=100, reset_simulation_fcn=None):

        if interface == 'ros':
            self.interface = RobotROSEnvInterface(robot_name='bigman',
                                                  mode=mode,
                                                  body_part_active=body_part_active, cmd_type=command_type,
                                                  observation_active=observation_active,
                                                  state_active=state_active,
                                                  cmd_freq=cmd_freq,
                                                  reset_simulation_fcn=reset_simulation_fcn)
        else:
            raise NotImplementedError("Only ROS interface has been implemented")

    def send_action(self, action):
        self.interface.send_action(action=action)

    def reset(self, **kwargs):
        self.interface.reset(**kwargs)

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

    def set_conditions(self, conditions):
        return self.interface.set_conditions(conditions)

    def add_condition(self, condition):
        return self.interface.add_condition(condition)

    def remove_condition(self, cond_idx):
        return self.interface.remove_condition(cond_idx)

    def get_conditions(self, cond=None):
        return self.interface.get_conditions(cond)

    def add_q0(self, q0):
        return self.interface.add_q0(q0)

    def get_q0_idx(self, q0):
        return self.interface.get_q0_idx(q0)

    def stop(self):
        return self.interface.stop()

