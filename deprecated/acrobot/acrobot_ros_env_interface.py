from __future__ import print_function
from robolearn.envs.ros_env_interface import *
import numpy as np

from sensor_msgs.msg import JointState as JointStateMsg

class AcrobotROSEnvInterface(ROSEnvInterface):
    def __init__(self, mode='simulation'):
        super(AcrobotROSEnvInterface, self).__init__(mode=mode)

        #Action 1: Joint2, 10Hz
        self.set_action_topic("/acrobot/joint2_effort_controller/command", Float64Msg, 10)

        #Observation 1: Joint state
        self.dof = 2
        self.joint_state_elements = ['position', 'velocity']
        self.set_observation_topic("/acrobot/joint_states", JointStateMsg)

        self.set_init_config([0]*self.dof)  # Only positions can be setted
        self.set_initial_acts([0])

        print("Acrobot ROS-environment ready!\n")

    def set_action(self, action):
        """
        Responsible to convert action array to array of ROS publish types
        :param self:
        :param action:
        :return:
        """
        # Only one continuous action (Action 1)
        self.last_acts = [action[0]]

    def get_observation(self, option=["all"]):
        if "all" or "state":
            observation = np.zeros((self.get_obs_dim(), 1))
            for ii, obs_element in enumerate(self.last_obs):
                if obs_element is None:
                    raise ValueError("obs_element %d is None!!!" % ii)
                for jj in range(len(self.joint_state_elements)):
                    state_element = getattr(obs_element, self.joint_state_elements[jj])
                    for hh in range(self.dof):
                        observation[hh+jj*self.dof] = state_element[hh]
        else:
            raise NotImplementedError("Only 'all' option is available")

        return observation

    def get_reward(self):
        # TODO Implement a better reward function
        current_state = np.zeros((4, 1))
        for ii, obs_element in enumerate(self.last_obs):
            if obs_element is None:
                raise ValueError("obs_element %d is None!!!" % ii)
            for jj in range(len(self.joint_state_elements)):
                state_element = getattr(obs_element, self.joint_state_elements[jj])
                for hh in range(self.dof):
                    current_state[jj+hh*self.dof] = state_element[hh]

        desired_state = np.array([-np.pi, 0, 0, 0]).reshape((4,1))
        r1 = sum(np.power(np.abs((desired_state + np.pi - current_state) % (2*np.pi) - np.pi), 2))
        return r1

    def set_initial_acts(self, initial_acts=[0]):
        self.init_acts = initial_acts

    def set_init_config(self, init_config=[0, 0]):
        self.initial_config = SetModelConfigurationRequest('acrobot', "", ['joint1', 'joint2'],
                                                           init_config)

    def get_action_dim(self):
        return 1

    def get_obs_dim(self):
        return self.dof*len(self.joint_state_elements)
