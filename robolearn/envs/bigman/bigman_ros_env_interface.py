from __future__ import print_function
from robolearn.envs.ros_env_interface import *
import numpy as np
import copy

from sensor_msgs.msg import JointState as JointStateMsg
from sensor_msgs.msg import Imu as ImuMsg

from custom_effort_controllers.msg import CommandArrayStamped
from custom_effort_controllers.msg import Command


class BigmanROSEnvInterface(ROSEnvInterface):
    def __init__(self, mode='simulation', joints_active='left_arm', command_type='torque'):
        super(BigmanROSEnvInterface, self).__init__(mode=mode)

        # ############ #
        # OBSERVATIONS #
        # ############ #

        #Observation 1: Joint state
        self.dof = 31
        self.joint_state_elements = ['position', 'velocity']
        self.set_observation_topic("/bigman/joint_states", JointStateMsg)
        self.topic_obs_info.append((self.dof*len(self.joint_state_elements), "joint_state"))
        #Observation 2: IMU1
        self.set_observation_topic("/bigman/sensor/IMU1", ImuMsg)
        self.topic_obs_info.append((10, 'imu'))  # orientation(4) + ang.vel(3) + lin.acc(3)
        #Observation 3: IMU2
        self.set_observation_topic("/bigman/sensor/IMU2", ImuMsg)
        self.topic_obs_info.append((10, 'imu'))  # orientation(4) + ang.vel(3) + lin.acc(3)
        self.obs_dim = sum(i for i,_ in self.topic_obs_info)

        # ####### #
        # ACTIONS #
        # ####### #
        #Action 1: Joint1:JointN, 10Hz
        self.command_type = command_type
        self.set_action_topic("/bigman/group_position_torque_controller/command", CommandArrayStamped, 100)

        if joints_active == 'left_leg':
            self.act_joints_id = range(0, 6)
        elif joints_active == 'right_leg':
            self.act_joints_id = range(6, 12)
        elif joints_active == 'torso':
            self.act_joints_id = range(12, 15)
        elif joints_active == 'left_arm':
            self.act_joints_id = range(15, 22)
        elif joints_active == 'head':
            self.act_joints_id = range(22, 24)
        elif joints_active == 'right_arm':
            self.act_joints_id = range(24, 31)
        elif joints_active == 'lower_body':
            self.act_joints_id = range(0, 12)
        elif joints_active == 'both_arms':
            self.act_joints_id = range(15, 22) + range(24, 31)
        elif joints_active == 'upper_body':
            self.act_joints_id = range(15, 22) + range(24, 31) + range(12, 15)
        elif joints_active == 'full_upper_body':
            self.act_joints_id = range(15, 22) + range(24, 31) + range(12, 15) + range(22, 24)
        elif joints_active == 'whole_body':
            self.act_joints_id = range(0, 31)
        else:
            raise ValueError("Wrong joints_active argument!")


        self.joint_names = ['LHipLat', # 0
                            'LHipYaw', # 1
                            'LHipSag', # 2
                            'LKneeSag', # 3
                            'LAnkSag', # 4
                            'LAnkLat', # 5
                            'RHipLat', # 6
                            'RHipYaw', # 7
                            'RHipSag', # 8
                            'RKneeSag', # 9
                            'RAnkSag', # 10
                            'RAnkLat', # 11
                            'WaistLat', # 12
                            'WaistSag', # 13
                            'WaistYaw', # 14
                            'LShSag', # 15
                            'LShLat', # 16
                            'LShYaw', # 17
                            'LElbj', # 18
                            'LForearmPlate', # 19
                            'LWrj1', # 20
                            'LWrj2', # 21
                            'NeckYawj', # 22
                            'NeckPitchj', # 23
                            'RShSag', # 24
                            'RShLat', # 25
                            'RShYaw', # 26
                            'RElbj', # 27
                            'RForearmPlate', # 28
                            'RWrj1', # 29
                            'RWrj2'] # 30

        self.set_init_config([0]*self.dof)  # Only positions can be setted

        init_bigman_cmd = CommandArrayStamped()
        #init_command = [Command()]*self.dof # TODO: NEVER DO THIS with classes!!!!
        init_command = [Command() for _ in range(self.dof)]

        init_command[0].position = 0    # 'LHipLat',
        init_command[1].position = 0    # 'LHipYaw',
        init_command[2].position = 0    # 'LHipSag',
        init_command[3].position = 0    # 'LKneeSag',
        init_command[4].position = 0    # 'LAnkSag',
        init_command[5].position = 0    # 'LAnkLat',
        init_command[6].position = 0    # 'RHipLat',
        init_command[7].position = 0    # 'RHipYaw',
        init_command[8].position = 0    # 'RHipSag',
        init_command[9].position = 0    # 'RKneeSag',
        init_command[10].position = 0   # 'RAnkSag',
        init_command[11].position = 0   # 'RAnkLat',
        init_command[12].position = 0   # 'WaistLat',
        init_command[13].position = 0   # 'WaistSag',
        init_command[14].position = 0   # 'WaistYaw',
        init_command[15].position = 0   # 'LShSag',
        init_command[16].position = 0   # 'LShLat',
        init_command[17].position = 0   # 'LShYaw',
        init_command[18].position = 0   # 'LElbj',
        init_command[19].position = 0   # 'LForearmPlate',
        init_command[20].position = 0   # 'LWrj1',
        init_command[21].position = 0   # 'LWrj2',
        init_command[22].position = 0   # 'NeckYawj',
        init_command[23].position = 0   # 'NeckPitchj',
        init_command[24].position = 0   # 'RShSag',
        init_command[25].position = 0   # 'RShLat',
        init_command[26].position = 0   # 'RShYaw',
        init_command[27].position = 0   # 'RElbj',
        init_command[28].position = 0   # 'RForearmPlate',
        init_command[29].position = 0   # 'RWrj1',
        init_command[30].position = 0   # 'RWrj2']

        init_command[0].torque = 0    # 'LHipLat',
        init_command[1].torque = 0    # 'LHipYaw',
        init_command[2].torque = 0    # 'LHipSag',
        init_command[3].torque = 0    # 'LKneeSag',
        init_command[4].torque = 0    # 'LAnkSag',
        init_command[5].torque = 0    # 'LAnkLat',
        init_command[6].torque = 0    # 'RHipLat',
        init_command[7].torque = 0    # 'RHipYaw',
        init_command[8].torque = 0    # 'RHipSag',
        init_command[9].torque = 0    # 'RKneeSag',
        init_command[10].torque = 0   # 'RAnkSag',
        init_command[11].torque = 0   # 'RAnkLat',
        init_command[12].torque = 0   # 'WaistLat',
        init_command[13].torque = 0   # 'WaistSag',
        init_command[14].torque = 0   # 'WaistYaw',
        init_command[15].torque = 0   # 'LShSag',
        init_command[16].torque = 0   # 'LShLat',
        init_command[17].torque = 0   # 'LShYaw',
        init_command[18].torque = 0   # 'LElbj',
        init_command[19].torque = 0   # 'LForearmPlate',
        init_command[20].torque = 0   # 'LWrj1',
        init_command[21].torque = 0   # 'LWrj2',
        init_command[22].torque = 0   # 'NeckYawj',
        init_command[23].torque = 0   # 'NeckPitchj',
        init_command[24].torque = 0   # 'RShSag',
        init_command[25].torque = 0   # 'RShLat',
        init_command[26].torque = 0   # 'RShYaw',
        init_command[27].torque = 0   # 'RElbj',
        init_command[28].torque = 0   # 'RForearmPlate',
        init_command[29].torque = 0   # 'RWrj1',
        init_command[30].torque = 0   # 'RWrj2']

        init_bigman_cmd.commands = init_command

        self.set_initial_acts(initial_acts=[init_bigman_cmd])

        self.act_dim = len(self.act_joints_id)

        self.run()

        print("Bigman ROS-environment ready!\n")

    def send_action(self, action):
        """
        Responsible to convert action array to array of ROS publish types
        :param self:
        :param action:
        :return:
        """
        last_act = copy.deepcopy(self.init_acts[0])

        for ii, des_action in enumerate(action):
            # last_act.commands[arm_id+ii].position = des_action
            if self.command_type == 'torque':
                last_act.commands[self.act_joints_id[ii]].torque = des_action
            else:
                last_act.commands[self.act_joints_id[ii]].position = des_action

        self.last_acts = [last_act]
        self.publish_action = True  # TODO Deactivating constant publishing of publish threads

    def read_observation(self, option=["all"]):
        observation = np.zeros((self.get_obs_dim(), 1))
        count_obs = 0

        if "all" or "state":
            for ii, obs_element in enumerate(self.last_obs):
                if obs_element is None:
                    raise ValueError("obs_element %d is None!!!" % ii)
                if self.topic_obs_info[ii][1] == "joint_state":
                    for jj in range(len(self.joint_state_elements)):
                        state_element = getattr(obs_element, self.joint_state_elements[jj])
                        for hh in range(self.dof):
                            observation[hh+jj*self.dof+count_obs] = state_element[hh]
                elif self.topic_obs_info[ii][1] == "imu":
                    observation[count_obs] = obs_element.orientation.x
                    observation[count_obs+1] = obs_element.orientation.y
                    observation[count_obs+2] = obs_element.orientation.z
                    observation[count_obs+3] = obs_element.orientation.w
                    observation[count_obs+4] = obs_element.angular_velocity.x
                    observation[count_obs+5] = obs_element.angular_velocity.y
                    observation[count_obs+6] = obs_element.angular_velocity.z
                    observation[count_obs+7] = obs_element.linear_acceleration.x
                    observation[count_obs+8] = obs_element.linear_acceleration.y
                    observation[count_obs+9] = obs_element.linear_acceleration.z
                else:
                    raise ValueError("wrong topic_obs_info element %d is None!!!" % ii)

                count_obs = count_obs + self.topic_obs_info[ii][0]

        else:
            raise NotImplementedError("Only 'all' option is available")

        return observation

    def set_initial_acts(self, initial_acts):
        self.init_acts = initial_acts

    def set_init_config(self, init_config=[0, 0]):
        self.initial_config = SetModelConfigurationRequest('bigman', "", self.joint_names,
                                                           init_config)

    def get_action_dim(self):
        return self.act_dim

    def get_obs_dim(self):
        return self.obs_dim
