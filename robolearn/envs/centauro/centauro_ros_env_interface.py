from __future__ import print_function
from robolearn.envs.ros_env_interface import *
import numpy as np
import copy

from XCM.msg import JointStateAdvr
from XCM.msg import CommandAdvr
from sensor_msgs.msg import JointState as JointStateMsg
from sensor_msgs.msg import Imu as ImuMsg

from custom_effort_controllers.msg import CommandArrayStamped
from custom_effort_controllers.msg import Command

from robolearn.utils.iit_robots_params import centauro_params
from robolearn.utils.iit_robots_ros import *



class CentauroROSEnvInterface(ROSEnvInterface):
    def __init__(self, mode='simulation', body_part_active='LA', cmd_type='position',
                 state_vars=[]):
        super(CentauroROSEnvInterface, self).__init__(mode=mode)

        # Centauro fields
        self.robot_params = centauro_params
        self.joint_names = self.robot_params['joints_names']
        self.joint_ids = self.robot_params['joint_ids']
        self.q0 = self.robot_params['q0']

        # Set initial configuration using first configuration loaded from default params
        self.set_init_config(self.q0)

        # Configure actuated joints
        self.act_joint_names = self.get_joints_names(body_part_active)
        self.act_dof = len(self.act_joint_names)

        # ############ #
        # OBSERVATIONS #
        # ############ #

        # Observation 1: Joint state
        obs_msg_id = self.set_observation_topic("/xbotcore/centauro/joint_states", JointStateAdvr)
        #self.topic_obs_info.append((self.dof*len(self.joint_state_elements), "joint_state"))
        obs_idx = range(14)  # TODO: THIS IS A DUMMY VALUE!!!

        print("Waiting to receive joint_state message...")
        while self.last_obs[obs_msg_id] is None:
            pass
            #print("Waiting to receive joint_state message...")
        joint_state_obs_id = self.set_observation_type(self.last_obs[obs_msg_id], 'joint_state', obs_idx)
        print("Receiving Joint_state observation!!")

        self.obs_dim = self.get_total_obs_dof()



        # ##### #
        # STATE #
        # ##### #
        self.joint_state_elements = ['motor_position', 'motor_velocity']
        # TODO: Temporally, assuming that actuated joints are the only joints who are part of the state. Solved with parameter in class
        self.state_joint_names = self.act_joint_names
        total_q_idx = range(len(self.joint_state_elements)*len(self.state_joint_names))
        #joint_state_dof = len(total_q_idx)
        for ii, state_element in enumerate(self.joint_state_elements):
            state_idx = total_q_idx[len(self.state_joint_names)*ii:len(self.state_joint_names)*(ii+1)]
            state_id = self.set_state_type(self.obs_types[joint_state_obs_id], state_element, state_idx)

        self.state_dim = self.get_total_state_dof()

        #self.set_x0(np.zeros(31))  # TODO: Check if this is useful
        ## State from state_vars
        #if not state_vars:  # Fully observed
        #    self.state_dim = self.obs_dim
        #else:
        #    for var in state_vars:
        #        # TODO: Create something for this
        #        pass

        # ####### #
        # ACTIONS #
        # ####### #

        #Action 1: Joint1:JointN, 100Hz
        self.set_action_topic("/xbotcore/centauro/command", CommandAdvr, 100)  # TODO: Check if 100 is OK
        if cmd_type == 'position':
            init_cmd_vals = self.initial_config[0][self.get_joints_indeces(body_part_active)]  # TODO: TEMPORAL SOLUTION
        else:
            raise NotImplementedError("Only position command has been implemented!")

        act_idx = range(self.act_dof)
        action_id = self.set_action_type(init_cmd_vals, cmd_type, act_idx, act_joint_names=self.act_joint_names)

        # After all actions have been configured
        self.set_initial_acts(initial_acts=[action_type['ros_msg'] for action_type in self.action_types])  # TODO: Check if it is useful or not
        self.act_dim = self.get_total_action_dof()

        self.run()

        print("Centauro ROS-environment ready!\n")

    def send_action(self, action):
        """
        Responsible to convert action array to array of ROS publish types
        :param self:
        :param action:
        :return:
        """
        for ii, des_action in enumerate(self.action_types):
            if des_action['type'] in ['position', 'velocity', 'effort']:
                self.action_types[ii]['ros_msg'] = update_advr_command(des_action['ros_msg'],
                                                                       des_action['type'],
                                                                       action[des_action['act_idx']])
            else:
                raise NotImplementedError("Only advr commands available!")

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

    def set_init_config(self, init_config):
        #self.initial_config = SetModelConfigurationRequest('bigman', "", self.joint_names,
        #                                                   init_config)
        self.initial_config = []
        for config in init_config:
            self.initial_config.append(np.array(config))

    def get_action_dim(self):
        return self.act_dim

    def get_state_dim(self):
        return self.state_dim

    def get_obs_dim(self):
        return self.obs_dim

    def get_x0(self):
        return self.x0

    def set_x0(self, x0):
        self.x0 = x0

    def get_joints_names(self, body_part):
        if body_part not in self.joint_ids:
            raise ValueError("wrong body part option")

        return [self.joint_names[joint_id] for joint_id in self.joint_ids[body_part]]

    def get_joints_indeces(self, joint_names):
        if isinstance(joint_names, list):  # Assuming is a list of joint_names
            return [self.joint_names.index(a) for a in joint_names]
        else:  # Assuming is a body_part string
            if joint_names not in self.joint_ids:
                raise ValueError("wrong body part option")
            return self.joint_ids[joint_names]


