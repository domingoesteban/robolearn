from __future__ import print_function
from robolearn.envs.ros_env_interface import *
import numpy as np
import copy

from robolearn.utils.iit_robots_ros import *


class RobotROSEnvInterface(ROSEnvInterface):
    def __init__(self, robot_name=None, mode='simulation', body_part_active='LA', cmd_type='position',
                 observation_active=None, state_active=None):
        super(RobotROSEnvInterface, self).__init__(mode=mode)

        if robot_name is None:
            raise AttributeError("robot has not been defined!")

        if robot_name == 'centauro':
            # Centauro fields
            self.robot_params = centauro_params
        elif robot_name == 'bigman':
            self.robot_params = bigman_params
        else:
            raise NotImplementedError("robot %s has not been implemented!" % robot_name)

        self.robot_name = robot_name
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
        if observation_active is None:
            observation_active = self.robot_params['observation_active']

        obs_idx = []
        for obs_to_activate in observation_active:
            if obs_to_activate['type'] == 'joint_state':
                ros_topic_type = JointStateAdvr
                self.obs_joint_fields = list(obs_to_activate['fields'])  # Used in get_obs
                self.obs_joint_names = [self.joint_names[id] for id in obs_to_activate['joints']]
                obs_dof = len(obs_to_activate['fields'])*len(self.obs_joint_names)

            elif obs_to_activate['type'] == 'ft_sensor':
                ros_topic_type = WrenchStamped
                self.ft_sensor_fields = list(obs_to_activate['fields'])  # Used in get_obs
                obs_dof = sum([ft_sensor_dof[x] for x in obs_to_activate['fields']])

            elif obs_to_activate['type'] == 'imu':
                ros_topic_type = Imu
                self.imu_sensor_fields = list(obs_to_activate['fields'])  # Used in get_obs
                obs_dof = sum([imu_sensor_dof[x] for x in obs_to_activate['fields']])

            else:
                raise NotImplementedError("observation %s is not supported!!" % obs_to_activate['type'])


            if obs_to_activate['type'] == 'joint_state':
                obs_msg_id = self.set_observation_topic(obs_to_activate['ros_topic'], ros_topic_type,
                                                        obs_to_activate['fields']+['name'])
            else:
                obs_msg_id = self.set_observation_topic(obs_to_activate['ros_topic'], ros_topic_type,
                                                        obs_to_activate['fields'])

            obs_idx = range(len(obs_idx), len(obs_idx) + obs_dof)

            print("Waiting to receive %s message in %s ..." % (obs_to_activate['type'], obs_to_activate['ros_topic']))
            while self.last_obs[obs_msg_id] is None:
                pass
                #print("Waiting to receive joint_state message...")

            obs_id = self.set_observation_type(obs_to_activate['name'], obs_msg_id,
                                               obs_to_activate['type'], obs_idx)
            print("Receiving %s observation!!" % obs_to_activate['type'])

        self.obs_dim = self.get_total_obs_dof()


        # ##### #
        # STATE #
        # ##### #
        if state_active is None:
            state_active = self.robot_params['state_active']

        state_idx = []
        for state_to_activate in state_active:
            if state_to_activate['type'] == 'joint_state':
                self.state_joint_names = [self.joint_names[id] for id in state_to_activate['joints']]
                for ii, state_element in enumerate(state_to_activate['fields']):
                    # Check if state field is observed
                    if not state_element in self.obs_joint_fields:
                        raise AttributeError("Joint state type %s is not being observed. Current observations are %s" %
                                             (state_element, self.obs_joint_fields))
                    state_dof = len(state_to_activate['joints'])
                    state_idx = range(len(state_idx), len(state_idx) + state_dof)
                    state_id = self.set_state_type(state_element, self.get_obs_id('joint_state'), state_element, state_idx)
            else:
                raise NotImplementedError("state %s is not supported!!" % state_to_activate['type'])

        self.state_dim = self.get_total_state_dof()


        # ####### #
        # ACTIONS #
        # ####### #

        #Action 1: Joint1:JointN, 100Hz
        self.set_action_topic("/xbotcore/"+self.robot_name+"/command", CommandAdvr, 100)  # TODO: Check if 100 is OK
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
            return get_indeces_from_list(self.joint_names, joint_names)
        else:  # Assuming is a body_part string
            if joint_names not in self.joint_ids:
                raise ValueError("wrong body part option")
            return self.joint_ids[joint_names]

    def get_observation(self):
        observation = np.empty(self.obs_dim)

        for obs in self.obs_types:
            if obs['type'] == 'joint_state':
                observation[obs['obs_idx']] = obs_vector_joint_state(self.obs_joint_fields,
                                                                     self.obs_joint_names,
                                                                     obs['ros_msg'])

            elif obs['type'] == 'ft_sensor':
                observation[obs['obs_idx']] = obs_vector_ft_sensor(self.ft_sensor_fields,
                                                                   obs['ros_msg'])

            elif obs['type'] == 'imu':
                observation[obs['obs_idx']] = obs_vector_imu(self.imu_sensor_fields,
                                                             obs['ros_msg'])
            else:
                raise NotImplementedError("Observation type %s has not been implemented in get_observation()" %
                                          obs['type'])

        return observation

    def get_state(self):
        state = np.empty(self.state_dim)

        for x in self.state_types:
            state[x['state_idx']] = get_advr_sensor_data(x['ros_msg'], x['type'])[get_indeces_from_list(x['ros_msg'].name,
                                                                                                        self.state_joint_names)]

        #print(state)
        return state

    def get_obs_id(self, name):
        for ii, obs in enumerate(self.obs_types):
            if obs['name'] == name:
                return ii
        raise ValueError("There is not observation with name %s" % name)

    def get_obs_info(self):
        obs_info = {'names': [obs['name'] for obs in self.obs_types],
                    'dimensions': [len(obs['obs_idx']) for obs in self.obs_types],
                    'idx': [obs['obs_idx'] for obs in self.obs_types]}
        return obs_info

    def get_state_info(self):
        state_info = {'names': [state['name'] for state in self.state_types],
                      'dimensions': [len(state['state_idx']) for state in self.state_types],
                      'idx': [state['state_idx'] for state in self.state_types]}
        return state_info

    def get_env_info(self):
        env_info = {'obs': self.get_obs_info(),
                    'state': self.get_state_info()}
        return env_info

