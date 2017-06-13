from __future__ import print_function
from robolearn.envs.ros_env_interface import *
import numpy as np
import copy

from robolearn.utils.iit.iit_robots_ros import *
from robolearn.utils.trajectory_interpolators import polynomial5_interpolation


class RobotROSEnvInterface(ROSEnvInterface):
    def __init__(self, robot_name=None, mode='simulation', body_part_active='LA', cmd_type='position',
                 observation_active=None, state_active=None, cmd_freq=100, reset_simulation_fcn=None):
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
        self.conditions = []  # Necessary for GPS

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

        obs_idx = [-1]
        for obs_to_activate in observation_active:
            if obs_to_activate['type'] == 'joint_state':
                ros_topic_type = JointStateAdvr
                self.obs_joint_fields = list(obs_to_activate['fields'])  # Used in get_obs
                self.obs_joint_names = [self.joint_names[id] for id in obs_to_activate['joints']]
                obs_dof = len(obs_to_activate['fields'])*len(self.obs_joint_names)

            elif obs_to_activate['type'] == 'ft_sensor':
                ros_topic_type = WrenchStamped
                self.obs_ft_sensor_fields = list(obs_to_activate['fields'])  # Used in get_obs
                obs_dof = sum([ft_sensor_dof[x] for x in obs_to_activate['fields']])

            elif obs_to_activate['type'] == 'imu':
                ros_topic_type = Imu
                self.obs_imu_sensor_fields = list(obs_to_activate['fields'])  # Used in get_obs
                obs_dof = sum([imu_sensor_dof[x] for x in obs_to_activate['fields']])

            elif obs_to_activate['type'] == 'optitrack':
                ros_topic_type = RelativePose
                self.obs_optitrack_fields = list(obs_to_activate['fields'])  # Used in get_obs
                self.obs_optitrack_bodies = obs_to_activate['bodies']
                obs_dof = sum([optitrack_dof[x]*len(self.obs_optitrack_bodies) for x in obs_to_activate['fields']])

            else:
                raise NotImplementedError("observation %s is not supported!!" % obs_to_activate['type'])

            if obs_to_activate['type'] in ['joint_state', 'optitrack']:
                obs_msg_id = self.set_observation_topic(obs_to_activate['ros_topic'], ros_topic_type,
                                                        obs_to_activate['fields']+['name'])
            else:
                obs_msg_id = self.set_observation_topic(obs_to_activate['ros_topic'], ros_topic_type,
                                                        obs_to_activate['fields'])

            obs_idx = range(obs_idx[-1] + 1, obs_idx[-1] + 1 + obs_dof)

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

        state_idx = [-1]
        for state_to_activate in state_active:
            if state_to_activate['type'] == 'joint_state':
                self.state_joint_names = [self.joint_names[id] for id in state_to_activate['joints']]
                for ii, state_element in enumerate(state_to_activate['fields']):
                    # Check if state field is observed
                    if state_element not in self.obs_joint_fields:
                        raise AttributeError("Joint state type %s is not being observed. Current observations are %s" %
                                             (state_element, self.obs_joint_fields))
                    state_dof = len(state_to_activate['joints'])
                    state_idx = range(state_idx[-1]+1, state_idx[-1] + 1 + state_dof)
                    state_id = self.set_state_type(state_element,  # State name
                                                   self.get_obs_idx(name='joint_state'),  # Obs_type ID
                                                   state_element,  # State type
                                                   state_idx)  # State indexes

            elif state_to_activate['type'] == 'optitrack':
                self.state_optitrack_bodies = state_to_activate['bodies']
                self.state_optitrack_fields = state_to_activate['fields']
                #for hh, body_name in enumerate(self.state_optitrack_bodies):
                    #for ii, state_element in enumerate(self.state_optitrack_fields):
                    #    if not state_element in self.obs_optitrack_fields:
                    #        raise AttributeError("Joint state type %s is not being observed. Current observations are %s" %
                    #                             (state_element, self.obs_joint_fields))
                state_dof = len(self.state_optitrack_bodies)*sum([optitrack_dof[x] for x in self.state_optitrack_fields])
                state_idx = range(state_idx[-1] + 1, state_idx[-1] + 1 + state_dof)
                state_id = self.set_state_type('optitrack',  # State name
                                               self.get_obs_idx(name='optitrack'),  # Obs_type ID
                                               'optitrack',  # State type
                                               state_idx)  # State indexes

            else:
                raise NotImplementedError("state %s is not supported!!" % state_to_activate['type'])

        self.state_dim = self.get_total_state_dof()


        # ####### #
        # ACTIONS #
        # ####### #

        #Action 1: Joint1:JointN, 100Hz
        self.cmd_freq = cmd_freq
        self.set_action_topic("/xbotcore/"+self.robot_name+"/command", CommandAdvr, self.cmd_freq)  # TODO: Check if 100 is OK
        if cmd_type == 'position':
            init_cmd_vals = self.initial_config[0][self.get_joints_indexes(body_part_active)]  # TODO: TEMPORAL SOLUTION
            self.cmd_type = cmd_type
        elif cmd_type == 'velocity':
            init_cmd_vals = np.zeros_like(self.initial_config[0][self.get_joints_indexes(body_part_active)])
            self.cmd_type = cmd_type
            cmd_type = 'position'  # TEMPORAL
        elif cmd_type == 'effort':
            # TODO: Check if initiate with zeros_like is a good idea
            init_cmd_vals = np.zeros_like(self.initial_config[0][self.get_joints_indexes(body_part_active)])
            self.cmd_type = cmd_type
            cmd_type = 'effort'  # TEMPORAL
        else:
            raise NotImplementedError("Only position command has been implemented!")

        act_idx = range(self.act_dof)
        action_id = self.set_action_type(init_cmd_vals, cmd_type, act_idx, act_joint_names=self.act_joint_names)

        # After all actions have been configured
        self.set_initial_acts(initial_acts=[action_type['ros_msg'] for action_type in self.action_types])  # TODO: Check if it is useful or not
        self.act_dim = self.get_total_action_dof()


        ## ##### #
        ## RESET #
        ## ##### #
        ## TODO: Find a better way to reset the robot
        #self.srv_xbot_comm_plugin = rospy.ServiceProxy('/XBotCommunicationPlugin_switch', SetBool)
        #self.srv_homing_ex_plugin = rospy.ServiceProxy('/HomingExample_switch', SetBool)
        self.reset_simulation_fcn = reset_simulation_fcn

        # Resetting before get initial state
        #print("Resetting %s robot to initial q0[0]..." % self.robot_name)
        print("NO RESETTING BEFORE PUBLISHING!!")
        self.x0 = self.get_state()

        self.run()

        print("%s ROS-environment ready!" % self.robot_name.upper())

    def get_action_dim(self):
        """
        Return the environment's action dimension.
        :return: Action dimension
        :rtype: int
        """
        return self.act_dim

    def get_obs_dim(self):
        """
        Return the environment's observation dimension.
        :return: Observation dimension
        :rtype: int
        """
        return self.obs_dim

    def get_state_dim(self):
        """
        Return the environment's state dimension.
        :return: State dimension
        :rtype: int
        """
        return self.state_dim

    def send_action(self, action):
        """
        Update the ROS messages that will be published from a desired action vector.
        :param action: Desired action vector.
        :return: None
        """
        for ii, des_action in enumerate(self.action_types):
            if des_action['type'] in ['position', 'velocity', 'effort']:
                if self.cmd_type == 'velocity':  #TODO: TEMPORAL HACK / Velocity not implemented
                    #current_pos = state_vector_joint_state(['link_position'], self.act_joint_names, self.get_obs_ros_msg(name='joint_state')).ravel()
                    current_pos = state_vector_joint_state(['position'], self.act_joint_names, self.get_action_ros_msg(action_type='position')).ravel()
                    vel = action[des_action['act_idx']]*1./self.cmd_freq
                    now = rospy.get_rostime()
                    action[des_action['act_idx']] = vel + current_pos  # Integrating position

                update_advr_command(des_action['ros_msg'], des_action['type'], action[des_action['act_idx']])
                #self.action_types[ii]['ros_msg'] = update_advr_command(des_action['ros_msg'],
                #                                                       des_action['type'],
                #                                                       action[des_action['act_idx']])
            else:
                raise NotImplementedError("Only Advr commands: position, velocity or effort available!")

        # Start publishing in ROS
        self.publish_action = True

    def set_initial_acts(self, initial_acts):
        self.init_acts = initial_acts

    def set_init_config(self, init_config):
        #self.initial_config = SetModelConfigurationRequest('bigman', "", self.joint_names,
        #                                                   init_config)
        self.initial_config = []
        for config in init_config:
            self.initial_config.append(np.array(config))

    def get_x0(self):
        return self.x0

    def set_x0(self, x0):
        self.x0 = x0

    def get_joints_names(self, body_part):
        """
        Get the joint names from Robot body part.
        :param body_part: Name of the body part. E.g. LA, RA, WB, etc.
        :type body_part: str
        :return: 
        """
        if body_part not in self.joint_ids:
            raise ValueError("Wrong body part option")

        return [self.joint_names[joint_id] for joint_id in self.joint_ids[body_part]]

    def get_joints_indexes(self, joint_names):
        """
        Get the joint indexes from a list of joint names
        :param joint_names: 
        :return: List of joint indexes (ids)
        :rtype: list
        """
        if isinstance(joint_names, list):  # Assuming is a list of joint_names
            return get_indexes_from_list(self.joint_names, joint_names)
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
                observation[obs['obs_idx']] = obs_vector_ft_sensor(self.obs_ft_sensor_fields,
                                                                   obs['ros_msg'])

            elif obs['type'] == 'imu':
                observation[obs['obs_idx']] = obs_vector_imu(self.obs_imu_sensor_fields,
                                                             obs['ros_msg'])

            elif obs['type'] == 'optitrack':
                observation[obs['obs_idx']] = obs_vector_optitrack(self.obs_optitrack_fields,
                                                                   self.obs_optitrack_bodies,
                                                                   obs['ros_msg'])
            else:
                raise NotImplementedError("Observation type %s has not been implemented in get_observation()" %
                                          obs['type'])

        return observation

    def get_state(self):
        state = np.empty(self.state_dim)

        for x in self.state_types:
            if x['type'] in joint_state_fields:
                state[x['state_idx']] = get_advr_sensor_data(x['ros_msg'], x['type'])[get_indexes_from_list(x['ros_msg'].name,
                                                                                                            self.state_joint_names)]

            elif x['type'] == 'optitrack':
                state[x['state_idx']] = obs_vector_optitrack(self.state_optitrack_fields,
                                                             self.state_optitrack_bodies, x['ros_msg'])

            else:
                raise NotImplementedError("State type %s has not been implemented in get_state()" %
                                          x['type'])

        #print(state)
        return state

    def get_obs_info(self, name=None):
        if name is None:
            obs_info = {'names': [obs['name'] for obs in self.obs_types],
                        'dimensions': [len(obs['obs_idx']) for obs in self.obs_types],
                        'idx': [obs['obs_idx'] for obs in self.obs_types]}
        else:
            obs_idx = self.get_obs_idx(name=name)
            obs_info = {'names': self.obs_types[obs_idx]['name'],
                        'dimensions': len(self.obs_types[obs_idx]['obs_idx']),
                        'idx': self.obs_types[obs_idx]['obs_idx']}
        return obs_info

    def get_state_info(self, name=None):
        if name is None:
            state_info = {'names': [state['name'] for state in self.state_types],
                          'dimensions': [len(state['state_idx']) for state in self.state_types],
                          'idx': [state['state_idx'] for state in self.state_types]}
        else:
            state_idx = self.get_state_idx(name)
            state_info = {'names': self.state_types[state_idx]['name'],
                          'dimensions': len(self.state_types[state_idx]['state_idx']),
                          'idx': self.state_types[state_idx]['state_idx']}
        return state_info

    def get_env_info(self):
        env_info = {'obs': self.get_obs_info(),
                    'state': self.get_state_info()}
        return env_info

    def get_q_from_condition(self, condition):
        state_info = self.get_state_info()
        if 'link_position' in state_info['names']:
            return condition[state_info['idx'][state_info['names'].index('link_position')]]
        else:
            raise TypeError("Link position is not in the state!!")

    def reset(self, time=None, freq=None, cond=0):
        # Stop First
        self.stop()

        if freq is None:
            freq = 100

        if time is None:
            time = 5

        if cond > len(self.conditions)-1:
            raise AttributeError("Desired condition not available. %d > %d" % (cond, len(self.conditions)-1))

        N = int(np.ceil(time*freq))
        pub_rate = rospy.Rate(freq)
        reset_cmd = CommandAdvr()

        # Wait for getting zero velocity and acceleration
        #rospy.sleep(1)  # Because I need to find a good way to reset

        joint_positions = get_last_advr_state_field(self.robot_name, 'link_position', self.act_joint_names)

        reset_cmd.name = self.act_joint_names
        reset_publisher = rospy.Publisher("/xbotcore/"+self.robot_name+"/command", CommandAdvr, queue_size=10)

        final_positions = self.get_q_from_condition(self.conditions[cond])
        joint_trajectory = polynomial5_interpolation(N, final_positions, joint_positions)[0]
        print("Moving to condition '%d' in position control mode..." % cond)
        for ii in range(joint_trajectory.shape[0]):
            reset_cmd.position = joint_trajectory[ii, :]
            reset_publisher.publish(reset_cmd)
            pub_rate.sleep()

        # Interpolate from current position
        rospy.sleep(1)  # Because I need to find a good way to reset

        # Resetting Gazebo also
        super(RobotROSEnvInterface, self).reset(model_name=self.robot_name)

        # Custom simulation reset function
        if self.mode == 'simulation' and self.reset_simulation_fcn is not None:
            self.reset_simulation_fcn(self.conditions[cond], self.get_state_info())

    def set_conditions(self, conditions):
        self.conditions = conditions

    def add_condition(self, condition):
        #TODO: Check condition size
        self.conditions.append(condition)
        return len(self.conditions)-1

    def remove_condition(self, cond_idx):
        self.conditions.pop(cond_idx)

    def get_conditions(self, condition=None):
        return self.conditions[condition] if condition is not None else self.conditions

    def add_q0(self, q0):
        if len(q0) != len(self.q0[-1]):
            raise ValueError("Desired q0 dimension (%d) does not match with Robot (%d)" % (len(q0), len(self.q0[-1])))
        self.q0.append(q0)

        return len(self.q0)-1

    def get_q0_idx(self, q0):
        if len(q0) != len(self.q0[-1]):
            raise ValueError("Desired q0 dimension (%d) does not match with Robot (%d)" % (len(q0), len(self.q0[-1])))
        return self.q0.index(q0)

    def stop(self):
        self.publish_action = False  # Stop sending
        time = 1
        freq = 100
        N = int(np.ceil(time*freq))
        pub_rate = rospy.Rate(freq)
        stop_cmd = CommandAdvr()

        stop_publisher = rospy.Publisher("/xbotcore/"+self.robot_name+"/command", CommandAdvr, queue_size=10)
        joint_positions = get_last_advr_state_field(self.robot_name, 'link_position', self.act_joint_names)
        stop_cmd.name = self.act_joint_names

        joint_ids = [self.joint_names.index(joint_name) for joint_name in self.act_joint_names]
        print("Stop robot! Changing to position control mode...")
        for ii in range(N):
            stop_cmd.position = joint_positions
            stop_cmd.stiffness = bigman_params['stiffness_gains'][joint_ids]
            stop_cmd.damping = bigman_params['damping_gains'][joint_ids]
            stop_publisher.publish(stop_cmd)
            pub_rate.sleep()

