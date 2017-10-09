from __future__ import print_function
import time
import numpy as np
import rospy
import roslib; roslib.load_manifest('urdfdom_py')
from urdf_parser_py.urdf import URDF

from robolearn.envs.ros_env_interface import ROSEnvInterface
from robolearn.utils.iit.xbot_ros import state_vector_xbot_joint_state, update_xbot_command
from robolearn.utils.gazebo_ros.ros_utils import get_indexes_from_list, obs_vector_joint_state, get_sensor_data
from robolearn.utils.gazebo_ros.ros_utils import joint_state_fields, JointState, JointStateAdvr
from robolearn.utils.iit.iit_robots_params import xbot_joint_state_fields
from robolearn.utils.trajectory.trajectory_interpolators import polynomial5_interpolation
from robolearn.utils.transformations_utils import compute_cartesian_error, pose_transform, quaternion_inner


class GazeboROSEnvInterface(ROSEnvInterface):
    def __init__(self, robot_name=None, mode='simulation', joints_active=None, urdf_file=None, action_types=None,
                 action_topic_infos=None,
                 observation_active=None, state_active=None, cmd_freq=100, robot_dyn_model=None,
                 optional_env_params=None, reset_simulation_fcn=None):
        super(GazeboROSEnvInterface, self).__init__(mode=mode)

        #if robot_name is None:
        #    raise AttributeError("robot has not been defined!")

        if urdf_file is None:
            print("Waiting for URDF model in: %s" % 'robot_description')
            while not rospy.has_param('robot_description'):
                time.sleep(0.5)
            robot_urdf = URDF.from_parameter_server()
        else:
            robot_urdf = URDF.from_xml_file(urdf_file)

        if robot_name is None:
            self.robot_name = robot_urdf.name
        else:
            self.robot_name = robot_name

        self.joints = self.get_active_joints_urdf(robot_urdf)
        self.joint_names = [joint.name for joint in self.joints]

        if joints_active is None:
            self.act_dof = range(len(self.joints))
        else:
            self.act_dof = joints_active

        self.robot_urdf = robot_urdf

        #self.q0 = self.robot_params['q0']
        #self.conditions = []  # Necessary for GPS

        ##TODO:TEMPORAL
        #self.temp_object_name = optional_env_params['temp_object_name']

        #self.robot_dyn_model = robot_dyn_model
        #if self.robot_dyn_model is not None:
        #    self.temp_effort = np.zeros(self.robot_dyn_model.qdot_size)

        ## Set initial configuration using first configuration loaded from default params
        #self.set_init_config(self.q0)

        ## Configure actuated joints
        #self.act_joint_names = self.get_joints_names(body_part_active)
        #self.act_joint_ids = get_indexes_from_list(self.joint_names, self.act_joint_names)
        #self.act_dof = len(self.act_joint_names)

        # ####### #
        # ACTIONS #
        # ####### #

        #Action 1: Joint1:JointN, 100Hz
        #self.cmd_freq = cmd_freq
        if not isinstance(action_topic_infos, list):
            action_topic_infos = [action_topic_infos]

        if not isinstance(action_types, list):
            action_types = [action_types]

        if len(action_types) != len(action_topic_infos):
            raise ValueError("Action type (%d) does not have same length than action_topic_infos (%d)" % (len(action_types),
                                                                                                          len(action_topic_infos)))

        act_idx = [-1]
        for topic_info, action_type in zip(action_topic_infos, action_types):
            # self.set_action_topic("/xbotcore/"+self.robot_name+"/command", CommandAdvr, self.cmd_freq)  # TODO: Check if 100 is OK
            self.set_action_topic(topic_info['name'], topic_info['type'], topic_info['freq'])  # TODO: Check if 100 is OK

            if action_type['name'] == 'xbot_position':
                raise NotImplemented
                #init_cmd_vals = self.initial_config[0][self.get_joints_indexes(body_part_active)]  # TODO: TEMPORAL SOLUTION
                #self.cmd_types.append(cmd_type)
            elif action_type['name'] == 'xbot_effort':
                # TODO: Check if initiate with zeros_like is a good idea
                init_cmd_vals = np.zeros_like(action_type['dof'])
            elif action_type['name'] == 'joint_effort':
                init_cmd_vals = np.zeros_like(action_type['dof'])
            else:
                raise NotImplementedError("Only position command has been implemented!")

            act_idx = range(act_idx[-1] + 1, act_idx[-1] + 1 + action_type['dof'])
            # action_id = self.set_action_type(init_cmd_vals, action_type['name'], act_idx, act_joint_names=self.act_joint_names)

            action_id = self.set_action_type(init_cmd_vals, action_type['name'], act_idx,
                                             ros_msg_class=topic_info['type'])

            print("Sending %s action!!" % action_type['name'])

        # After all actions have been configured
        #self.set_initial_acts(initial_acts=[action_type['ros_msg'] for action_type in self.action_types])  # TODO: Check if it is useful or not
        self.act_dim = self.get_total_action_dof()
        self.act_vector = np.zeros(self.act_dim)
        self.prev_act = np.zeros(self.act_dim)

        ## ######## #
        ## TEMPORAL #
        ## ######## #
        #self.temp_joint_pos_state = np.zeros(self.robot_dyn_model.qdot_size)  # Assuming joint state only gives actuated joints state
        #self.temp_joint_vel_state = np.zeros(self.robot_dyn_model.qdot_size)
        #self.temp_joint_effort_state = np.zeros(self.robot_dyn_model.qdot_size)
        #self.temp_joint_stiffness_state = np.zeros(self.robot_dyn_model.qdot_size)
        #self.temp_joint_damping_state = np.zeros(self.robot_dyn_model.qdot_size)
        #self.temp_joint_effort_reference = np.zeros(self.robot_dyn_model.qdot_size)
        #self.temp_joint_state_id = []
        #self.temp_subscriber = rospy.Subscriber("/xbotcore/bigman/joint_states", JointStateAdvr,
        #                                        self.temp_state_callback, (self.temp_joint_state_id,
        #                                                                   self.temp_joint_pos_state,
        #                                                                   self.temp_joint_vel_state,
        #                                                                   self.temp_joint_effort_state,
        #                                                                   self.temp_joint_stiffness_state,
        #                                                                   self.temp_joint_damping_state,
        #                                                                   self.temp_joint_effort_reference))


        #self.distance_vectors_idx = list()
        #self.distance_vectors_params = list()
        #self.distance_vectors = list()
        #self.prev_quat_vectors = list()
        #self.target_pose = np.zeros(7)
        #self.robot_pose = np.zeros(7)
        #self.receiving_target = False
        #self.temp_subscriber = rospy.Subscriber("/optitrack/relative_poses", RelativePose,
        #                                        self.temp_target_callback)


        #self.distance_object_vector = None


        # ############ #
        # OBSERVATIONS #
        # ############ #
        #if observation_active is None:
        #    observation_active = self.robot_params['observation_active']

        obs_idx = [-1]
        for obs_to_activate in observation_active:
            if obs_to_activate['type'] == 'xbot_joint_state':
                ros_topic_type = JointStateAdvr
                self.obs_joint_fields = list(obs_to_activate['fields'])  # Used in get_obs
                self.obs_joint_names = [self.joint_names[id] for id in obs_to_activate['joints']]
                self.obs_joint_ids = get_indexes_from_list(self.joint_names, self.obs_joint_names)
                obs_dof = len(obs_to_activate['fields'])*len(self.obs_joint_names)

            elif obs_to_activate['type'] == 'joint_state':
                ros_topic_type = obs_to_activate['ros_class']
                self.obs_joint_fields = list(obs_to_activate['fields'])  # Used in get_obs
                self.obs_joint_names = [self.joint_names[id] for id in obs_to_activate['joints']]
                self.obs_joint_ids = get_indexes_from_list(self.joint_names, self.obs_joint_names)
                obs_dof = len(obs_to_activate['fields'])*len(self.obs_joint_names)

            elif obs_to_activate['type'] == 'ft_sensor':
                ros_topic_type = WrenchStamped
                self.obs_ft_sensor_fields = list(obs_to_activate['fields'])  # Used in get_obs
                obs_dof = sum([ft_sensor_dof[x] for x in obs_to_activate['fields']])

            elif obs_to_activate['type'] == 'imu':
                ros_topic_type = Imu
                self.obs_imu_sensor_fields = list(obs_to_activate['fields'])  # Used in get_obs
                obs_dof = sum([imu_sensor_dof[x] for x in obs_to_activate['fields']])

            # elif obs_to_activate['type'] == 'optitrack':
            #     ros_topic_type = RelativePose
            #     self.obs_optitrack_fields = list(obs_to_activate['fields'])  # Used in get_obs
            #     self.obs_optitrack_bodies = obs_to_activate['bodies']
            #     obs_dof = sum([optitrack_dof[x]*len(self.obs_optitrack_bodies) for x in obs_to_activate['fields']])

            # elif obs_to_activate['type'] == 'prev_cmd':
            #     obs_dof = self.act_dim
            #     obs_to_activate['ros_topic'] = None
            #     obs_to_activate['fields'] = None
            #     ros_topic_type = 'prev_cmd'

            # elif obs_to_activate['type'] == 'fk_pose':
            #     obs_dof = 0
            #     if 'orientation' in obs_to_activate['fields']:
            #         obs_dof += 3
            #     if 'position' in obs_to_activate['fields']:
            #         obs_dof += 3
            #     obs_to_activate['ros_topic'] = None
            #     ros_topic_type = 'fk_pose'

            # elif obs_to_activate['type'] == 'object_pose':
            #     obs_dof = 0
            #     if 'orientation' in obs_to_activate['fields']:
            #         obs_dof += 3
            #     if 'position' in obs_to_activate['fields']:
            #         obs_dof += 3
            #     obs_to_activate['ros_topic'] = None
            #     ros_topic_type = 'object_pose'

            else:
                raise NotImplementedError("observation %s is not supported!!" % obs_to_activate['type'])

            if obs_to_activate['type'] in ['joint_state', 'optitrack', 'xbot_joint_state']:
                obs_msg_id = self.set_observation_topic(obs_to_activate['ros_topic'], ros_topic_type,
                                                        obs_to_activate['fields']+['name'])
            else:
                obs_msg_id = self.set_observation_topic(obs_to_activate['ros_topic'], ros_topic_type,
                                                        obs_to_activate['fields'])

            ## TODO: Find a better way
            #if obs_to_activate['type'] == 'prev_cmd':
            #    self.last_obs[obs_msg_id] = self.prev_act

            ## TODO: Find a better way
            #if obs_to_activate['type'] == 'fk_pose':
            #    self.distance_vectors_idx.append(obs_msg_id)
            #    self.distance_vectors_params.append({'body_name': obs_to_activate['body_name'],
            #                                         'body_offset': obs_to_activate['body_offset'],
            #                                         'target_offset': obs_to_activate['target_offset'],
            #                                         'fields': obs_to_activate['fields']})
            #    self.distance_vectors.append(np.zeros(obs_dof))
            #    self.prev_quat_vectors.append(np.array([0, 0, 0, 1, 0, 0, 0]))
            #    self.last_obs[obs_msg_id] = self.distance_vectors[-1]
            #    print("Waiting to receive target message")
            #    while self.receiving_target is False:
            #        pass

            ## TODO: Find a better way
            #if obs_to_activate['type'] == 'object_pose':
            #    self.distance_object_vector_idx = obs_msg_id
            #    self.distance_object_vector_params = {'body_name': obs_to_activate['body_name'],
            #                                          'target_rel_pose': obs_to_activate['target_rel_pose'],
            #                                          'fields': obs_to_activate['fields']}
            #    self.distance_object_vector = np.zeros(obs_dof)
            #    self.prev_quat_object_vector = np.array([0, 0, 0, 1, 0, 0, 0])
            #    self.last_obs[obs_msg_id] = self.distance_object_vector
            #    print("Waiting to receive target object message")
            #    while self.receiving_target is False:
            #        pass

            obs_idx = range(obs_idx[-1] + 1, obs_idx[-1] + 1 + obs_dof)

            print("Waiting to receive %s message in %s ..." % (obs_to_activate['type'], obs_to_activate['ros_topic']))
            while self.last_obs[obs_msg_id] is None:
                pass
                # print("Waiting to receive joint_state message...")

            obs_id = self.set_observation_type(obs_to_activate['name'], obs_msg_id,
                                               obs_to_activate['type'], obs_idx)
            print("Receiving %s observation!!" % obs_to_activate['type'])

        self.obs_dim = self.get_total_obs_dof()

        # ##### #
        # STATE #
        # ##### #
        #if state_active is None:
        #    state_active = self.robot_params['state_active']

        state_idx = [-1]
        for state_to_activate in state_active:
            if state_to_activate['type'] == 'xbot_joint_state':
                self.state_joint_names = [self.joint_names[id] for id in state_to_activate['joints']]
                for ii, state_element in enumerate(state_to_activate['fields']):
                    # Check if state field is observed
                    if state_element not in self.obs_joint_fields:
                        raise AttributeError("Joint state type %s is not being observed. Current observations are %s" %
                                             (state_element, self.obs_joint_fields))
                    state_dof = len(state_to_activate['joints'])
                    state_idx = range(state_idx[-1]+1, state_idx[-1] + 1 + state_dof)
                    state_id = self.set_state_type(state_element,  # State name
                                                   self.get_obs_idx(name='xbot_joint_state'),  # Obs_type ID
                                                   state_element,  # State type
                                                   state_idx)  # State indexes

            elif state_to_activate['type'] == 'joint_state':
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
                # for hh, body_name in enumerate(self.state_optitrack_bodies):
                    # for ii, state_element in enumerate(self.state_optitrack_fields):
                    #     if not state_element in self.obs_optitrack_fields:
                    #         raise AttributeError("Joint state type %s is not being observed. Current observations are %s" %
                    #                              (state_element, self.obs_joint_fields))
                state_dof = len(self.state_optitrack_bodies)*sum([optitrack_dof[x] for x in self.state_optitrack_fields])
                state_idx = range(state_idx[-1] + 1, state_idx[-1] + 1 + state_dof)
                state_id = self.set_state_type('optitrack',  # State name
                                               self.get_obs_idx(name='optitrack'),  # Obs_type Name
                                               'optitrack',  # State type
                                               state_idx)  # State indexes

            elif state_to_activate['type'] == 'prev_cmd':
                state_dof = len(self.prev_act)
                state_idx = range(state_idx[-1] + 1, state_idx[-1] + 1 + state_dof)
                state_id = self.set_state_type('prev_cmd',  # State name
                                               self.get_obs_idx(name='prev_cmd'),  # Obs_type Name
                                               'prev_cmd',  # State type
                                               state_idx)  # State indexes

            elif state_to_activate['type'] == 'fk_pose':
                distance_vector_idx = self.distance_vectors_idx.index(self.get_obs_idx(name=state_to_activate['name']))
                state_dof = len(self.distance_vectors[distance_vector_idx])
                state_idx = range(state_idx[-1] + 1, state_idx[-1] + 1 + state_dof)
                state_id = self.set_state_type(state_to_activate['name'],  # State name
                                               self.get_obs_idx(name=state_to_activate['name']),  # Obs_type Name
                                               state_to_activate['type'],  # State type
                                               state_idx)  # State indexes

            elif state_to_activate['type'] == 'object_pose':
                state_dof = len(self.distance_object_vector)
                state_idx = range(state_idx[-1] + 1, state_idx[-1] + 1 + state_dof)
                state_id = self.set_state_type(state_to_activate['name'],  # State name
                                               self.get_obs_idx(name=state_to_activate['name']),  # Obs_type Name
                                               state_to_activate['type'],  # State type
                                               state_idx)  # State indexes

            else:
                raise NotImplementedError("state %s is not supported!!" % state_to_activate['type'])

        self.state_dim = self.get_total_state_dof()


        # ##### #
        # RESET #
        # ##### #
        # # TODO: Find a better way to reset the robot
        # self.srv_xbot_comm_plugin = rospy.ServiceProxy('/XBotCommunicationPlugin_switch', SetBool)
        # self.srv_homing_ex_plugin = rospy.ServiceProxy('/HomingExample_switch', SetBool)
        self.reset_simulation_fcn = reset_simulation_fcn

        ## Resetting before get initial state
        ##print("Resetting %s robot to initial q0[0]..." % self.robot_name)
        #print("NO RESETTING BEFORE PUBLISHING!!")
        #self.x0 = self.get_state()


        ## TEMPORAL
        #self.temp_publisher = rospy.Publisher("/xbotcore/bigman/command", CommandAdvr, queue_size=10)
        #self.pub_rate = rospy.Rate(100)
        #self.des_cmd = CommandAdvr()
        #self.des_cmd.name = self.act_joint_names

        self.run()

        print("%s ROS-environment ready!" % self.robot_name.upper())

    @staticmethod
    def get_active_joints_urdf(robot_urdf):
        joints = list()
        for joint in robot_urdf.joints:
            if joint.type != 'fixed':
                joints.append(joint)
        return joints

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
        self.act_vector[:] = action[:]
        for ii, des_action in enumerate(self.action_types):
            if des_action['type'] in ['position', 'velocity', 'effort']:
                self.prev_act[:] = action[:]  # TODO: TEMPORAL, ASSUMING ONLY ADVR_COMMAND IS IN PREV_ACT

                if self.cmd_type == 'velocity':  # TODO: TEMPORAL HACK / Velocity not implemented
                    # current_pos = state_vector_joint_state(['link_position'], self.act_joint_names,
                    #                                        self.get_obs_ros_msg(name='joint_state')).ravel()
                    current_pos = state_vector_joint_state(['position'], self.act_joint_names,
                                                           self.get_action_ros_msg(action_type='position')).ravel()
                    vel = self.act_vector[des_action['act_idx']]*1./self.cmd_freq
                    # now = rospy.get_rostime()
                    self.act_vector[des_action['act_idx']] = vel + current_pos  # Integrating position
                if self.cmd_type == 'effort':  # TODO: TEMPORAL HACK / Add gravity compensation
                    if self.robot_dyn_model is not None:
                        # current_pos = np.zeros(self.robot_dyn_model.qdot_size)
                        # current_pos[self.obs_joint_ids] = obs_vector_joint_state(['link_position'],
                        #                                                          self.obs_joint_names,
                        #                                                          self.get_obs_ros_msg(name='joint_state')).ravel()
                        current_pos = self.temp_joint_pos_state
                        self.robot_dyn_model.update_gravity_forces(self.temp_effort, current_pos)
                        # self.robot_dyn_model.update_nonlinear_forces(self.temp_effort, current_pos)
                        self.act_vector[des_action['act_idx']] += self.temp_effort[self.act_joint_ids]
                        # self.des_cmd.position = []
                        # # self.des_cmd.effort = self.temp_effort[self.act_joint_ids]
                        # self.des_cmd.effort = self.act_vector[des_action['act_idx']]
                        # self.des_cmd.stiffness = np.zeros_like(self.temp_effort[self.act_joint_ids])
                        # self.des_cmd.damping = np.zeros_like(self.temp_effort[self.act_joint_ids])
                update_xbot_command(des_action['ros_msg'], des_action['type'], self.act_vector[des_action['act_idx']])

            elif des_action['type'] == 'joint_effort':
                des_action['ros_msg'].data = self.act_vector[des_action['act_idx']]

            else:
                raise NotImplementedError("Only Advr commands: position, velocity or effort available!")


        # Start publishing in ROS
        self.publish_action = True
        # self.temp_publisher.publish(self.des_cmd)
        # self.publish_action = False

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

        for oo, obs in enumerate(self.obs_types):
            if obs['type'] == 'xbot_joint_state':
                observation[obs['obs_idx']] = obs_vector_joint_state(self.obs_joint_fields,
                                                                     self.obs_joint_names,
                                                                     obs['ros_msg'])
            elif obs['type'] == 'joint_state':
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

            elif obs['type'] == 'prev_cmd':
                observation[obs['obs_idx']] = self.prev_act.copy()

            elif obs['type'] == 'fk_pose':
                fk_pose_idx = self.distance_vectors_idx.index(oo)
                observation[obs['obs_idx']] = self.distance_vectors[fk_pose_idx].copy()

            elif obs['type'] == 'object_pose':
                observation[obs['obs_idx']] = self.distance_object_vector.copy()

            else:
                raise NotImplementedError("Observation type %s has not been implemented in get_observation()" %
                                          obs['type'])

        return observation

    def get_state(self):
        state = np.empty(self.state_dim)

        for xx, x in enumerate(self.state_types):
            if x['type'] in xbot_joint_state_fields and issubclass(type(x['ros_msg']), JointStateAdvr):
                state[x['state_idx']] = \
                    get_xbot_sensor_data(x['ros_msg'], x['type'])[get_indexes_from_list(x['ros_msg'].name,
                                                                                        self.state_joint_names)]

            elif x['type'] in joint_state_fields and issubclass(type(x['ros_msg']), JointState):
                state[x['state_idx']] = \
                    get_sensor_data(x['ros_msg'], x['type'])[get_indexes_from_list(x['ros_msg'].name,
                                                                                        self.state_joint_names)]

            elif x['type'] == 'optitrack':
                state[x['state_idx']] = obs_vector_optitrack(self.state_optitrack_fields,
                                                             self.state_optitrack_bodies, x['ros_msg'])

            elif x['type'] == 'prev_cmd':
                state[x['state_idx']] = self.prev_act[:]#.copy()

            elif x['type'] == 'fk_pose':
                # distance_vector_idx = self.distance_vectors_idx.index(self.get_obs_idx(name=state_to_activate['name']))
                state[x['state_idx']] = x['ros_msg'][:]

            elif x['type'] == 'object_pose':
                # distance_vector_idx = self.distance_vectors_idx.index(self.get_obs_idx(name=state_to_activate['name']))
                state[x['state_idx']] = x['ros_msg'][:]

            else:
                raise NotImplementedError("State type %s has not been implemented in get_state()" %
                                          x['type'])
        return state

    def get_obs_info(self, name=None):
        """
        Return Observation info dictionary.
        :param name: Name of the observation. If not specified, returns for all the observations.
        :return: obs_info dictionary with keys: names, dimensions and idx.
        """
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
        """
        Return State info dictionary.
        :param name: Name of the state. If not specified, returns for all the states.
        :return: state_info dictionary with keys: names, dimensions and idx.
        """
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
        """
        Return Observation and State info dictionary.
        :return: Dictionary with obs_info and state_info dictionaries. Each one with keys: names, dimensions and idx.
        """
        env_info = {'obs': self.get_obs_info(),
                    'state': self.get_state_info()}
        return env_info

    def get_q_from_condition(self, condition):
        """
        Get joint positions from condition (state)
        :param condition: Condition ID
        :return: Joint position array
        """
        state_info = self.get_state_info()
        if 'link_position' in state_info['names']:
            return condition[state_info['idx'][state_info['names'].index('link_position')]]
        else:
            raise TypeError("Link position is not in the state!!")

    def reset(self, time=None, freq=None, cond=0):
        # Stop First
        self.stop()

        # if freq is None:
        #     freq = 100
        # if time is None:
        #     time = 5
        # if cond > len(self.conditions)-1:
        #     raise AttributeError("Desired condition not available. %d > %d" % (cond, len(self.conditions)-1))

        # N = int(np.ceil(time*freq))
        # pub_rate = rospy.Rate(freq)
        # reset_cmd = CommandAdvr()
        # # Wait for getting zero velocity and acceleration
        # # rospy.sleep(1)  # Because I need to find a good way to reset
        # reset_cmd.name = self.act_joint_names
        # reset_publisher = rospy.Publisher("/xbotcore/"+self.robot_name+"/command", CommandAdvr, queue_size=10)
        # joint_positions = get_last_xbot_state_field(self.robot_name, 'link_position', self.act_joint_names)
        # final_positions = np.zeros(7)
        # final_positions[1] = np.deg2rad(-90)
        # joint_trajectory = polynomial5_interpolation(N*2, final_positions, joint_positions)[0]
        # print('TODO: TEMPORALLY MOVING TO A VIA POINT IN RESET')
        # for ii in range(joint_trajectory.shape[0]):
        #     reset_cmd.position = joint_trajectory[ii, :]
        #     reset_publisher.publish(reset_cmd)
        #     pub_rate.sleep()
        # joint_positions = get_last_xbot_state_field(self.robot_name, 'link_position', self.act_joint_names)
        # final_positions = self.get_q_from_condition(self.conditions[cond])
        # joint_trajectory = polynomial5_interpolation(N*2, final_positions, joint_positions)[0]
        # print("Moving to condition '%d' in position control mode..." % cond)
        # for ii in range(joint_trajectory.shape[0]):
        #     reset_cmd.position = joint_trajectory[ii, :]
        #     reset_publisher.publish(reset_cmd)
        #     pub_rate.sleep()
        # rospy.sleep(5)  # Because I need to find a good way to reset

        # Resetting Gazebo also
        super(GazeboROSEnvInterface, self).reset(model_name=self.robot_name)

        # Custom simulation reset function
        if self.mode == 'simulation' and self.reset_simulation_fcn is not None:
            # self.reset_simulation_fcn(self.conditions[cond], self.get_state_info())
            self.reset_simulation_fcn.reset(cond)

    def set_conditions(self, conditions):
        # TODO: Check conditions size
        self.conditions = conditions

    def add_condition(self, condition):
        if len(condition) != self.state_dim:
            raise AttributeError("Condition and state does not have same dimension!! (%d != %d)" % (len(condition),
                                                                                                    self.state_dim))
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
        """
        Stop the robot (Change it to position mode)
        :return: None
        """
        self.publish_action = False  # Stop sending
        self.prev_act[:] = 0

        time = 1
        freq = 100
        N = int(np.ceil(time*freq))
        for ii in range(N):
            self.send_action(np.zeros(self.act_dim))

        self.publish_action = False  # Stop sending
        self.prev_act[:] = 0

        #pub_rate = rospy.Rate(freq)
        #stop_cmd = CommandAdvr()
        #stop_publisher = rospy.Publisher("/xbotcore/"+self.robot_name+"/command", CommandAdvr, queue_size=10)
        #joint_positions = get_last_xbot_state_field(self.robot_name, 'link_position', self.act_joint_names)
        #stop_cmd.name = self.act_joint_names
        #joint_ids = [self.joint_names.index(joint_name) for joint_name in self.act_joint_names]
        #print("Stop robot! Changing to position control mode...")
        #for ii in range(N):
        #    stop_cmd.position = joint_positions
        #    stop_cmd.stiffness = bigman_params['stiffness_gains'][joint_ids]
        #    stop_cmd.damping = bigman_params['damping_gains'][joint_ids]
        #    stop_publisher.publish(stop_cmd)
        #    pub_rate.sleep()

    def temp_state_callback(self, data, params):
        joint_ids = params[0]
        joint_pos = params[1]
        joint_vel = params[2]
        joint_effort = params[3]
        joint_stiffness = params[4]
        joint_damping = params[5]
        joint_effort_reference = params[6]
        # if not joint_ids:
        #     joint_ids[:] = [bigman_params['joints_names'].index(name) for name in data.name]
        joint_ids[:] = [bigman_params['joints_names'].index(name) for name in data.name]
        joint_pos[joint_ids] = data.link_position
        joint_effort[joint_ids] = data.effort
        joint_vel[joint_ids] = data.link_velocity
        joint_stiffness[joint_ids] = data.stiffness
        joint_damping[joint_ids] = data.damping
        joint_effort_reference[joint_ids] = data.effort_reference

    def temp_target_callback(self, data):
        self.receiving_target = True
        box_idx = data.name.index(self.temp_object_name)
        self.target_pose[0] = data.pose[box_idx].orientation.x
        self.target_pose[1] = data.pose[box_idx].orientation.y
        self.target_pose[2] = data.pose[box_idx].orientation.z
        self.target_pose[3] = data.pose[box_idx].orientation.w
        self.target_pose[4] = data.pose[box_idx].position.x
        self.target_pose[5] = data.pose[box_idx].position.y
        self.target_pose[6] = data.pose[box_idx].position.z

        robot_idx = data.name.index('base_link')
        self.robot_pose[0] = data.pose[robot_idx].orientation.x
        self.robot_pose[1] = data.pose[robot_idx].orientation.y
        self.robot_pose[2] = data.pose[robot_idx].orientation.z
        self.robot_pose[3] = data.pose[robot_idx].orientation.w
        self.robot_pose[4] = data.pose[robot_idx].position.x
        self.robot_pose[5] = data.pose[robot_idx].position.y
        self.robot_pose[6] = data.pose[robot_idx].position.z

        q = self.temp_joint_pos_state

        if self.distance_vectors:
            for hh, distance_vector in enumerate(self.distance_vectors):
                tgt = pose_transform(self.target_pose, self.distance_vectors_params[hh]['target_offset'])

                op_point = self.robot_dyn_model.fk(self.distance_vectors_params[hh]['body_name'],
                                                   q=q,
                                                   body_offset=self.distance_vectors_params[hh]['body_offset'],
                                                   update_kinematics=True,
                                                   rotation_rep='quat')

                # Check quaternion inversion
                if quaternion_inner(op_point[:4], self.prev_quat_vectors[hh][:4]) < 0:
                    op_point[:4] *= -1
                    # print('CHANGING QUATERNION TRANSFORMATION!!!')
                self.prev_quat_vectors[hh] = op_point[:]

                distance = compute_cartesian_error(op_point, tgt)
                prev_idx = 0
                for ii, obs_field in enumerate(self.distance_vectors_params[hh]['fields']):
                    if obs_field == 'position':
                        self.distance_vectors[hh][prev_idx:prev_idx+3] = distance[-3:]
                        prev_idx += 3
                    elif obs_field == 'orientation':
                        self.distance_vectors[hh][prev_idx:prev_idx+3] = distance[:3]
                        prev_idx += 3
                    else:
                        raise ValueError("Wrong fk_pose field")

        if self.distance_object_vector is not None:
            object_pose = self.target_pose
            tgt = self.distance_object_vector_params['target_rel_pose']
            distance = compute_cartesian_error(object_pose, tgt)
            prev_idx = 0
            for obs_field in self.distance_object_vector_params['fields']:
                if obs_field == 'position':
                    self.distance_object_vector[prev_idx:prev_idx+3] = distance[-3:]
                    prev_idx += 3
                elif obs_field == 'orientation':
                    self.distance_object_vector[prev_idx:prev_idx+3] = distance[:3]
                    prev_idx += 3
                else:
                    raise ValueError("Wrong fk_pose field")
            #print(tgt)
            #print(object_pose)
            #print(self.distance_object_vector)
            #print("-----")

    def get_target_pose(self):
        return self.target_pose.copy()

    def get_robot_pose(self):
        return self.robot_pose.copy()
