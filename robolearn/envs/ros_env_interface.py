from __future__ import print_function
from robolearn.envs.environment import EnvInterface
from robolearn.utils.iit.iit_robots_ros import copy_class_attr, config_xbot_command

# Useful packages
from threading import Thread

# ROS packages
import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelConfiguration
from std_srvs.srv import Empty
from gazebo_robolearn.srv import ResetPhysicsStatesModel


class ROSEnvInterface(EnvInterface):
    """
    ROSEnvInterface class.
    Responsible to send and receive information from ROS.
    """
    def __init__(self, mode='simulation'):
        super(ROSEnvInterface, self).__init__()

        mode = mode.lower()
        if mode != 'simulation' and mode != 'real':
            raise ValueError("Wrong ROSEnvInterface mode. Options: 'simulation' or 'real'.")

        if mode == 'real':
            raise NotImplementedError("ROSEnvInterface 'real' mode not implemented yet.")

        self.mode = mode

        print("ROSEnvInterface mode: '%s'." % self.mode)

        rospy.init_node('RoboLearnEnvInterface')

        if mode == 'simulation':
            # Create some service proxies to interact with Gazebo simulator
            self.reset_srv = rospy.ServiceProxy('/gazebo/reset_world', Empty)
            #self.reset_srv = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
            self.unpause_srv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
            self.pause_srv = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
            self.set_config_srv = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
            self.reset_model_physics_srv = rospy.ServiceProxy('/gazebo/reset_physics_states_model',
                                                              ResetPhysicsStatesModel)

        self.init_act = 0
        self.init_acts = None
        self.initial_config = None

        # ROS Actions
        self.action_pubs = []
        self.pub_threads = []
        self.action_types = []  # Array of dict = {type, cmd_msg, idx from action array)}
        self.publish_action = False  # Flag that indicates if the action loop will publish the actions in ROS

        # ROS Subscribers
        self.observation_subs = []
        self.obs_types = []  # Array of dict = {type, obs_msg, idx from state array)}
        self.last_obs = []
        self.state_types = []  # Array of dict = {type, obs_msg, idx from state array)}

    def set_action_topic(self, topic_name, topic_type, topic_freq):
        """
        Append an action topic to the action_pubs list.
        :param topic_name: Name of the ROS topic to publish.
        :param topic_type: Type of the ROS topic to publish.
        :param topic_freq: Frequency in which the ROS message will be published.
        :return: Index of the action topic in the action_pubs list.
        """
        action_id = len(self.action_pubs)
        self.action_pubs.append((rospy.Publisher(topic_name, topic_type, queue_size=10), topic_freq))
        return action_id

    def set_observation_topic(self, topic_name, topic_type, attribute_names):
        """
        Append an observation topic to the observation_subs list and a None observation to the last_obs list.
        :param topic_name: Name of the ROS topic to subscribe.
        :param topic_type: Type of the ROS topic to subscribe.
        :param attribute_names: Desired ROS message's fields that will be updated.
        :return: Index of the observation topic in the observation_subs list.
        """
        obs_id = len(self.observation_subs)
        if topic_name is None:
            self.observation_subs.append(None)
        else:
            self.observation_subs.append(rospy.Subscriber(topic_name, topic_type, self.callback_observation,
                                                          (obs_id, attribute_names)))
        self.last_obs.append(None)
        return obs_id

    def callback_observation(self, msg, params):
        # TODO: Check if it is better to get attribute_names from the obs_types
        """

        :param msg:
        :param params:
        :return:
        """
        obs_id = params[0]
        attribute_names = params[1]
        if not len(self.last_obs):
            raise AttributeError("last_obs has not been configured")
        # if isinstance(msg, WrenchStamped):
        #     print(msg)
        # print(msg.link_position)
        if self.last_obs[obs_id] is None:
            self.last_obs[obs_id] = msg
        else:
            copy_class_attr(msg, self.last_obs[obs_id], attribute_names)
        # print("Obs_id %d " % obs_id)

    def set_action_type(self, init_cmd_vals, cmd_type_name, act_idx, **kwargs):

        action_id = len(self.action_types)

        if cmd_type_name in ['xbot_position', 'xbot_velocity', 'xbot_effort']:
            act_joint_names = kwargs['act_joint_names']
            cmd_msg = config_xbot_command(act_joint_names, cmd_type_name, init_cmd_vals)
        elif cmd_type_name in ['joint_effort']:
            ros_msg_class = kwargs['ros_msg_class']
            cmd_msg = ros_msg_class()
        else:
            print('OH NOoo')
            raise NotImplementedError("ros_env_interbace command %s has not been implemented!" % cmd_type_name)

        self.action_types.append({'ros_msg': cmd_msg, 'type': cmd_type_name, 'act_idx': act_idx})
        return action_id

    def set_observation_type(self, obs_name, obs_id, obs_type, obs_idx):

        obs_type_id = len(self.obs_types)

        # if obs_type in ['joint_state']:
        #     obs_msg = config_xbot_command(act_joint_names, obs_type, init_cmd_vals)
        # else:
        #     raise NotImplementedError("Only XBOT joint_state observations has been implemented!")

        obs_msg = self.last_obs[obs_id]
        self.obs_types.append({'name': obs_name, 'ros_msg': obs_msg, 'type': obs_type, 'obs_idx': obs_idx})
        # self.obs_types.append([obs_type, obs_msg, obs_idx])
        return obs_type_id

    def set_state_type(self, state_name, obs_type_id, state_type, state_idx):
        state_id = len(self.state_types)

        obs_msg = self.obs_types[obs_type_id]['ros_msg']

        # TODO: Temporal removing obs_msg existence because there is a problem in "optitrack" state
        # if not hasattr(obs_msg, state_type):
        #     raise ValueError("Wrong XBOT field type option")

        state_msg = obs_msg  # TODO: Doing this until we find a better solution to get a 'pointer' to a dict key

        self.state_types.append({'name': state_name, 'ros_msg': state_msg, 'type': state_type, 'state_idx': state_idx})
        return state_id

    def get_total_state_dof(self):
        total_dof = 0
        for state_type in self.state_types:
            total_dof += len(state_type['state_idx'])
        return total_dof

    def get_total_obs_dof(self):
        total_dof = 0
        for obs_type in self.obs_types:
            total_dof += len(obs_type['obs_idx'])
        return total_dof

    def get_total_action_dof(self):
        total_dof = 0
        for action_type in self.action_types:
            total_dof += len(action_type['act_idx'])
        return total_dof

    def run(self):
        """
        Run each ROS publisher in the action_pubs list in a thread, and append them into pub_threads list.
        :return: None
        """
        print("ROSEnvInterface: Start to send commands to Gazebo... ", end='')

        for action_id, (publisher, rate) in enumerate(self.action_pubs):
            self.pub_threads.append(Thread(target=self.publish, args=[publisher, rate, action_id]))
            self.pub_threads[-1].start()

        if not self.action_pubs:
            print("WARNING: No action configured, publishing nothing!!!")
        else:
            print("DONE!")

    def publish(self, publisher, rate, action_idx):
        """
        Function run by each thread in pub_threads list.
        :param publisher: ROS publisher already configured.
        :param rate: Rate in which the ROS message will be published.
        :param action_idx: Index of the ROS message in the action_types list.
        :return: None
        """
        pub_rate = rospy.Rate(rate)
        while not rospy.is_shutdown():
            if self.publish_action:  # Publish only when some send_action was alredy set.
                publisher.publish(self.action_types[action_idx]['ros_msg'])
                self.publish_action = False
                pub_rate.sleep()
            # else:
            #     print("Not sending anything to ROS !!!")

    def reset_config(self):
        """
        Reset to configuration
        :return:
        """
        # Set Model Config
        if self.mode == 'simulation':
            try:
                self.set_config_srv(self.initial_config)
                # self.set_config_srv(model_name=self.initial_config.model_name,
                #                     joint_names=self.initial_config.joint_names,
                #                     joint_positions=self.initial_config.joint_positions)
            except rospy.ServiceException as exc:
                print("/gazebo/set_model_configuration service call failed: %s" % str(exc))

    def reset(self, model_name=None):
        # if self.initial_config is None:
        #     raise AttributeError("Robot initial configuration not defined!")

        # if self.init_acts is None:
        #     raise AttributeError("Robot initial actions not defined!")

        if self.mode == 'simulation':
            print("Resetting in gazebo!")
            rospy.wait_for_service('/gazebo/reset_simulation')
            # try:
            #     self.pause_srv()  # It does not response anything
            # except rospy.ServiceException as exc:
            #     print("/gazebo/pause_physics service call failed: %s" % str(exc))

            try:
                self.reset_srv()
            except rospy.ServiceException as exc:
                print("/gazebo/reset_world service call failed: %s" % str(exc))

            if model_name is not None:
                rospy.wait_for_service('/gazebo/reset_physics_states_model')
                try:
                    self.reset_model_physics_srv(model_name)
                except rospy.ServiceException as exc:
                    print("/gazebo/reset_physics_states_model service call failed: %s" % str(exc))

            # rospy.sleep(0.5)

        # #print("Reset config!")
        # self.reset_config()
        # #print("SLeeping 2 seconds")
        # #rospy.sleep(2)

        # #print("Reset gazebo!")
        # #try:
        # #    self.reset_srv()
        # #except rospy.ServiceException as exc:
        # #    print("/gazebo/reset_world service call failed: %s" % str(exc))

        # #print("SLeeping 2 seconds")
        # #rospy.sleep(2)

        # #print("Reset config!")
        # #self.reset_config()
        # #print("SLeeping 2 MORE seconds")
        # #rospy.sleep(2)

        # #print("Reset gazebo!")
        # #try:
        # #    self.reset_srv()
        # #except rospy.ServiceException as exc:
        # #    print("/gazebo/reset_world service call failed: %s" % str(exc))

        # #try:
        # #    self.unpause_srv()  # It does not response anything
        # #except rospy.ServiceException as exc:
        # #    print("/gazebo/unpause_physics service call failed: %s" % str(exc))

    def get_action_idx(self, name=None, action_type=None):
        """
        Return the index of the action that match the specified name or action_type
        :param name:
        :type name: str
        :param action_type: 
        :type action_type: str
        :return: Index of the observation
        :rtype: int
        """
        if name is not None:
            for ii, action in enumerate(self.action_types):
                if action['name'] == name:
                    return ii
            raise ValueError("There is not action with name %s" % name)

        if action_type is not None:
            for ii, action in enumerate(self.action_types):
                if action['type'] == action_type:
                    return ii
            raise ValueError("There is not action with type %s" % action_type)

        if name is None and action_type is None:
            raise AttributeError("No name or action_type specified")

    def get_obs_idx(self, name=None, obs_type=None):
        """
        Return the index of the observation that match the specified name or obs_type
        :param name:
        :type name: str
        :param obs_type: 
        :type obs_type: str
        :return: Index of the observation
        :rtype: int
        """
        if name is not None:
            for ii, obs in enumerate(self.obs_types):
                if obs['name'] == name:
                    return ii
            raise ValueError("There is not observation with name %s" % name)

        if obs_type is not None:
            for ii, obs in enumerate(self.obs_types):
                if obs['type'] == obs_type:
                    return ii
            raise ValueError("There is not observation with type %s" % obs_type)

        if name is None and obs_type is None:
            raise AttributeError("No name or obs_type specified")

    def get_state_idx(self, name=None, state_type=None):
        """
        Return the index of the state that match the specified name or state_type
        :param name:
        :type name: str
        :param state_type: 
        :type state_type: str
        :return: Index of the observation
        :rtype: int
        """
        if name is not None:
            for ii, state in enumerate(self.state_types):
                if state['name'] == name:
                    return ii
            raise ValueError("There is not state with name %s" % name)

        if state_type is not None:
            for ii, state in enumerate(self.state_types):
                if state['type'] == state_type:
                    return ii
            raise ValueError("There is not state with type %s" % state_type)

        if name is None and state_type is None:
            raise AttributeError("No name or state_type specified")

    def get_action_ros_msg(self, name=None, action_type=None):
        """
        Return the ROS msg corresponding to the name or action type
        :param name:
        :type name: str
        :param action_type:
        :type action_type: str
        :return: ROS message
        """
        return self.action_types[self.get_action_idx(name, action_type)]['ros_msg']

    def get_obs_ros_msg(self, name=None, obs_type=None):
        """
        Return the ROS msg corresponding to the name or observation type.
        :param name:
        :type name: str
        :param obs_type:
        :type obs_type: str
        :return: ROS message
        """
        return self.obs_types[self.get_obs_idx(name, obs_type)]['ros_msg']

    def get_state_ros_msg(self, name=None, state_type=None):
        """
        Return the ROS msg corresponding to the name or state type.
        :param name:
        :type name: str
        :param state_type:
        :type state_type: str
        :return: ROS message
        """
        return self.state_types[self.get_state_idx(name, state_type)]['ros_msg']

    def send_action(self, action):
        """
        Update, from a desired action vector, the ROS messages that will be published,
        and set publish_action flag to True
        :param action: Desired action vector.
        :return: None
        """
        raise NotImplementedError

    def get_observation(self):
        """
        Return defined observations as an array.
        :return:
        """
        raise NotImplementedError

    def get_action_dim(self):
        raise NotImplementedError

    def get_obs_dim(self):
        raise NotImplementedError

    def get_state_dim(self):
        raise NotImplementedError
