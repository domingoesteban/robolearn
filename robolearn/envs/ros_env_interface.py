from __future__ import print_function
from robolearn.envs.environment import EnvInterface
from robolearn.utils.iit.iit_robots_ros import *

# Useful packages
from threading import Thread, Lock
#from multiprocessing import Process, Lock
import abc

# ROS packages
import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import *

from gazebo_robolearn.srv import ResetPhysicsStatesModel

from std_msgs.msg import Float64 as Float64Msg
from sensor_msgs.msg import Imu as ImuMsg
from gazebo_msgs.msg import ContactState as ContactStateMsg
from geometry_msgs.msg import WrenchStamped as WrenchStampedMsg
from controller_manager_msgs.srv import *
from std_srvs.srv import Empty



class ROSEnvInterface(EnvInterface):
    """
    ROSEnvInterface
    """
    def __init__(self, mode='simulation'):
        mode = mode.lower()
        if mode != 'simulation' and mode != 'real':
            raise ValueError("Wrong ROSEnvInterface mode. Options: 'simulation' or 'real'.")

        if mode == 'real':
            raise NotImplementedError("ROSEnvInterface 'real' mode not implemented yet.")

        self.mode = mode

        print("ROSEnvInterface mode: '%s'." % self.mode)

        rospy.init_node('ROSEnvInterface')

        if mode == 'simulation':
            self.reset_srv = rospy.ServiceProxy('/gazebo/reset_world', Empty)
            self.reset_model_physics_srv = rospy.ServiceProxy('/gazebo/reset_physics_states_model', ResetPhysicsStatesModel)
            #self.reset_srv = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
            self.unpause_srv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
            self.pause_srv = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
            self.set_config_srv = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)

        self.last_obs = None
        self.topic_obs_info = []
        self.last_act = None
        self.init_act = 0
        self.init_acts = None

        self.obs_dim = 0
        self.act_dim = 0

        self.initial_config = None

        # ROS Actions
        self.action_pubs = []
        self.pub_threads = []
        self.action_types = []  # Array of dict = {type, cmd_msg, idx from action array)}
        self.last_acts = []  # last_acts should disappear and be integrated with action_types['cmd_msg']

        # ROS Subscribers
        self.observation_subs = []
        self.obs_types = []  # Array of dict = {type, obs_msg, idx from state array)}
        self.last_obs = []
        self.state_types = []  # Array of dict = {type, obs_msg, idx from state array)}

        # Last sensor data
        self.last_joint_state = None

        self.publish_action = False

    def get_observation(self):
        """
        Return defined observations as an array
        :return:
        """
        raise NotImplementedError

    def get_reward(self):
        """
        Return defined observations as an array
        :return:
        """
        raise NotImplementedError

    def set_observation_topic(self, topic_name, topic_type, attribute_names):
        """

        :param topic_name:
        :param topic_type:
        :return:
        """
        obs_id = len(self.observation_subs)
        self.observation_subs.append(rospy.Subscriber(topic_name, topic_type, self.callback_observation, (obs_id, attribute_names)))
        self.last_obs.append(None)
        return obs_id

    def callback_observation(self, msg, params):
        #TODO: Check if it is better to get attribute_names from the obs_types
        """

        :param msg:
        :param obs_id:
        :return:
        """
        obs_id = params[0]
        attribute_names = params[1]
        if not len(self.last_obs):
            raise AttributeError("last_obs has not been configured")
        #if isinstance(msg, WrenchStamped):
        #    print(msg)
        #print(msg.link_position)
        if self.last_obs[obs_id] is None:
            self.last_obs[obs_id] = msg
        else:
            copy_class_attr(msg, self.last_obs[obs_id], attribute_names)


    def set_observation_type(self, obs_name, obs_id, obs_type, obs_idx):

        obs_type_id = len(self.obs_types)

        #if obs_type in ['joint_state']:
        #    obs_msg = config_advr_command(act_joint_names, obs_type, init_cmd_vals)
        #else:
        #    raise NotImplementedError("Only ADVR joint_state observations has been implemented!")

        obs_msg = self.last_obs[obs_id]
        self.obs_types.append({'name': obs_name, 'ros_msg': obs_msg, 'type': obs_type, 'obs_idx': obs_idx})
        #self.obs_types.append([obs_type, obs_msg, obs_idx])
        return obs_type_id

    def get_obs_dim(self):
        """
        :return:
        """
        raise NotImplementedError

    def get_action_dim(self):
        """
        :return:
        """
        raise NotImplementedError

    def set_action_topic(self, topic_name, topic_type, topic_freq):
        """

        :param topic_name:
        :param topic_type:
        :param topic_freq:
        :return:
        """
        action_id = len(self.action_pubs)
        self.action_pubs.append((rospy.Publisher(topic_name, topic_type, queue_size=10), topic_freq))
        return action_id

    def set_action_type(self, init_cmd_vals, cmd_type, act_idx, **kwargs):

        action_id = len(self.action_types)

        act_joint_names = kwargs['act_joint_names']

        if cmd_type in ['position', 'velocity', 'effort']:
            cmd_msg = config_advr_command(act_joint_names, cmd_type, init_cmd_vals)
        else:
            raise NotImplementedError("Only ADVR command has been implemented!")

        self.action_types.append({'ros_msg': cmd_msg, 'type': cmd_type, 'act_idx': act_idx})
        self.last_acts.append(cmd_msg)  # last_acts would be used for the ROS publisher
        return action_id

    def set_state_type(self, state_name, obs_type_id, state_type, state_idx):
        state_id = len(self.state_types)

        obs_msg = self.obs_types[obs_type_id]['ros_msg']

        # TODO: Temporal removing obs_msg existence because there is a problem in "optitrack" state
        #if not hasattr(obs_msg, state_type):
        #    raise ValueError("Wrong ADVR field type option")

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

        :return:
        """
        print("ROSEnvInterface: Start to send commands to Gazebo... ", end='')
        #self.pub_thread.start()

        for action_id, (publisher, rate) in enumerate(self.action_pubs):
            self.pub_threads.append(Thread(target=self.publish, args=[publisher, rate, action_id]))
            self.pub_threads[-1].start()

        if not self.action_pubs:
            print("WARNING: No action configured, publishing nothing!!!")
        else:
            print("DONE!")

    def publish(self, publisher, rate, action_id):
        """

        :param publisher:
        :param rate:
        :param action_id:
        :return:
        """
        pub_rate = rospy.Rate(rate)  # TODO Deactivating constant publishing
        while not rospy.is_shutdown():
            if self.last_acts and self.publish_action:
                #print("Sending to ROS %f" % self.last_acts[action_id])
                publisher.publish(self.last_acts[action_id])
                self.publish_action = False
                pub_rate.sleep()
            #else:
            #    print("Not sending anything to ROS !!!")

    def set_initial_acts(self):
        """
        :param init_acts:
        :return:
        """
        raise NotImplementedError

    def send_action(self, action):
        """
        Responsible to convert action array to array of ROS publish types
        :param self:
        :param action:
        :return:
        """
        raise NotImplementedError

    def set_acts(self, action_array):
        """
        :param action_array: is an array
        :return:
        """
        self.last_acts = action_array

    def reset_config(self):
        """

        :return:
        """
        # Set Model Config
        if self.mode == 'simulation':
            try:
                self.set_config_srv(self.initial_config)
                #self.set_config_srv(model_name=self.initial_config.model_name,
                #                    joint_names=self.initial_config.joint_names,
                #                    joint_positions=self.initial_config.joint_positions)
            except rospy.ServiceException as exc:
                print("/gazebo/set_model_configuration service call failed: %s" % str(exc))

    def reset(self, model_name=None):
        #NotImplementedError

        #if self.initial_config is None:
        #    raise AttributeError("Robot initial configuration not defined!")

        #if self.init_acts is None:
        #    raise AttributeError("Robot initial actions not defined!")

        ## Return commands to initial actions (because the ROS controllers)
        #for ii in range(10):  # Instead time, this should be checked with sensor data
        #    self.last_acts = self.init_acts
        #    self.publish_action = True
        #    rospy.sleep(0.05)

        if self.mode == 'simulation':
            print("Resetting in gazebo!")
            rospy.wait_for_service('/gazebo/reset_simulation')
            #try:
            #    self.pause_srv()  # It does not response anything
            #except rospy.ServiceException as exc:
            #    print("/gazebo/pause_physics service call failed: %s" % str(exc))

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



            #rospy.sleep(0.5)

        ##print("Reset config!")
        #self.reset_config()
        ##print("SLeeping 2 seconds")
        ##rospy.sleep(2)

        ##print("Reset gazebo!")
        ##try:
        ##    self.reset_srv()
        ##except rospy.ServiceException as exc:
        ##    print("/gazebo/reset_world service call failed: %s" % str(exc))

        ##print("SLeeping 2 seconds")
        ##rospy.sleep(2)

        ##print("Reset config!")
        ##self.reset_config()
        ##print("SLeeping 2 MORE seconds")
        ##rospy.sleep(2)

        ##print("Reset gazebo!")
        ##try:
        ##    self.reset_srv()
        ##except rospy.ServiceException as exc:
        ##    print("/gazebo/reset_world service call failed: %s" % str(exc))

        ##try:
        ##    self.unpause_srv()  # It does not response anything
        ##except rospy.ServiceException as exc:
        ##    print("/gazebo/unpause_physics service call failed: %s" % str(exc))



