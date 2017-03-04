from __future__ import print_function
from robolearn.envs.base import EnvInterface

# Useful packages
from threading import Thread, Lock
#from multiprocessing import Process, Lock

# ROS packages
import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import *

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
        if mode != 'simulation' and mode != 'real':
            raise ValueError("Wrong ROSEnvInterface mode. Options: 'simulation' or 'real'.")

        if mode == 'real':
            raise NotImplementedError("ROSEnvInterface 'real' mode not implemented yet.")

        self.mode = mode

        print("ROSEnvInterface mode: '%s'." % self.mode)

        rospy.init_node('ROSEnvInterface')

        self.reset_srv = rospy.ServiceProxy('/gazebo/reset_world', Empty)
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
        self.last_acts = []

        # ROS Subscribers
        self.observation_subs = []
        self.last_obs = []

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

    def set_observation_topic(self, topic_name, topic_type):
        """

        :param topic_name:
        :param topic_type:
        :return:
        """
        obs_id = len(self.observation_subs)
        self.observation_subs.append(rospy.Subscriber(topic_name, topic_type, self.callback_observation, obs_id))
        self.last_obs.append(None)
        return obs_id

    def callback_observation(self, msg, obs_id):
        """

        :param msg:
        :param obs_id:
        :return:
        """
        if not len(self.last_obs):
            raise AttributeError("last_obs has not been configured")
        #print(msg)
        self.last_obs[obs_id] = msg

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
        # pub_rate = rospy.Rate(rate)  # TODO Deactivating constant publishing
        while not rospy.is_shutdown():
            if self.last_acts and self.publish_action:
                #print("Sending to ROS %f" % self.last_acts[action_id])
                publisher.publish(self.last_acts[action_id])
                self.publish_action = False
                #pub_rate.sleep()
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
        try:
            self.set_config_srv(self.initial_config)
            #self.set_config_srv(model_name=self.initial_config.model_name,
            #                    joint_names=self.initial_config.joint_names,
            #                    joint_positions=self.initial_config.joint_positions)
        except rospy.ServiceException as exc:
            print("/gazebo/set_model_configuration service call failed: %s" % str(exc))


    def reset(self):
        rospy.wait_for_service('/gazebo/reset_simulation')

        if self.initial_config is None:
            raise AttributeError("Robot initial configuration not defined!")

        if self.init_acts is None:
            raise AttributeError("Robot initial actions not defined!")

        # Return commands to initial actions (because the ROS controllers)
        for ii in xrange(10):  # Instead time, this should be checked with sensor data
            self.last_acts = self.init_acts
            self.publish_action = True
            rospy.sleep(0.05)

        #try:
        #    self.pause_srv()  # It does not response anything
        #except rospy.ServiceException as exc:
        #    print("/gazebo/pause_physics service call failed: %s" % str(exc))

        #print("Reset gazebo!")
        try:
            self.reset_srv()
        except rospy.ServiceException as exc:
            print("/gazebo/reset_world service call failed: %s" % str(exc))
        rospy.sleep(0.5)

        #print("Reset config!")
        self.reset_config()
        #print("SLeeping 2 seconds")
        #rospy.sleep(2)

        #print("Reset gazebo!")
        #try:
        #    self.reset_srv()
        #except rospy.ServiceException as exc:
        #    print("/gazebo/reset_world service call failed: %s" % str(exc))

        #print("SLeeping 2 seconds")
        #rospy.sleep(2)

        #print("Reset config!")
        #self.reset_config()
        #print("SLeeping 2 MORE seconds")
        #rospy.sleep(2)

        #print("Reset gazebo!")
        #try:
        #    self.reset_srv()
        #except rospy.ServiceException as exc:
        #    print("/gazebo/reset_world service call failed: %s" % str(exc))

        #try:
        #    self.unpause_srv()  # It does not response anything
        #except rospy.ServiceException as exc:
        #    print("/gazebo/unpause_physics service call failed: %s" % str(exc))



