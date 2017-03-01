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
from sensor_msgs.msg import JointState as JointStateMsg
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
        self.last_act = None
        self.init_act = 0
        self.init_acts = None

        self.init_joint1 = 0
        self.init_joint2 = 0

        self.initial_config = None

        # ROS Actions
        self.action_pubs = []
        self.pub_threads = []
        self.last_acts = []

        # ROS Subscribers
        self.mutex_joint_state = Lock()
        self.subs_joint_state = rospy.Subscriber("/acrobot/joint_states", JointStateMsg, self.callback_joint_state)

        # Last sensor data
        self.last_joint_state = None

        self.resetting = False


    def set_action_topic(self, topic_name, topic_type, topic_freq):
        """

        :param topic_name:
        :param topic_type:
        :param topic_freq:
        :return:
        """
        self.action_pubs.append((rospy.Publisher(topic_name, topic_type, queue_size=10), topic_freq))
        return len(self.action_pubs)-1 #TODO: Find a nicer way to return the element number

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
        pub_rate = rospy.Rate(rate)
        while not rospy.is_shutdown():
            if self.last_acts and not self.resetting:
                #print("Sending to ROS %f" % self.last_acts[action_id])
                publisher.publish(self.last_acts[action_id])
                pub_rate.sleep()
            #else:
            #    print("Not sending anything to ROS !!!")

    def set_initial_acts(self):
        """
        :param init_acts:
        :return:
        """
        raise NotImplementedError

    def set_action(self, action):
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


    def callback_joint_state(self, msg):
        self.last_joint_state = msg
        #print("Receiving joint state")

    def get_joint_state(self):
        return self.last_joint_state



    # def ros_service_proxy(self, srv_name=None, srv_type=None):
    #     if srv_name is None:
    #         raise ValueError("No service name has been specified.")
    #     if srv_type is None:
    #         raise ValueError("No service type has been specified.")
    #
    #     print("Waiting for service '%s'..." % srv_name)
    #     rospy.wait_for_service(srv_name)
    #     return rospy.ServiceProxy(srv_name, srv_type)

    def reset_config(self):
        # Set Model Config
        #print("Setting model config in Gazebo...")
        try:
            self.set_config_srv(self.initial_config)
            #self.set_config_srv(model_name=self.initial_config.model_name,
            #                    joint_names=self.initial_config.joint_names,
            #                    joint_positions=self.initial_config.joint_positions)
        except rospy.ServiceException as exc:
            print("/gazebo/set_model_configuration service call failed: %s" % str(exc))


    def reset(self):
        #self.resetting = True
        # Check
        #rospy.wait_for_service('/gazebo/reset_world')
        rospy.wait_for_service('/gazebo/reset_simulation')
        #rospy.wait_for_service('/gazebo/pause_physics')
        #rospy.wait_for_service('/gazebo/unpause_physics')

        if self.initial_config is None:
            raise AttributeError("Robot initial configuration not defined!")

        if self.init_acts is None:
            raise AttributeError("Robot initial actions not defined!")

        self.last_acts = self.init_acts

        ## Stop Controller
        #print("Stopping controller...")
        #self.switch_controller_srv(start_controllers=[],
        #                           stop_controllers=['joint2_effort_controller'],
        #                           strictness=2)

        ## Unload Controller
        #print("Unloading controller...")
        #self.unload_controller_srv(name='joint2_effort_controller')

        #self.last_act = None

        # Pause physics
        #print("Pausing Gazebo...")
        try:
            self.pause_srv() # It does not response anything
        except rospy.ServiceException as exc:
            print("/gazebo/pause_physics service call failed: %s" % str(exc))

        self.reset_config()

        # Unpause physics
        #print("Unpausing Gazebo...")
        try:
            self.unpause_srv() # It does not response anything
        except rospy.ServiceException as exc:
            print("/gazebo/unpause_physics service call failed: %s" % str(exc))


            #print("Reset physics!!!")
            #self.physics_properties.gravity.z = 0
            #try:
            #    print(self.set_physics_srv(time_step=self.physics_properties.time_step,
            #                               max_update_rate=self.physics_properties.max_update_rate,
            #                               gravity=self.physics_properties.gravity,
            #                               ode_config=self.physics_properties.ode_config))
            #except rospy.ServiceException as exc:
            #    print("/gazebo/set_physics_properties service call failed: %s" % str(exc))

            ## Load Controller
            #print("Loading controller...")
            #self.load_controller_srv(name='joint2_effort_controller')

            #print("Pausing 2 sec...")
            #rospy.sleep(duration=2)
            #print("Unpaused!!!")

            ## Start Controller
            #print("Starting controller...")
            #self.switch_controller_srv(start_controllers=['joint2_effort_controller'],
            #                           stop_controllers=[],
            #                           strictness=2)

            #self.resetting = False

            #print("Pausing 2 sec...")
            #rospy.sleep(duration=2)
            #print("Unpaused!!!")


            ## Pause physics
            #print("Pausing Gazebo...")
            #try:
            #    self.pause_srv() # It does not response anything
            #except rospy.ServiceException as exc:
            #    print("/gazebo/pause_physics service call failed: %s" % str(exc))

            ## Reset states with gazebo reset simulation
            #print("Resetting Gazebo...")
            #try:
            #    self.reset_srv() # It does not response anything
            #except rospy.ServiceException as exc:
            #    print("/gazebo/reset_simulation service call failed: %s" % str(exc))

            ## Unpause physics
            #print("Unpausing Gazebo...")
            #try:
            #    self.unpause_srv() # It does not response anything
            #except rospy.ServiceException as exc:
            #    print("/gazebo/unpause_physics service call failed: %s" % str(exc))


            # Pause some time so the controller can go to init_act
            #print("Pausing 2 sec...")
            #rospy.sleep(duration=2)
            #print("Unpaused!!!")

            #print("Reset physics!!!")
            #self.physics_properties.gravity.z = -9.8
            #try:
            #    print(self.set_physics_srv(time_step=self.physics_properties.time_step,
            #                               max_update_rate=self.physics_properties.max_update_rate,
            #                               gravity=self.physics_properties.gravity,
            #                               ode_config=self.physics_properties.ode_config))
            #except rospy.ServiceException as exc:
            #    print("/gazebo/set_physics_properties service call failed: %s" % str(exc))

            #print("Pausing 2 sec...")
            #rospy.sleep(duration=2)
            #print("Unpaused!!!")

    def get_observation(self, option=[]):
        """
        :param option:
        :return: observations as numpy vector
        """

        if len(option) == 0:
            #print('return all')
            pass

        return self.last_obs


