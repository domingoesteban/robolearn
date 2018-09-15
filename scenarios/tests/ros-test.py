from __future__ import print_function

# Threading modules
import threading

from robolearn.old_envs.base import EnvInterface

from std_msgs.msg import Float64
from gazebo_msgs.msg import ModelStates

from custom_effort_controllers.msg import CommandArrayStamped
from sensor_msgs.msg import JointState as JointStateMsg
from sensor_msgs.msg import Imu as ImuMsg
from gazebo_msgs.msg import ContactState as ContactStateMsg
from geometry_msgs.msg import WrenchStamped as WrenchStampedMsg

from std_srvs.srv import Empty

import rospy

import subprocess
import os
import signal
import time

ROSInterface_mode = ['simulation', 'real']


class ROSEnvInterface(EnvInterface):

    def __init__(self, mode='simulation'):
        if mode != 'simulation' and mode != 'real':
            raise ValueError("Wrong ROSEnvInterface mode. Options: 'simulation' or 'real'.")

        if mode == 'real':
            raise NotImplementedError("ROSEnvInterface 'real' mode not implemented yet.")

        self.mode = mode

        print("ROSEnvInterface mode: '%s'." % self.mode)

        rospy.init_node('ROSEnvInterface')

        #self.reset_proxy = self.ros_service_proxy('/gazebo/reset_world', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        self.last_obs = None
        self.last_act = None


        # ROS Publishers
        self.group_torque_pub = rospy.Publisher("/bigman/group_position_torque_controller/command", CommandArrayStamped, queue_size=10)

        # ROS Subscribers
        self.subs_joint_state = rospy.Subscriber("/bigman/joint_states", JointStateMsg, self.callback_joint_state)
        self.subs_imu1 = rospy.Subscriber("/bigman/sensor/IMU1", ImuMsg, self.callback_imu1)
        self.subs_imu2 = rospy.Subscriber("/bigman/sensor/IMU2", ImuMsg, self.callback_imu2)
        #self.subs_foot_bumper_l = rospy.Subscriber("/bigman/sensor/bumper/LFoot_bumper", ContactStateMsg, self.callback_foot_bumper_l)
        #self.subs_foot_bumper_r = rospy.Subscriber("/bigman/sensor/bumper/RFoot_bumper", ContactStateMsg, self.callback_foot_bumper_r)
        self.subs_ft_l = rospy.Subscriber("/bigman/sensor/ft_sensor/LAnkle", WrenchStampedMsg, self.callback_ft_l)
        self.subs_ft_r = rospy.Subscriber("/bigman/sensor/ft_sensor/RAnkle", WrenchStampedMsg, self.callback_ft_r)

        # Last sensor data
        self.last_joint_state = None
        self.last_imu1 = None
        self.last_imu2 = None
        self.last_foot_bumper_l = None
        self.last_foot_bumper_r = None
        self.last_ft_l = None
        self.last_ft_r = None


    def run(self):
        # self.
        pass


    def callback_joint_state(self, msg):
        self.last_joint_state = msg
        #print("Receiving joint state")

    def callback_imu1(self, msg):
        self.last_imu1 = msg
        #print("Receiving imu1")

    def callback_imu2(self, msg):
        self.last_imu2 = msg
        #print("Receiving imu2")

    def callback_foot_bumper_l(self, msg):
        self.last_foot_bumper_l = msg
        #print("Receiving left foot bumper")

    def callback_foot_bumper_r(self, msg):
        self.last_foot_bumper_r = msg
        #print("Receiving right foot bumper")

    def callback_ft_l(self, msg):
        self.last_ft_l = msg
        #print("Receiving left FT sensor")

    def callback_ft_r(self, msg):
        self.last_ft_r = msg
        #print("Receiving right FT sensor")


    # def ros_service_proxy(self, srv_name=None, srv_type=None):
    #     if srv_name is None:
    #         raise ValueError("No service name has been specified.")
    #     if srv_type is None:
    #         raise ValueError("No service type has been specified.")
    #
    #     print("Waiting for service '%s'..." % srv_name)
    #     rospy.wait_for_service(srv_name)
    #     return rospy.ServiceProxy(srv_name, srv_type)


    def reset(self):
        print("Resetting gazebo...")
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_proxy() # It does not response anything
            #print("/gazebo/reset_world service response: %s" % str(reset_response))
        except rospy.ServiceException as exc:
            print("/gazebo/reset_world service call failed: %s" % str(exc))



if __name__ == '__main__':
    # Create a ROS EnvInterface
    ros_interface = ROSEnvInterface('simulation')

    while not rospy.is_shutdown():
        rospy.spin()

    #raw_input("Press a key...")
    ros_interface.reset()

    while True:
        print("holaaa")
        time.sleep(1)






    # #process_roscore = subprocess.Popen("roscore")
# roslaunch_file = "/home/domingo/robotlearning-superbuild/catkin_ws/src/bigman/bigman_gazebo/launch/bigman_floating_base_whole_body.launch"
# process_roslaunch = subprocess.Popen(["roslaunch", roslaunch_file])#, preexec_fn=os.setsid)
# raw_input("APRETAR PARA MATAR")
# os.killpg(os.getpgid(process_roslaunch.pid), signal.SIGTERM)



