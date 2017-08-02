#!/usr/bin/env python

#TODO THIS SCRIPT IS NOT FINISH, IT WILL NOT WORK


import sys
import os
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from XCM.msg import JointStateAdvr
from robolearn.utils.robot_model import RobotModel


class Node(object):
    def __init__(self, argv):
        rospy.init_node('fk_node')

        # Robot Model
        robot_urdf = os.environ["ROBOTOLOGY_ROOT"]+'/robots/iit-bigman-ros-pkg/bigman_urdf/urdf/bigman.urdf'
        self.robot_model = RobotModel(robot_urdf)
        self.LH_name = 'LWrMot3'
        self.RH_name = 'RWrMot3'
        l_soft_hand_offset = np.array([0.000, -0.030, -0.210])
        r_soft_hand_offset = np.array([0.000, 0.030, -0.210])
        torso_joints = bigman_params['joint_ids']['TO']
        bigman_righ_left_sign = np.array([1, -1, -1, 1, -1, 1, -1])

        # Topic Publisher
        self.pub_pose = rospy.Publisher('fk_node/pose', Pose, queue_size=10)
        self.pub_twist = rospy.Publisher('fk_node/twist', Twist, queue_size=10)

        # Topic Subscriber
        self.sub = rospy.Subscriber("/xbotcore/bigman/joint_states", JointStateAdvr, self.state_callback)

    def state_callback(self, data):
        # if not joint_ids:
        #     joint_ids[:] = [bigman_params['joints_names'].index(name) for name in data.name]
        self.joint_ids[:] = [bigman_params['joints_names'].index(name) for name in data.name]
        self.joint_pos[joint_ids] = data.link_position
        self.joint_effort[joint_ids] = data.effort
        self.joint_vel[joint_ids] = data.link_velocity
        self.joint_stiffness[joint_ids] = data.stiffness
        self.joint_damping[joint_ids] = data.damping
    subscriber = rospy.Subscriber("/xbotcore/bigman/joint_states", JointStateAdvr, state_callback, (joint_state_id,
                                                                                                    joint_pos_state,
                                                                                                    joint_vel_state,
                                                                                                    joint_effort_state,
                                                                                                    joint_stiffness_state,
                                                                                                    joint_damping_state))


if __name__ == '__main__':
    node = Node(sys.argv)
    rospy.loginfo("FK Node Running...!")
    rospy.spin()
