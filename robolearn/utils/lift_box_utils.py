import numpy as np
import tf
from robolearn.utils.gazebo_utils import *
from robolearn.utils.transformations import homogeneous_matrix
from robolearn.utils.iit.robot_poses.bigman.poses import bigman_pose
from gazebo_msgs.srv import GetModelState
import rospkg


def create_bigman_box_condition(q, bigman_box_pose, joint_idxs=None):
    if isinstance(q, str):
        if q not in bigman_pose.keys():
            raise ValueError("Pose %s has not been defined in Bigman!" % q)
        q = bigman_pose[q]

    if joint_idxs is not None:
        q = q[joint_idxs]

    return np.hstack((q, np.zeros_like(q), bigman_box_pose))


def create_box_relative_pose(box_x=0.75, box_y=0.00, box_z=0.0184, box_yaw=0):
    """
    Calculate a box pose 
    :param box_x: Box position in axis X (meters) relative to base_link of robot
    :param box_y: Box position in axis Y (meters) relative to base_link of robot
    :param box_z: Box position in axis Z (meters) relative to base_link of robot
    :param box_yaw: Box Yaw (axisZ) orientation (degrees) relative to base_link of robot
    :return:         
    """
    box_quat = tf.transformations.quaternion_from_matrix(tf.transformations.rotation_matrix(np.deg2rad(box_yaw), [0, 0, 1]))
    #box_matrix = homogeneous_matrix(rot=box_orient, pos=box_position)
    return np.hstack((box_x, box_y, box_z, box_quat))


def create_ee_relative_pose(box_pose, ee_x=0.0, ee_y=0.0, ee_z=0.0, ee_yaw=0):

    box_matrix = tf.transformations.quaternion_matrix(box_pose[3:])
    box_matrix[:3, -1] = box_pose[:3]

    box_RH_matrix = homogeneous_matrix(pos=np.array([ee_x, ee_y, ee_z]))

    ee_matrix = box_matrix.dot(box_RH_matrix)
    ee_matrix = ee_matrix.dot(tf.transformations.rotation_matrix(np.deg2rad(-90), [0, 1, 0]))
    ee_matrix = ee_matrix.dot(tf.transformations.rotation_matrix(np.deg2rad(ee_yaw), [1, 0, 0]))
    ee_pose = np.zeros(7)
    ee_pose[:3] = tf.transformations.translation_from_matrix(ee_matrix)
    ee_pose[3:] = tf.transformations.quaternion_from_matrix(ee_matrix)
    return ee_pose


def reset_condition_bigman_box_gazebo(condition, state_info):
    state_name = 'optitrack'
    if state_name in state_info['names']:
        bigman_box_pose = condition[state_info['idx'][state_info['names'].index(state_name)]]
        reset_bigman_box_gazebo(bigman_box_pose, box_size=None)
    else:
        raise TypeError("No state with name '%s' in bigman environment" % state_name)


def reset_bigman_box_gazebo(bigman_box_pose, box_size=None):

    #delete_gazebo_model('box')
    #delete_gazebo_model('box_support')

    spawn_box_gazebo(bigman_box_pose, box_size=box_size)


def spawn_box_gazebo(bigman_box_pose, box_size=None):
    box_pose = np.zeros(7)
    box_support_pose = np.zeros(7)
    if box_size is None:
        box_size = [0.4, 0.5, 0.3]  # FOR NOW WE ARE FIXING IT

    #TODO: Apparently spawn gazebo is spawning considering the bottom of the box, then we substract the difference in Z
    bigman_box_pose[2] -= box_size[2]/2.

    bigman_pose = get_gazebo_model_pose('bigman', 'map')
    bigman_matrix = homogeneous_matrix(pos=[bigman_pose.position.x, bigman_pose.position.x, bigman_pose.position.z],
                                       rot=tf.transformations.quaternion_matrix([bigman_pose.orientation.x,
                                                                                 bigman_pose.orientation.y,
                                                                                 bigman_pose.orientation.z,
                                                                                 bigman_pose.orientation.w]))

    bigman_box_pose = homogeneous_matrix(pos=bigman_box_pose[:3],
                                         rot=tf.transformations.quaternion_matrix(bigman_box_pose[3:]))

    box_matrix = bigman_matrix.dot(bigman_box_pose)
    box_pose[:3] = tf.transformations.translation_from_matrix(box_matrix)
    box_pose[3:] = tf.transformations.quaternion_from_matrix(box_matrix)

    box_support_pose[:] = box_pose[:]
    box_support_pose[2] = 0

    rospack = rospkg.RosPack()
    box_sdf = open(rospack.get_path('robolearn_gazebo_env')+'/models/cardboard_cube_box/model.sdf', 'r').read()
    box_support_sdf = open(rospack.get_path('robolearn_gazebo_env')+'/models/big_support/model.sdf', 'r').read()

    spawn_gazebo_model('box_support', box_support_sdf, box_support_pose)
    spawn_gazebo_model('box', box_sdf, box_pose)
