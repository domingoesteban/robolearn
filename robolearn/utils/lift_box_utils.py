import numpy as np
import tf
import os
from robolearn.utils.gazebo_utils import *
from robolearn.utils.transformations import homogeneous_matrix
from robolearn.utils.iit.robot_poses.bigman.poses import bigman_pose
from robolearn.utils.iit.iit_robots_params import bigman_params
from robolearn.utils.trajectory_interpolators import polynomial5_interpolation
from robolearn.utils.robot_model import RobotModel
from gazebo_msgs.srv import GetModelState
import rospkg

# Robot Model
robot_urdf = os.environ["ROBOTOLOGY_ROOT"]+'/robots/iit-bigman-ros-pkg/bigman_urdf/urdf/bigman.urdf'
robot_model = RobotModel(robot_urdf)
LH_name = 'LWrMot3'
RH_name = 'RWrMot3'
l_soft_hand_offset = np.array([0.000, -0.030, -0.210])
r_soft_hand_offset = np.array([0.000, 0.030, -0.210])
torso_joints = bigman_params['joint_ids']['TO']
bigman_righ_left_sign = np.array([1, -1, -1, 1, -1, 1, -1])


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
    box_quat = tf.transformations.quaternion_from_matrix(tf.transformations.rotation_matrix(np.deg2rad(box_yaw),
                                                                                            [0, 0, 1]))
    #box_matrix = homogeneous_matrix(rot=box_orient, pos=box_position)
    return np.hstack((box_x, box_y, box_z, box_quat))


def create_hand_relative_pose(box_pose, hand_x=0.0, hand_y=0.0, hand_z=0.0, hand_yaw=0):
    """
    Create Hand Operational point relative pose
    :param box_pose: (pos+orient)
    :param hand_x: 
    :param hand_y: 
    :param hand_z: 
    :param hand_yaw: 
    :return: 
    """

    box_matrix = tf.transformations.quaternion_matrix(box_pose[3:])
    box_matrix[:3, -1] = box_pose[:3]

    box_RH_matrix = homogeneous_matrix(pos=np.array([hand_x, hand_y, hand_z]))

    hand_matrix = box_matrix.dot(box_RH_matrix)
    hand_matrix = hand_matrix.dot(tf.transformations.rotation_matrix(np.deg2rad(-90), [0, 1, 0]))
    hand_matrix = hand_matrix.dot(tf.transformations.rotation_matrix(np.deg2rad(hand_yaw), [1, 0, 0]))
    hand_pose = np.zeros(7)
    hand_pose[:3] = tf.transformations.translation_from_matrix(hand_matrix)
    hand_pose[3:] = tf.transformations.quaternion_from_matrix(hand_matrix)
    return hand_pose


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


def generate_reach_joints_trajectories(box_relative_pose, box_size, T, q_init, option=0, dt=1):
    # reach_option 0: IK desired final pose, interpolate in joint space
    # reach_option 1: Trajectory in EEs, then IK whole trajectory
    # reach_option 2: Trajectory in EEs, IK with Jacobians

    ik_method = 'optimization'  # iterative / optimization
    LH_reach_pose = create_hand_relative_pose(box_relative_pose, hand_x=0, hand_y=box_size[1]/2-0.02, hand_z=0, hand_yaw=0)
    RH_reach_pose = create_hand_relative_pose(box_relative_pose, hand_x=0, hand_y=-box_size[1]/2+0.02, hand_z=0, hand_yaw=0)
    N = int(np.ceil(T/dt))

    # Swap position, orientation
    LH_reach_pose = np.concatenate((LH_reach_pose[3:], LH_reach_pose[:3]))
    RH_reach_pose = np.concatenate((RH_reach_pose[3:], RH_reach_pose[:3]))

    # ######### #
    # Reach Box #
    # ######### #
    if option == 0:
        q_reach = robot_model.ik(LH_name, LH_reach_pose, body_offset=l_soft_hand_offset,
                                 mask_joints=torso_joints, joints_limits=bigman_params['joints_limits'],
                                 method=ik_method)
        q_reachRA = robot_model.ik(RH_name, RH_reach_pose, body_offset=r_soft_hand_offset,
                                   mask_joints=torso_joints, joints_limits=bigman_params['joints_limits'],
                                   method=ik_method)

        q_reach[bigman_params['joint_ids']['RA']] = q_reachRA[bigman_params['joint_ids']['RA']]

        # Trajectory
        reach_qs, reach_q_dots, reach_q_ddots = polynomial5_interpolation(N, q_reach, q_init)
    else:
        raise ValueError("Wrong reach_option %d" % option)

    reach_q_dots *= 1./dt
    reach_q_ddots *= 1./dt*1./dt

    return reach_qs, reach_q_dots, reach_q_ddots


def generate_lift_joints_trajectories(box_relative_pose, box_size, T, q_init, option=0, dt=1):
    # ######## #
    # Lift box #
    # ######## #
    if option == 0:
        pass
    else:
        raise ValueError("Wrong lift_option %d" % option)
