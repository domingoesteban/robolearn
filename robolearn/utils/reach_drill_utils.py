import numpy as np
import tf
import os
import time
from robolearn.utils.gazebo_utils import *
from robolearn.utils.transformations import homogeneous_matrix
from robolearn.utils.iit.robot_poses.bigman.poses import bigman_pose
from robolearn.utils.iit.iit_robots_params import bigman_params
from robolearn.utils.trajectory_interpolators import polynomial5_interpolation
from robolearn.utils.robot_model import RobotModel
from gazebo_msgs.srv import GetModelState
from robolearn.utils.plot_utils import plot_multi_info
import rospkg

from robolearn.utils.data_logger import DataLogger
from robolearn.utils.transformations import quaternion_inner, compute_cartesian_error
from robolearn.utils.trajectory_interpolators import polynomial5_interpolation, quaternion_slerp_interpolation
from robolearn.utils.sample.sample import Sample
from robolearn.utils.sample.sample_list import SampleList
from robolearn.agents.agent_utils import generate_noise
from robolearn.utils.print_utils import *
import rospy
from XBotCore.msg import JointStateAdvr
import matplotlib.pyplot as plt
import datetime

# Robot Model
robot_urdf = os.environ["ROBOTOLOGY_ROOT"]+'/robots/iit-bigman-ros-pkg/bigman_urdf/urdf/bigman.urdf'
robot_model = RobotModel(robot_urdf)
LH_name = 'LWrMot3'
RH_name = 'RWrMot3'
l_soft_hand_offset = np.array([0.000, -0.030, -0.210])
r_soft_hand_offset = np.array([0.000, 0.030, -0.210])
torso_joints = bigman_params['joint_ids']['TO']
bigman_righ_left_sign = np.array([1, -1, -1, 1, -1, 1, -1])


def create_bigman_drill_condition(q, bigman_drill_pose, state_info, joint_idxs=None):
    if isinstance(q, str):
        if q not in bigman_pose.keys():
            raise ValueError("Pose %s has not been defined in Bigman!" % q)
        q = bigman_pose[q]

#     distance = compute_cartesian_error(self.robot_dyn_model.fk(self.distance_vectors_params[hh]['body_name'],
#                                                                q=q,
#                                                                body_offset=self.distance_vectors_params[hh]['body_offset'],
#                                                                update_kinematics=True,
#                                                                rotation_rep='quat'),
#                                        drill_pose)

    if joint_idxs is not None:
        q = q[joint_idxs]

    condition = np.zeros(sum(state_info['dimensions']))

    joint_state_idx = state_info['idx'][state_info['names'].index('link_position')]
    condition[joint_state_idx] = q

    if 'optitrack' in state_info['names']:
        optitrack_idx = state_info['idx'][state_info['names'].index('optitrack')]
        condition[optitrack_idx] = bigman_drill_pose

    return condition


def create_centauro_drill_condition(q, centauro_drill_pose, state_info, joint_idxs=None):
    if joint_idxs is not None:
        q = q[joint_idxs]

    condition = np.zeros(sum(state_info['dimensions']))

    joint_state_idx = state_info['idx'][state_info['names'].index('link_position')]
    condition[joint_state_idx] = q

    if 'optitrack' in state_info['names']:
        optitrack_idx = state_info['idx'][state_info['names'].index('optitrack')]
        condition[optitrack_idx] = centauro_drill_pose

    return condition


def create_drill_relative_pose(drill_x=0.75,
                               drill_y=0.00,
                               drill_z=0.0184,
                               drill_yaw=0):
    """
    Calculate a drill pose 
    :param drill_x: Box position in axis X (meters) relative to base_link of robot
    :param drill_y: Box position in axis Y (meters) relative to base_link of robot
    :param drill_z: Box position in axis Z (meters) relative to base_link of robot
    :param drill_yaw: Box Yaw (axisZ) orientation (degrees) relative to base_link of robot
    :return:         
    """
    quat_from_matrix = tf.transformations.quaternion_from_matrix
    rotation_matrix = tf.transformations.rotation_matrix
    drill_quat = quat_from_matrix(rotation_matrix(np.deg2rad(drill_yaw),
                                                  [0, 0, 1]))
    #drill_matrix = homogeneous_matrix(rot=drill_orient, pos=drill_position)
    return np.hstack((drill_quat, drill_x, drill_y, drill_z))


def create_hand_relative_pose(drill_pose, hand_x=0.0, hand_y=0.0, hand_z=0.0,
                              hand_yaw=0, hand_pitch=0, hand_roll=0):
    """ Create Hand Operational point relative pose.
    Rotation is first yaw, then pitch, and finally roll
    :param drill_pose: (orient+pos)
    :param hand_x: 
    :param hand_y: 
    :param hand_z: 
    :param hand_roll: 
    :param hand_pitch: 
    :param hand_yaw: 
    :return: (orient+pos)
    """
    rotation_matrix = tf.transformations.rotation_matrix
    drill_matrix = tf.transformations.quaternion_matrix(drill_pose[:4])
    drill_matrix[:3, -1] = drill_pose[4:]

    drill_RH_matrix = homogeneous_matrix(pos=np.array([hand_x, hand_y, hand_z]))

    hand_matrix = drill_matrix.dot(drill_RH_matrix)
    hand_matrix = hand_matrix.dot(rotation_matrix(np.deg2rad(-90),
                                                  [0, 1, 0]))  # Align hand to base
    hand_matrix = hand_matrix.dot(rotation_matrix(np.deg2rad(hand_yaw),
                                                  [1, 0, 0]))
    hand_matrix = hand_matrix.dot(rotation_matrix(np.deg2rad(hand_pitch),
                                                  [0, 1, 0]))
    hand_matrix = hand_matrix.dot(rotation_matrix(np.deg2rad(hand_roll),
                                                  [0, 0, 1]))
    hand_pose = np.zeros(7)
    hand_pose[4:] = tf.transformations.translation_from_matrix(hand_matrix)
    hand_pose[:4] = tf.transformations.quaternion_from_matrix(hand_matrix)
    return hand_pose


def reset_condition_bigman_drill_gazebo(condition, state_info):
    state_name = 'optitrack'
    if state_name in state_info['names']:
        bigman_drill_pose = \
            condition[state_info['idx'][state_info['names'].index(state_name)]]
        reset_robot_drill_gazebo(bigman_drill_pose, drill_size=None)
    else:
        raise TypeError("No state with name '%s' in bigman environment"
                        % state_name)

def reset_condition_centauro_drill_gazebo(condition, state_info):
    state_name = 'optitrack'
    if state_name in state_info['names']:
        centauro_drill_pose = \
            condition[state_info['idx'][state_info['names'].index(state_name)]]
        reset_robot_drill_gazebo(centauro_drill_pose, drill_size=None)
    else:
        raise TypeError("No state with name '%s' in bigman environment"
                        % state_name)


class Reset_condition_bigman_drill_gazebo(object):
    def __init__(self):
        self.bigman_drill_poses = list()

    def add_reset_poses(self, bigman_drill_pose):
        self.bigman_drill_poses.append(bigman_drill_pose)

    def reset(self, condition):
        if self.bigman_drill_poses:
            bigman_drill_pose = self.bigman_drill_poses[condition]
            reset_robot_drill_gazebo(bigman_drill_pose, drill_size=None)
        else:
            raise ValueError('No drill_bigman pose configured in '
                             'Reset_condition function')


class Reset_condition_centauro_drill_gazebo(object):
    def __init__(self):
        self.centauro_drill_poses = list()

    def add_reset_poses(self, centauro_drill_pose):
        self.centauro_drill_poses.append(centauro_drill_pose)

    def reset(self, condition):
        if self.centauro_drill_poses:
            centauro_drill_pose = self.centauro_drill_poses[condition]
            reset_robot_drill_gazebo(centauro_drill_pose, drill_size=None)
        else:
            raise ValueError('No drill_centauro pose configured in '
                             'Reset_condition function')


def reset_robot_drill_gazebo(robot_drill_pose, drill_size=None,
                             robot_gz_name='bigman'):

    #delete_gazebo_model('drill')
    #delete_gazebo_model('drill_support')

    #spawn_drill_gazebo(robot_drill_pose, drill_size=drill_size)
    set_drill_gazebo_pose(robot_drill_pose, drill_size=drill_size, robot_gz_name=robot_gz_name)

    # TODO: Wait a little
    time.sleep(2)


def spawn_drill_gazebo(robot_drill_pose, drill_size=None, other_object=False,
                       robot_gz_name='bigman'):

    # IF there is already a drill do not do anything
    drill_pose = np.zeros(7)
    drill_support_pose = np.zeros(7)
    if drill_size is None:
        drill_size = [0.4, 0.5, 0.3]  # FOR NOW WE ARE FIXING IT

    #TODO: Apparently spawn gazebo is spawning considering the bottom of the drill, then we substract the difference in Z
    #robot_drill_pose[2] -= drill_size[2]/2.

    robot_pose = get_gazebo_model_pose(robot_gz_name, 'map')
    quaternion_matrix = tf.transformations.quaternion_matrix
    robot_matrix = \
        homogeneous_matrix(pos=[robot_pose.position.x,
                                robot_pose.position.y,
                                robot_pose.position.z],
                           rot=quaternion_matrix([robot_pose.orientation.x,
                                                  robot_pose.orientation.y,
                                                  robot_pose.orientation.z,
                                                  robot_pose.orientation.w]))

    robot_drill_pose = \
        homogeneous_matrix(pos=robot_drill_pose[4:],
                           rot=quaternion_matrix(robot_drill_pose[:4]))

    drill_matrix = robot_matrix.dot(robot_drill_pose)
    drill_pose[4:] = tf.transformations.translation_from_matrix(drill_matrix)
    drill_pose[:4] = tf.transformations.quaternion_from_matrix(drill_matrix)

    drill_support_pose[:] = drill_pose[:]
    drill_support_pose[-1] = 0

    rospack = rospkg.RosPack()
    drill_sdf = open(rospack.get_path('robolearn_gazebo_env') +
                     '/models/cordless_drill/model.sdf', 'r').read()
    drill_support_sdf = open(rospack.get_path('robolearn_gazebo_env') +
                             '/models/big_support/model.sdf', 'r').read()

    drill_support_pose[:] = drill_support_pose[[4, 5, 6, 0, 1, 2, 3]]
    drill_pose[:] = drill_pose[[4, 5, 6, 0, 1, 2, 3]]

    # Spawn only if model does not exist
    if get_gazebo_model_pose('drill') is not None:
        return
    else:
        spawn_gazebo_model('drill', drill_sdf, drill_pose)

    if get_gazebo_model_pose('drill_support') is not None:
        return
    else:
        spawn_gazebo_model('drill_support', drill_support_sdf,
                           drill_support_pose)

    if other_object:
        beer_sdf = open(rospack.get_path('robolearn_gazebo_env') +
                        '/models/beer/model.sdf', 'r').read()
        beer_pose = np.zeros(7)
        beer_pose[:] = drill_pose[:]
        beer_pose[1] -= 0.45
        spawn_gazebo_model('beer', beer_sdf, beer_pose)


def set_drill_gazebo_pose(bigman_drill_pose, drill_size=None, robot_gz_name='bigman'):
    drill_pose = np.zeros(7)
    drill_support_pose = np.zeros(7)
    if drill_size is None:
        drill_size = [0.4, 0.5, 0.3]  # FOR NOW WE ARE FIXING IT

    quaternion_matrix = tf.transformations.quaternion_matrix

    ##TODO: Apparently spawn gazebo is spawning considering the bottom of the drill, then we substract the difference in Z
    bigman_drill_pose = bigman_drill_pose.copy()

    # bigman_drill_pose[-1] -= drill_size[2]/2.  # Drill
    # #bigman_drill_pose[-1] -= 0  # Bottle

    bigman_pose = get_gazebo_model_pose(robot_gz_name, 'map')
    bigman_matrix = \
        homogeneous_matrix(pos=[bigman_pose.position.x,
                                bigman_pose.position.y,
                                bigman_pose.position.z],
                           rot=quaternion_matrix([bigman_pose.orientation.x,
                                                  bigman_pose.orientation.y,
                                                  bigman_pose.orientation.z,
                                                  bigman_pose.orientation.w]))

    bigman_drill_pose = \
        homogeneous_matrix(pos=bigman_drill_pose[4:],
                           rot=quaternion_matrix(bigman_drill_pose[:4]))

    drill_matrix = bigman_matrix.dot(bigman_drill_pose)
    drill_pose[4:] = tf.transformations.translation_from_matrix(drill_matrix)
    drill_pose[:4] = tf.transformations.quaternion_from_matrix(drill_matrix)

    drill_support_pose[:] = drill_pose[:]
    drill_support_pose[-1] = 0
    drill_support_pose[:] = drill_support_pose[[4, 5, 6, 0, 1, 2, 3]]
    drill_pose[:] = drill_pose[[4, 5, 6, 0, 1, 2, 3]]
    #set_gazebo_model_pose('drill_support', drill_support_pose)
    set_gazebo_model_pose('drill', drill_pose)


def generate_reach_joints_trajectories(drill_relative_pose, drill_size, T,
                                       q_init, option=0, dt=1, arm='RA'):
    # reach_option 0: IK desired final pose, interpolate in joint space
    # reach_option 1: Trajectory in EEs, then IK whole trajectory
    # reach_option 2: Trajectory in EEs, IK with Jacobians

    ik_method = 'optimization'  # iterative / optimization
    if arm in ['LA', 'BA']:
        LH_reach_pose = \
            create_hand_relative_pose(drill_relative_pose,
                                      hand_x=0, hand_y=drill_size[1]/2-0.02,
                                      hand_z=0, hand_yaw=0)
    if arm in ['RA', 'BA']:
        RH_reach_pose = \
            create_hand_relative_pose(drill_relative_pose,
                                      hand_x=0, hand_y=-drill_size[1]/2+0.02,
                                      hand_z=0, hand_yaw=0)

    N = int(np.ceil(T/dt))

    # # Swap position, orientation
    # LH_reach_pose = np.concatenate((LH_reach_pose[3:], LH_reach_pose[:3]))
    # RH_reach_pose = np.concatenate((RH_reach_pose[3:], RH_reach_pose[:3]))

    # ######### #
    # Reach Box #
    # ######### #
    if option == 0:
        q_reach = robot_model.ik(LH_name, LH_reach_pose,
                                 body_offset=l_soft_hand_offset,
                                 mask_joints=torso_joints,
                                 joints_limits=bigman_params['joints_limits'],
                                 method=ik_method)
        q_reachRA = robot_model.ik(RH_name, RH_reach_pose,
                                   body_offset=r_soft_hand_offset,
                                   mask_joints=torso_joints,
                                   joints_limits=bigman_params['joints_limits'],
                                   method=ik_method)

        q_reach[bigman_params['joint_ids']['RA']] = \
            q_reachRA[bigman_params['joint_ids']['RA']]

        # Trajectory
        reach_qs, reach_q_dots, reach_q_ddots = \
            polynomial5_interpolation(N, q_reach, q_init)
    else:
        raise ValueError("Wrong reach_option %d" % option)

    reach_q_dots *= 1./dt
    reach_q_ddots *= 1./dt*1./dt

    return reach_qs, reach_q_dots, reach_q_ddots


def task_space_torque_control_demos(**kwargs):

    bigman_env = kwargs['bigman_env']
    drill_relative_pose = kwargs['drill_relative_pose']
    drill_size = kwargs['drill_size']
    Treach = kwargs['Treach']
    Tlift = kwargs['Tlift']
    Tinter = kwargs['Tinter']
    Tend = kwargs['Tend']
    Ts = kwargs['Ts']
    conditions_to_sample = kwargs['conditions_to_sample']
    n_samples = kwargs['n_samples']
    noisy = kwargs['noisy']
    noise_hyperparams = kwargs['noise_hyperparams']
    final_drill_height = kwargs['final_drill_height']

    dir_path = './TASKSPACE_TORQUE_CTRL_DEMO_'+str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    data_logger = DataLogger(dir_path)

    # ROS robot-state
    joint_pos_state = np.zeros(robot_model.qdot_size)  # Assuming joint state only gives actuated joints state
    joint_vel_state = np.zeros(robot_model.qdot_size)
    joint_effort_state = np.zeros(robot_model.qdot_size)
    joint_stiffness_state = np.zeros(robot_model.qdot_size)
    joint_damping_state = np.zeros(robot_model.qdot_size)
    joint_state_id = []

    def state_callback(data, params):
        joint_ids = params[0]
        joint_pos = params[1]
        joint_vel = params[2]
        joint_effort = params[3]
        joint_stiffness = params[4]
        joint_damping = params[5]
        # if not joint_ids:
        #     joint_ids[:] = [bigman_params['joints_names'].index(name) for name in data.name]
        joint_ids[:] = [bigman_params['joints_names'].index(name) for name in data.name]
        joint_pos[joint_ids] = data.link_position
        joint_effort[joint_ids] = data.effort
        joint_vel[joint_ids] = data.link_velocity
        joint_stiffness[joint_ids] = data.stiffness
        joint_damping[joint_ids] = data.damping
    subscriber = rospy.Subscriber("/xbotcore/bigman/joint_states", JointStateAdvr, state_callback, (joint_state_id,
                                                                                                    joint_pos_state,
                                                                                                    joint_vel_state,
                                                                                                    joint_effort_state,
                                                                                                    joint_stiffness_state,
                                                                                                    joint_damping_state))

    Nreach = int(Treach/Ts)*1
    Ninter = int(Tinter/Ts)*1
    Ninter_acum = Nreach + Ninter
    Nlift = int(Tlift/Ts)*1
    Nlift_acum = Ninter_acum + Nlift
    Nend = int(Tend/Ts)*1
    Ntotal = Nreach + Ninter + Nlift + Nend
    dU = bigman_env.get_action_dim()
    dX = bigman_env.get_state_dim()
    dO = bigman_env.get_obs_dim()
    J_left = np.zeros((6, robot_model.qdot_size))
    J_right = np.zeros((6, robot_model.qdot_size))
    M_left = np.zeros((robot_model.qdot_size, robot_model.qdot_size))
    M_right = np.zeros((robot_model.qdot_size, robot_model.qdot_size))
    M_left_bar = np.zeros((6, 6))
    M_right_bar = np.zeros((6, 6))
    g = np.zeros(robot_model.qdot_size)

    default_joint_stiffness = np.array([8000.,  5000.,  8000.,  5000.,  5000.,  2000.,
                                        8000.,  5000.,  5000.,  5000.,  5000.,  2000.,
                                        5000.,  8000.,  5000.,
                                        5000.,  8000.,  5000.,  5000.,   300.,  2000.,   300.,
                                        300.,   300.,
                                        5000.,  8000.,  5000.,  5000.,   300.,  2000.,   300.])
    default_joint_damping = np.array([30.,  50.,  30.,  30.,  30.,   5.,
                                      30.,  50.,  30.,  30.,  30.,   5.,
                                      30.,  50.,  30.,
                                      30.,  50.,  30.,  30.,   1.,   5.,   1.,
                                      1.,   1.,
                                      30.,  50.,  30.,  30.,   1.,   5.,   1.])
    Kp_tau = np.eye(robot_model.q_size)*default_joint_stiffness/100
    Kd_tau = np.eye(robot_model.qdot_size)*default_joint_damping/10
    Kd_q = Kd_tau
    Kp_null = np.eye(robot_model.qdot_size)*0.6
    K_ori = np.tile(50, 3)#*0.1
    K_pos = np.tile(20, 3)#*0.1
    Kp_task = np.eye(6)*np.r_[K_ori, K_pos]
    Kd_task = np.sqrt(Kp_task)

    # left_hand_base_pose = create_hand_relative_pose(drill_relative_pose,
    #                                                 hand_x=0.02, hand_y=drill_size[1]/2-0.08, hand_z=-0.02, hand_yaw=0)
    # right_hand_base_pose = create_hand_relative_pose(drill_relative_pose,
    #                                                  hand_x=0.02, hand_y=-drill_size[1]/2+0.08, hand_z=-0.02, hand_yaw=0)
    left_hand_base_pose = create_hand_relative_pose(drill_relative_pose,
                                                    hand_x=0.0, hand_y=drill_size[1]/2-0.02, hand_z=-0.0, hand_yaw=0)
    right_hand_base_pose = create_hand_relative_pose(drill_relative_pose,
                                                     hand_x=0.0, hand_y=-drill_size[1]/2+0.02, hand_z=-0.0, hand_yaw=0)
    left_hand_base_pose_lift = left_hand_base_pose.copy()
    left_hand_base_pose_lift[-1] += final_drill_height
    right_hand_base_pose_lift = right_hand_base_pose.copy()
    right_hand_base_pose_lift[-1] += final_drill_height

    #final_positions = self.get_q_from_condition(.conditions[cond])

    demos_samples = [list() for _ in range(len(conditions_to_sample))]

    for cond_id in range(len(conditions_to_sample)):
        cond = conditions_to_sample[cond_id]

        i = 0
        while i < n_samples:
            left_task_space_traj = np.zeros((Ntotal, 7))
            left_task_space_traj_dots = np.zeros((Ntotal, 6))
            left_task_space_traj_ddots = np.zeros((Ntotal, 6))
            right_task_space_traj = np.zeros((Ntotal, 7))
            right_task_space_traj_dots = np.zeros((Ntotal, 6))
            right_task_space_traj_ddots = np.zeros((Ntotal, 6))
            joint_traj = np.zeros((Ntotal, robot_model.q_size))
            joint_traj_dots = np.zeros((Ntotal, robot_model.qdot_size))
            joint_traj_ddots = np.zeros((Ntotal, robot_model.qdot_size))
            real_left_task_space_traj = np.zeros_like(left_task_space_traj)
            real_right_task_space_traj = np.zeros_like(right_task_space_traj)
            real_left_task_space_traj_dots = np.zeros_like(left_task_space_traj_dots)
            real_right_task_space_traj_dots = np.zeros_like(right_task_space_traj_dots)

            # Interpolation
            interpolation_type = 1
            if interpolation_type == 0:
                q_init = joint_pos_state.copy()
                init_left_hand_pose = robot_model.fk(LH_name, q=q_init, body_offset=l_soft_hand_offset,
                                                     update_kinematics=True, rotation_rep='quat')
                init_right_hand_pose = robot_model.fk(RH_name, q=q_init, body_offset=r_soft_hand_offset,
                                                      update_kinematics=True, rotation_rep='quat')
                # Interpolation type 0: First task_space interp, then joint_space
                # ---------------------------------------------------------------
                print('Create task_space trajectory...')
                left_task_space_traj[:, 4:], left_task_space_traj_dots[:, 3:], left_task_space_traj_ddots[:, 3:] = \
                    polynomial5_interpolation(Nreach, left_hand_base_pose[4:], init_left_hand_pose[4:])
                left_task_space_traj[:, :4], left_task_space_traj_dots[:, :3], left_task_space_traj_ddots[:, :3] = \
                    quaternion_slerp_interpolation(Nreach, left_hand_base_pose[:4], init_left_hand_pose[:4])
                left_task_space_traj_dots *= 1./Ts
                left_task_space_traj_ddots *= (1./Ts)**2

                right_task_space_traj[:, 4:], right_task_space_traj_dots[:, 3:], right_task_space_traj_ddots[:, 3:] = \
                    polynomial5_interpolation(Nreach, right_hand_base_pose[4:], init_right_hand_pose[4:])
                right_task_space_traj[:, :4], right_task_space_traj_dots[:, :3], right_task_space_traj_ddots[:, :3] = \
                    quaternion_slerp_interpolation(Nreach, right_hand_base_pose[:4], init_right_hand_pose[:4])
                right_task_space_traj_dots *= 1./Ts
                right_task_space_traj_ddots *= 1./Ts**2

                print('Create joint_space trajectory...')
                joint_traj[0, :] = q_init
                J_left = np.zeros((6, robot_model.qdot_size))
                J_right = np.zeros((6, robot_model.qdot_size))
                mask_joints = bigman_params['joint_ids']['TO']
                rarm_joints = bigman_params['joint_ids']['RA']
                for ii in range(Nreach-1):
                    # Compute the Jacobian matrix
                    robot_model.update_jacobian(J_left, LH_name, joint_traj[ii, :], l_soft_hand_offset,
                                                update_kinematics=True)
                    robot_model.update_jacobian(J_right, RH_name, joint_traj[ii, :], r_soft_hand_offset,
                                                update_kinematics=True)
                    J_left[:, mask_joints] = 0
                    J_right[:, mask_joints] = 0
                    joint_traj_dots[ii, :] = np.linalg.lstsq(J_left, left_task_space_traj_dots[ii, :])[0]
                    joint_traj_dots[ii, rarm_joints] = np.linalg.lstsq(J_left, left_task_space_traj_dots[ii, :])[0][rarm_joints]
                    #joint_traj[ii, :] = robot_model.ik(LH_name, left_task_space_traj[ii, :], body_offset=l_soft_hand_offset,
                    #                                   mask_joints=bigman_params['joint_ids']['TO'],
                    #                                   joints_limits=bigman_params['joints_limits'],
                    #                                   #method='iterative',
                    #                                   method='optimization', regularization_parameter=0.1,
                    #                                   q_init=joint_traj[ii-1, :])
                    joint_traj[ii+1, :] = joint_traj[ii, :] + joint_traj_dots[ii, :] * Ts
                #joint_traj_dots = np.vstack((np.diff(joint_traj, axis=0), np.zeros((1, robot_rbdl_model.qdot_size))))
                #joint_traj_dots *= freq
                joint_traj_ddots = np.vstack((np.diff(joint_traj_dots, axis=0), np.zeros((1, robot_model.qdot_size))))
                joint_traj_ddots *= 1./Ts

            elif interpolation_type == 1:
                q_init = joint_pos_state.copy()
                arms_init = bigman_env.get_conditions(cond)[:14]
                q_init[bigman_params['joint_ids']['BA']] = arms_init

                q_reach = robot_model.ik(LH_name, left_hand_base_pose, body_offset=l_soft_hand_offset,
                                         mask_joints=bigman_params['joint_ids']['TO'],
                                         joints_limits=bigman_params['joints_limits'], method='optimization')
                # Get configuration only for Right Arm
                q_reach[bigman_params['joint_ids']['RA']] = robot_model.ik(RH_name, right_hand_base_pose,
                                                                           body_offset=r_soft_hand_offset,
                                                                           mask_joints=bigman_params['joint_ids']['TO'],
                                                                           joints_limits=bigman_params['joints_limits'],
                                                                           method='optimization')[bigman_params['joint_ids']['RA']]

                # Interpolation type 1: First joint_space interp, then task_space
                # ---------------------------------------------------------------
                print('Create joint_space REACH trajectory...')
                joint_traj[:Nreach, :], joint_traj_dots[:Nreach, :], joint_traj_ddots[:Nreach, :] = polynomial5_interpolation(Nreach,
                                                                                                            q_reach,
                                                                                                            q_init)
                joint_traj_dots[:Nreach, :] *= 1./Ts
                joint_traj_ddots[:Nreach, :] *= (1./Ts) * (1./Ts)

                print('Create task_space REACH trajectory...')
                J_left = np.zeros((6, robot_model.qdot_size))
                J_right = np.zeros((6, robot_model.qdot_size))
                mask_joints = bigman_params['joint_ids']['TO']
                for ii in range(Nreach):
                    left_task_space_traj[ii, :] = robot_model.fk(LH_name, q=joint_traj[ii, :],
                                                                 body_offset=l_soft_hand_offset, update_kinematics=True,
                                                                 rotation_rep='quat')
                    right_task_space_traj[ii, :] = robot_model.fk(RH_name, q=joint_traj[ii, :],
                                                                  body_offset=r_soft_hand_offset,
                                                                  update_kinematics=True, rotation_rep='quat')
                    if ii > 0:
                        if quaternion_inner(left_task_space_traj[ii, :4], left_task_space_traj[ii-1, :4]) < 0:
                            left_task_space_traj[ii, :4] *= -1
                        if quaternion_inner(right_task_space_traj[ii, :4], right_task_space_traj[ii-1, :4]) < 0:
                            right_task_space_traj[ii, :4] *= -1
                    robot_model.update_jacobian(J_left, LH_name, joint_traj[ii, :], l_soft_hand_offset,
                                                update_kinematics=True)
                    robot_model.update_jacobian(J_right, RH_name, joint_traj[ii, :], r_soft_hand_offset,
                                                update_kinematics=True)
                    J_left[:, mask_joints] = 0
                    J_right[:, mask_joints] = 0
                    left_task_space_traj_dots[ii, :] = J_left.dot(joint_traj_dots[ii, :])
                    right_task_space_traj_dots[ii, :] = J_right.dot(joint_traj_dots[ii, :])
                left_task_space_traj_ddots[:Nreach, :] = np.vstack((np.diff(left_task_space_traj_dots[:Nreach, :], axis=0), np.zeros((1, 6))))
                right_task_space_traj_ddots[:Nreach, :] = np.vstack((np.diff(right_task_space_traj_dots[:Nreach, :], axis=0), np.zeros((1, 6))))
                left_task_space_traj_ddots[:Nreach, :] *= (1./Ts)
                right_task_space_traj_ddots[:Nreach, :] *= (1./Ts)

            if Tinter > 0:
                left_task_space_traj[Nreach:Nreach+Ninter, :] = left_task_space_traj[Nreach-1, :]
                left_task_space_traj_dots[Nreach:Nreach+Ninter, :] = left_task_space_traj_dots[Nreach-1, :]
                left_task_space_traj_ddots[Nreach:Nreach+Ninter, :] = left_task_space_traj_ddots[Nreach-1, :]
                right_task_space_traj[Nreach:Nreach+Ninter, :] = right_task_space_traj[Nreach-1, :]
                right_task_space_traj_dots[Nreach:Nreach+Ninter, :] = right_task_space_traj_dots[Nreach-1, :]
                right_task_space_traj_ddots[Nreach:Nreach+Ninter, :] = right_task_space_traj_ddots[Nreach-1, :]
                joint_traj[Nreach:Nreach+Ninter, :] = joint_traj[Nreach-1, :]
                joint_traj_dots[Nreach:Nreach+Ninter, :] = joint_traj_dots[Nreach-1, :]
                joint_traj_ddots[Nreach:Nreach+Ninter, :] = joint_traj_ddots[Nreach-1, :]

            if Tlift > 0:
                print('Create task_space LIFT trajectory...')
                left_task_space_traj[Ninter_acum:Nlift_acum, 4:], left_task_space_traj_dots[Ninter_acum:Nlift_acum, 3:], left_task_space_traj_ddots[Ninter_acum:Nlift_acum, 3:] = \
                    polynomial5_interpolation(Nlift, left_hand_base_pose_lift[4:], left_hand_base_pose[4:])
                left_task_space_traj[Ninter_acum:Nlift_acum, :4], left_task_space_traj_dots[Ninter_acum:Nlift_acum, :3], left_task_space_traj_ddots[Ninter_acum:Nlift_acum, :3] = \
                    quaternion_slerp_interpolation(Nlift, left_hand_base_pose_lift[:4], left_hand_base_pose[:4])
                left_task_space_traj_dots[Ninter_acum:Nlift_acum, :] *= 1./Ts
                left_task_space_traj_ddots[Ninter_acum:Nlift_acum, :] *= (1./Ts)**2

                right_task_space_traj[Ninter_acum:Nlift_acum, 4:], right_task_space_traj_dots[Ninter_acum:Nlift_acum, 3:], right_task_space_traj_ddots[Ninter_acum:Nlift_acum, 3:] = \
                    polynomial5_interpolation(Nlift, right_hand_base_pose_lift[4:], right_hand_base_pose[4:])
                right_task_space_traj[Ninter_acum:Nlift_acum, :4], right_task_space_traj_dots[Ninter_acum:Nlift_acum, :3], right_task_space_traj_ddots[Ninter_acum:Nlift_acum, :3] = \
                    quaternion_slerp_interpolation(Nlift, right_hand_base_pose_lift[:4], right_hand_base_pose[:4])
                right_task_space_traj_dots[Ninter_acum:Nlift_acum, :] *= 1./Ts
                right_task_space_traj_ddots[Ninter_acum:Nlift_acum, :] *= 1./Ts**2

                print('Create joint_space LIFT trajectory...')
                joint_traj[Ninter_acum:Nlift_acum, :] = joint_traj[Ninter_acum-1, :]
                J_left = np.zeros((6, robot_model.qdot_size))
                J_right = np.zeros((6, robot_model.qdot_size))
                mask_joints = bigman_params['joint_ids']['TO']
                rarm_joints = bigman_params['joint_ids']['RA']
                for ii in range(Ninter_acum, Nlift_acum-1):
                    # Compute the Jacobian matrix
                    robot_model.update_jacobian(J_left, LH_name, joint_traj[ii, :], l_soft_hand_offset,
                                                update_kinematics=True)
                    robot_model.update_jacobian(J_right, RH_name, joint_traj[ii, :], r_soft_hand_offset,
                                                update_kinematics=True)
                    J_left[:, mask_joints] = 0
                    J_right[:, mask_joints] = 0
                    joint_traj_dots[ii, :] = np.linalg.lstsq(J_left, left_task_space_traj_dots[ii, :])[0]
                    joint_traj_dots[ii, rarm_joints] = np.linalg.lstsq(J_left, left_task_space_traj_dots[ii, :])[0][rarm_joints]
                    #joint_traj[ii, :] = robot_model.ik(LH_name, left_task_space_traj[ii, :], body_offset=l_soft_hand_offset,
                    #                                   mask_joints=bigman_params['joint_ids']['TO'],
                    #                                   joints_limits=bigman_params['joints_limits'],
                    #                                   #method='iterative',
                    #                                   method='optimization', regularization_parameter=0.1,
                    #                                   q_init=joint_traj[ii-1, :])
                    joint_traj[ii+1, :] = joint_traj[ii, :] + joint_traj_dots[ii, :] * Ts
                #joint_traj_dots = np.vstack((np.diff(joint_traj, axis=0), np.zeros((1, robot_rbdl_model.qdot_size))))
                #joint_traj_dots *= freq
                joint_traj_ddots[Ninter_acum:Nlift_acum, :] = np.vstack((np.diff(joint_traj_dots[Ninter_acum:Nlift_acum, :], axis=0), np.zeros((1, robot_model.qdot_size))))
                joint_traj_ddots[Ninter_acum:Nlift_acum, :] *= 1./Ts

            if Tend > 0:
                left_task_space_traj[Nlift_acum:, :] = left_task_space_traj[Nlift_acum-1, :]
                left_task_space_traj_dots[Nlift_acum:, :] = left_task_space_traj_dots[Nlift_acum-1, :]
                left_task_space_traj_ddots[Nlift_acum:, :] = left_task_space_traj_ddots[Nlift_acum-1, :]
                right_task_space_traj[Nlift_acum:, :] = right_task_space_traj[Nlift_acum-1, :]
                right_task_space_traj_dots[Nlift_acum:, :] = right_task_space_traj_dots[Nlift_acum-1, :]
                right_task_space_traj_ddots[Nlift_acum:, :] = right_task_space_traj_ddots[Nlift_acum-1, :]
                joint_traj[Nlift_acum:, :] = joint_traj[Nlift_acum-1, :]
                joint_traj_dots[Nlift_acum:, :] = joint_traj_dots[Nlift_acum-1, :]
                joint_traj_ddots[Nlift_acum:, :] = joint_traj_ddots[Nlift_acum-1, :]

            # plt.plot(left_task_space_traj[:, :])
            # plt.plot(joint_traj[:, :])
            # plt.show()

            if noisy:
                noise = generate_noise(Ntotal, dU, noise_hyperparams)
            else:
                noise = np.zeros((Ntotal, dU))

            # Create a sample class
            sample = Sample(bigman_env, Ntotal)
            history = [None] * Ntotal
            obs_hist = [None] * Ntotal

            print("Resetting environment...")
            bigman_env.reset(time=2, cond=cond)

            ros_rate = rospy.Rate(int(1/Ts))  # hz
            # Collect history
            for t in range(Ntotal):
                print("Sample cond:%d | i:%d/%d | t:%d/%d" % (cond, i+1, n_samples, t+1, Ntotal))
                obs = bigman_env.get_observation()
                state = bigman_env.get_state()

                # Get current(sensed) joints values
                current_joint_pos = joint_pos_state.copy()
                current_joint_pos[bigman_params['joint_ids']['BA']] = state[:14]
                current_joint_vel = joint_vel_state.copy()
                current_joint_vel[bigman_params['joint_ids']['BA']] = state[14:28]

                # Update Jacobian(s)
                robot_model.update_jacobian(J_left, LH_name, current_joint_pos, l_soft_hand_offset,
                                            update_kinematics=True)
                robot_model.update_jacobian(J_right, RH_name, current_joint_pos, r_soft_hand_offset,
                                            update_kinematics=True)
                J_left[:, bigman_params['joint_ids']['LB']] = 0
                J_left[:, bigman_params['joint_ids']['TO']] = 0
                J_left[:, bigman_params['joint_ids']['RA']] = 0
                J_right[:, bigman_params['joint_ids']['LB']] = 0
                J_right[:, bigman_params['joint_ids']['TO']] = 0
                J_right[:, bigman_params['joint_ids']['LA']] = 0

                # Update gravity forces
                robot_model.update_gravity_forces(g, current_joint_pos)

                # Get J_dot_q_dot(s)
                J_left_dot_q_dot = robot_model.jdqd(LH_name, q=current_joint_pos, qdot=current_joint_vel,
                                                    body_offset=l_soft_hand_offset, update_kinematics=True)
                J_right_dot_q_dot = robot_model.jdqd(RH_name, q=current_joint_pos, qdot=current_joint_vel,
                                                     body_offset=r_soft_hand_offset, update_kinematics=True)

                # Get current operational point pose(s)
                real_left_task_space_traj[t, :] = robot_model.fk(LH_name, q=current_joint_pos, body_offset=l_soft_hand_offset,
                                                                 update_kinematics=True, rotation_rep='quat')
                real_right_task_space_traj[t, :] = robot_model.fk(RH_name, q=current_joint_pos, body_offset=r_soft_hand_offset,
                                                                  update_kinematics=True, rotation_rep='quat')
                # Check quaternion inversion
                if t > 0:
                    if quaternion_inner(real_left_task_space_traj[t, :4], real_left_task_space_traj[t-1, :4]) < 0:
                        real_left_task_space_traj[t, :4] *= -1
                    if quaternion_inner(real_right_task_space_traj[t, :4], real_right_task_space_traj[t-1, :4]) < 0:
                        real_right_task_space_traj[t, :4] *= -1

                # Calculate current task space velocity(ies)
                real_left_task_space_traj_dots[t, :] = J_left.dot(joint_vel_state)
                real_right_task_space_traj_dots[t, :] = J_right.dot(joint_vel_state)

                # Calculate pose and velocities errors
                task_left_pose_error = compute_cartesian_error(left_task_space_traj[t, :], real_left_task_space_traj[t, :])
                task_right_pose_error = compute_cartesian_error(right_task_space_traj[t, :], real_right_task_space_traj[t, :])
                task_left_vel_error = left_task_space_traj_dots[t, :] - real_left_task_space_traj_dots[t, :]
                task_right_vel_error = right_task_space_traj_dots[t, :] - real_right_task_space_traj_dots[t, :]

                # Reference task-space acceleration(s)
                x_left_ddot_r = left_task_space_traj_ddots[t, :] + Kp_task.dot(task_left_pose_error) + Kd_task.dot(task_left_vel_error)
                x_right_ddot_r = right_task_space_traj_ddots[t, :] + Kp_task.dot(task_right_pose_error) + Kd_task.dot(task_right_vel_error)

                # Update Mass matrix
                robot_model.update_inertia_matrix(M_left, current_joint_pos)
                robot_model.update_inertia_matrix(M_right, current_joint_pos)

                # Nakanishi: Gauss Controller (Operational Space Controller in Khatib (1987))
                M_left_bar[:, :] = np.linalg.inv(J_left.dot(np.linalg.inv(M_left)).dot(J_left.T))
                M_right_bar[:, :] = np.linalg.inv(J_right.dot(np.linalg.inv(M_right)).dot(J_right.T))
                J_left_bar = np.linalg.inv(M_left).dot(J_left.T).dot(M_left_bar)
                J_right_bar = np.linalg.inv(M_right).dot(J_right.T).dot(M_right_bar)
                q_error_left = np.zeros_like(current_joint_pos)
                q_error_right = np.zeros_like(current_joint_pos)
                q0 = joint_traj[t, :]
                q_error_left[bigman_params['joint_ids']['LA']] = (current_joint_pos - q0)[bigman_params['joint_ids']['LA']]
                q_error_right[bigman_params['joint_ids']['RA']] = (current_joint_pos - q0)[bigman_params['joint_ids']['RA']]
                q_grad_left = Kp_null.dot(q_error_left)
                q_grad_right = Kp_null.dot(q_error_right)
                alpha = 1
                torque_null_left = -Kd_q.dot(current_joint_vel)*0 - alpha*q_grad_left
                torque_null_right = -Kd_q.dot(current_joint_vel)*0 - alpha*q_grad_right
                left_projection_null_times_torque_null = (np.eye(robot_model.qdot_size)
                                                          - J_left.T.dot(J_left_bar.T)).dot(torque_null_left)
                right_projection_null_times_torque_null = (np.eye(robot_model.qdot_size)
                                                           - J_right.T.dot(J_right_bar.T)).dot(torque_null_right)

                tau_left = M_left.dot(J_left_bar).dot(x_left_ddot_r - J_left_dot_q_dot*0 + J_left.dot(np.linalg.inv(M_left)).dot(g)*0) \
                           + left_projection_null_times_torque_null
                tau_right = M_right.dot(J_right_bar).dot(x_right_ddot_r - J_right_dot_q_dot*0 + J_right.dot(np.linalg.inv(M_right)).dot(g)*0) \
                            + right_projection_null_times_torque_null
                # Multitask controller
                action = tau_left + tau_right
                action = action[bigman_params['joint_ids']['BA']]
                #if t >= 500:
                #    action = np.zeros_like(action)
                #action = np.zeros_like(action)
                bigman_env.send_action(action)
                obs_hist[t] = (obs, action)
                history[t] = (state, action)

                ros_rate.sleep()

            # Stop environment
            bigman_env.stop()

            all_actions = np.array([hist[1] for hist in history])
            all_states = np.array([hist[0] for hist in history])
            all_obs = np.array([hist[0] for hist in obs_hist])
            sample.set_acts(all_actions)  # Set all actions at the same time
            sample.set_obs(all_obs)  # Set all obs at the same time
            sample.set_states(all_states)  # Set all states at the same time
            sample.set_noise(noise)

            #add_answer = raw_input('Add sample to sample list? (y/n): ')
            add_answer = 'y'
            if add_answer.lower() == 'y':
                demos_samples[cond].append(sample)
                print('The sample was added to the sample list. Now there are %02d sample(s) for condition %02d'
                      % (len(demos_samples[cond]), cond))
                i += 1
            else:
                print('The sample was NOT added to the sample list. Sampling again...')

    sample_lists = [SampleList(samples) for samples in demos_samples]

    name_file = data_logger.pickle('demos_sample_lists.pkl', sample_lists)
    print("#"*30)
    print("Demos sample lists saved in %s" % dir_path+'/'+name_file)
    print("#"*30)

    return sample_lists


def task_space_torque_control_dual_demos(**kwargs):
    ask_to_confirm = True

    active_joints = kwargs['active_joints']
    bigman_env = kwargs['bigman_env']
    drill_relative_poses = kwargs['drill_relative_poses']
    drill_relative_pose_cond_id = kwargs['drill_relative_pose_cond_id']
    drill_size = kwargs['drill_size']
    good_offsets = kwargs['good_offsets']
    bad_offsets = kwargs['bad_offsets']
    good_interms = kwargs['good_interms']
    bad_interms = kwargs['bad_interms']
    Treach = kwargs['Treach']
    Tinterm = kwargs['Tinterm']
    Tlift = kwargs['Tlift']
    Tinter = kwargs['Tinter']
    Tend = kwargs['Tend']
    Ts = kwargs['Ts']
    conditions_to_sample = kwargs['conditions_to_sample']
    n_good_samples = kwargs['n_good_samples']
    n_bad_samples = kwargs['n_bad_samples']
    noisy = kwargs['noisy']
    noise_hyperparams = kwargs['noise_hyperparams']
    final_drill_height = kwargs['final_drill_height']

    dir_path = './DUAL_DEMOS_'+str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    data_logger = DataLogger(dir_path)

    # ROS robot-state
    joint_pos_state = np.zeros(robot_model.qdot_size)  # Assuming joint state only gives actuated joints state
    joint_vel_state = np.zeros(robot_model.qdot_size)
    joint_effort_state = np.zeros(robot_model.qdot_size)
    joint_stiffness_state = np.zeros(robot_model.qdot_size)
    joint_damping_state = np.zeros(robot_model.qdot_size)
    joint_state_id = []

    def state_callback(data, params):
        joint_ids = params[0]
        joint_pos = params[1]
        joint_vel = params[2]
        joint_effort = params[3]
        joint_stiffness = params[4]
        joint_damping = params[5]
        # if not joint_ids:
        #     joint_ids[:] = [bigman_params['joints_names'].index(name) for name in data.name]
        joint_ids[:] = [bigman_params['joints_names'].index(name) for name in data.name]
        joint_pos[joint_ids] = data.link_position
        joint_effort[joint_ids] = data.effort
        joint_vel[joint_ids] = data.link_velocity
        joint_stiffness[joint_ids] = data.stiffness
        joint_damping[joint_ids] = data.damping
    subscriber = rospy.Subscriber("/xbotcore/bigman/joint_states", JointStateAdvr, state_callback, (joint_state_id,
                                                                                                    joint_pos_state,
                                                                                                    joint_vel_state,
                                                                                                    joint_effort_state,
                                                                                                    joint_stiffness_state,
                                                                                                    joint_damping_state))
    # Wait until the state is received
    while not joint_state_id:
        pass

    Ninterm = int(Tinterm/Ts)*1
    Nreach = int(Treach/Ts)*1
    Nreach_acum = Ninterm + Nreach
    Ninter = int(Tinter/Ts)*1
    Ninter_acum = Nreach_acum + Ninter
    Nlift = int(Tlift/Ts)*1
    Nlift_acum = Ninter_acum + Nlift
    Nend = int(Tend/Ts)*1
    Ntotal = Ninterm + Nreach + Ninter + Nlift + Nend
    dU = bigman_env.get_action_dim()
    dX = bigman_env.get_state_dim()
    dO = bigman_env.get_obs_dim()

    g = np.zeros(robot_model.qdot_size)

    if active_joints in ['LA', 'BA']:
        J_left = np.zeros((6, robot_model.qdot_size))
        M_left = np.zeros((robot_model.qdot_size, robot_model.qdot_size))
        M_left_bar = np.zeros((6, 6))
    if active_joints in ['RA', 'BA']:
        J_right = np.zeros((6, robot_model.qdot_size))
        M_right = np.zeros((robot_model.qdot_size, robot_model.qdot_size))
        M_right_bar = np.zeros((6, 6))

    default_joint_stiffness = np.array([8000.,  5000.,  8000.,  5000.,  5000.,  2000.,
                                        8000.,  5000.,  5000.,  5000.,  5000.,  2000.,
                                        5000.,  8000.,  5000.,
                                        5000.,  8000.,  5000.,  5000.,   300.,  2000.,   300.,
                                        300.,   300.,
                                        5000.,  8000.,  5000.,  5000.,   300.,  2000.,   300.])
    default_joint_damping = np.array([30.,  50.,  30.,  30.,  30.,   5.,
                                      30.,  50.,  30.,  30.,  30.,   5.,
                                      30.,  50.,  30.,
                                      30.,  50.,  30.,  30.,   1.,   5.,   1.,
                                      1.,   1.,
                                      30.,  50.,  30.,  30.,   1.,   5.,   1.])
    Kp_tau = np.eye(robot_model.q_size)*default_joint_stiffness/100
    Kd_tau = np.eye(robot_model.qdot_size)*default_joint_damping/10
    Kd_q = Kd_tau
    Kp_null = np.eye(robot_model.qdot_size)*0.6
    #K_ori = np.tile(50, 3)#*0.1  # Prev 07/09/17 14:00 pm
    K_ori = np.tile(100, 3)#*0.1
    #K_pos = np.tile(20, 3)#*0.1   # Prev 07/09/17 14:00 pm
    K_pos = np.tile(40, 3)#*0.1
    Kp_task = np.eye(6)*np.r_[K_ori, K_pos]
    Kd_task = np.sqrt(Kp_task)


    #final_positions = self.get_q_from_condition(.conditions[cond])

    good_demos_samples = [list() for _ in range(len(conditions_to_sample))]
    bad_demos_samples = [list() for _ in range(len(conditions_to_sample))]

    for cond_id in range(len(conditions_to_sample)):
        cond = conditions_to_sample[cond_id]

        i = 0
        hand_interm_loaded = False
        hand_offset_loaded = False
        while i < n_good_samples + n_bad_samples:
            # drill_relative_pose = bigman_env.get_conditions(cond)[drill_relative_pose_cond_id]
            drill_relative_pose = drill_relative_poses[cond_id]

            if not hand_interm_loaded:
                if i < n_good_samples:
                    interm_list = good_interms[i]
                else:
                    interm_list = bad_interms[i-n_good_samples]
                hand_interm_roll = interm_list[0]
                hand_interm_pitch = interm_list[1]
                hand_interm_yaw = interm_list[2]
                hand_interm_x = interm_list[3]
                hand_interm_y = interm_list[4]
                hand_interm_z = interm_list[5]
                hand_interm_loaded = True

            if not hand_offset_loaded:
                if i < n_good_samples:
                    offset_list = good_offsets[i]
                else:
                    offset_list = bad_offsets[i-n_good_samples]
                hand_roll = offset_list[0]
                hand_pitch = offset_list[1]
                hand_yaw = offset_list[2]
                hand_x = offset_list[3]
                hand_y = offset_list[4]
                hand_z = offset_list[5]
                hand_offset_loaded = True

            if i < n_good_samples:
                print('Sampling GOOD sample %02d for condition %02d' % (len(good_demos_samples[cond])+1, cond))
            else:
                print('Sampling BAD sample %02d for condition %02d' % (len(bad_demos_samples[cond])+1, cond))

            print("Current hand interm (R:%.3f, P:%.3f, Y:%.3f, x:%.3f, y:%.3f, z:%.3f)" % (hand_interm_roll,
                                                                                            hand_interm_pitch,
                                                                                            hand_interm_yaw,
                                                                                            hand_interm_x,
                                                                                            hand_interm_y,
                                                                                            hand_interm_z))
            if ask_to_confirm:
                hand_interm_answer = raw_input("For modifying it press R,P,Y,x,y,z. | Any other key to accept: ")
                while hand_interm_answer in ['R', 'P', 'Y', 'x', 'y', 'z']:
                    if hand_interm_answer == 'R':
                        option_text = "ROLL (degree)"
                    elif hand_interm_answer == 'P':
                        option_text = "PITCH (degree)"
                    elif hand_interm_answer == 'Y':
                        option_text = "YAW (degree)"
                    elif hand_interm_answer == 'x':
                        option_text = "X POSITION"
                    elif hand_interm_answer == 'y':
                        option_text = "Y POSITION"
                    else:  # 'z':
                        option_text = "Z position"

                    hand_interm_value = float(raw_input("New %s value: " % option_text))
                    if hand_interm_answer == 'R':
                        hand_interm_roll = hand_interm_value
                    elif hand_interm_answer == 'P':
                        hand_interm_pitch = hand_interm_value
                    elif hand_interm_answer == 'Y':
                        hand_interm_yaw = hand_interm_value
                    elif hand_interm_answer == 'x':
                        hand_interm_x = hand_interm_value
                    elif hand_interm_answer == 'y':
                        hand_interm_y = hand_interm_value
                    else: # 'z':
                        hand_interm_z = hand_interm_value

                    print("Current hand interm (R:%.3f, P:%.3f, Y:%.3f, x:%.3f, y:%.3f, z:%.3f)" % (hand_interm_roll,
                                                                                                    hand_interm_pitch,
                                                                                                    hand_interm_yaw,
                                                                                                    hand_interm_x,
                                                                                                    hand_interm_y,
                                                                                                    hand_interm_z))
                    hand_interm_answer = raw_input("For modifying it press R,P,Y,x,y,z. | Any other key to accept: ")


            print("Current hand offset (R:%.3f, P:%.3f, Y:%.3f, x:%.3f, y:%.3f, z:%.3f)" % (hand_roll,
                                                                                            hand_pitch,
                                                                                            hand_yaw,
                                                                                            hand_x,
                                                                                            hand_y,
                                                                                            hand_z))
            if ask_to_confirm:
                hand_offset_answer = raw_input("For modifying it press R,P,Y,x,y,z. | Any other key to accept: ")
                while hand_offset_answer in ['R', 'P', 'Y', 'x', 'y', 'z']:
                    if hand_offset_answer == 'R':
                        option_text = "ROLL (degree)"
                    elif hand_offset_answer == 'P':
                        option_text = "PITCH (degree)"
                    elif hand_offset_answer == 'Y':
                        option_text = "YAW (degree)"
                    elif hand_offset_answer == 'x':
                        option_text = "X POSITION"
                    elif hand_offset_answer == 'y':
                        option_text = "Y POSITION"
                    else:  # 'z':
                        option_text = "Z position"

                    hand_offset_value = float(raw_input("New %s value: " % option_text))
                    if hand_offset_answer == 'R':
                        hand_roll = hand_offset_value
                    elif hand_offset_answer == 'P':
                        hand_pitch = hand_offset_value
                    elif hand_offset_answer == 'Y':
                        hand_yaw = hand_offset_value
                    elif hand_offset_answer == 'x':
                        hand_x = hand_offset_value
                    elif hand_offset_answer == 'y':
                        hand_y = hand_offset_value
                    else: # 'z':
                        hand_z = hand_offset_value

                    print("Current hand offset (R:%.3f, P:%.3f, Y:%.3f, x:%.3f, y:%.3f, z:%.3f)" % (hand_roll,
                                                                                                    hand_pitch,
                                                                                                    hand_yaw,
                                                                                                    hand_x,
                                                                                                    hand_y,
                                                                                                    hand_z))
                    hand_offset_answer = raw_input("For modifying it press R,P,Y,x,y,z. | Any other key to accept: ")

            if active_joints in ['LA', 'BA']:
                left_hand_base_pose = create_hand_relative_pose(drill_relative_pose,
                                                                hand_x=hand_x, hand_y=hand_y, hand_z=hand_z,
                                                                hand_yaw=hand_yaw)
                left_hand_base_pose_lift = left_hand_base_pose.copy()
                left_hand_base_pose_lift[-1] += final_drill_height
            if active_joints in ['RA', 'BA']:
                right_hand_interm_pose = create_hand_relative_pose(drill_relative_pose,
                                                                   hand_x=hand_interm_x, hand_y=hand_interm_y,
                                                                   hand_z=hand_interm_z, hand_yaw=hand_interm_yaw,
                                                                   hand_pitch=hand_interm_pitch, hand_roll=hand_interm_roll)
                right_hand_base_pose = create_hand_relative_pose(drill_relative_pose,
                                                                 hand_x=hand_x, hand_y=hand_y, hand_z=hand_z,
                                                                 hand_yaw=hand_yaw, hand_pitch=hand_pitch,
                                                                 hand_roll=hand_roll)
                right_hand_base_pose_lift = right_hand_base_pose.copy()
                right_hand_base_pose_lift[-1] += final_drill_height

            if active_joints in ['LA', 'BA']:
                left_task_space_traj = np.zeros((Ntotal, 7))
                left_task_space_traj_dots = np.zeros((Ntotal, 6))
                left_task_space_traj_ddots = np.zeros((Ntotal, 6))
                real_left_task_space_traj = np.zeros_like(left_task_space_traj)
                real_left_task_space_traj_dots = np.zeros_like(left_task_space_traj_dots)

            if active_joints in ['RA', 'BA']:
                right_task_space_traj = np.zeros((Ntotal, 7))
                right_task_space_traj_dots = np.zeros((Ntotal, 6))
                right_task_space_traj_ddots = np.zeros((Ntotal, 6))
                real_right_task_space_traj = np.zeros_like(right_task_space_traj)
                real_right_task_space_traj_dots = np.zeros_like(right_task_space_traj_dots)

            joint_traj = np.zeros((Ntotal, robot_model.q_size))
            joint_traj_dots = np.zeros((Ntotal, robot_model.qdot_size))
            joint_traj_ddots = np.zeros((Ntotal, robot_model.qdot_size))

            if Tinterm > 0:
                # Interpolation
                interpolation_type = 1
                if interpolation_type == 0:
                    raise NotImplementedError("NOT IMPLEMENTED")
                    q_init = joint_pos_state.copy()
                    if active_joints in ['LA', 'BA']:
                        init_left_hand_pose = robot_model.fk(LH_name, q=q_init, body_offset=l_soft_hand_offset,
                                                             update_kinematics=True, rotation_rep='quat')
                    if active_joints in ['RA', 'BA']:
                        init_right_hand_pose = robot_model.fk(RH_name, q=q_init, body_offset=r_soft_hand_offset,
                                                              update_kinematics=True, rotation_rep='quat')

                    # Interpolation type 0: First task_space interp, then joint_space
                    # ---------------------------------------------------------------
                    print('Create task_space trajectory...')
                    if active_joints in ['LA', 'BA']:
                        left_task_space_traj[:, 4:], left_task_space_traj_dots[:, 3:], left_task_space_traj_ddots[:, 3:] = \
                            polynomial5_interpolation(Ninterm, left_hand_base_pose[4:], init_left_hand_pose[4:])
                        left_task_space_traj[:, :4], left_task_space_traj_dots[:, :3], left_task_space_traj_ddots[:, :3] = \
                            quaternion_slerp_interpolation(Ninterm, left_hand_base_pose[:4], init_left_hand_pose[:4])
                        left_task_space_traj_dots *= 1./Ts
                        left_task_space_traj_ddots *= (1./Ts)**2

                    if active_joints in ['RA', 'BA']:
                        right_task_space_traj[:, 4:], right_task_space_traj_dots[:, 3:], right_task_space_traj_ddots[:, 3:] = \
                            polynomial5_interpolation(Ninterm, right_hand_base_pose[4:], init_right_hand_pose[4:])
                        right_task_space_traj[:, :4], right_task_space_traj_dots[:, :3], right_task_space_traj_ddots[:, :3] = \
                            quaternion_slerp_interpolation(Ninterm, right_hand_base_pose[:4], init_right_hand_pose[:4])
                        right_task_space_traj_dots *= 1./Ts
                        right_task_space_traj_ddots *= 1./Ts**2

                    print('Create joint_space trajectory...')
                    joint_traj[0, :] = q_init
                    J_left = np.zeros((6, robot_model.qdot_size))
                    J_right = np.zeros((6, robot_model.qdot_size))
                    mask_joints = bigman_params['joint_ids']['TO']
                    rarm_joints = bigman_params['joint_ids']['RA']
                    for ii in range(Ninterm-1):
                        # Compute the Jacobian matrix
                        robot_model.update_jacobian(J_left, LH_name, joint_traj[ii, :], l_soft_hand_offset,
                                                    update_kinematics=True)
                        robot_model.update_jacobian(J_right, RH_name, joint_traj[ii, :], r_soft_hand_offset,
                                                    update_kinematics=True)
                        J_left[:, mask_joints] = 0
                        J_right[:, mask_joints] = 0
                        joint_traj_dots[ii, :] = np.linalg.lstsq(J_left, left_task_space_traj_dots[ii, :])[0]
                        joint_traj_dots[ii, rarm_joints] = np.linalg.lstsq(J_left, left_task_space_traj_dots[ii, :])[0][rarm_joints]
                        #joint_traj[ii, :] = robot_model.ik(LH_name, left_task_space_traj[ii, :], body_offset=l_soft_hand_offset,
                        #                                   mask_joints=bigman_params['joint_ids']['TO'],
                        #                                   joints_limits=bigman_params['joints_limits'],
                        #                                   #method='iterative',
                        #                                   method='optimization', regularization_parameter=0.1,
                        #                                   q_init=joint_traj[ii-1, :])
                        joint_traj[ii+1, :] = joint_traj[ii, :] + joint_traj_dots[ii, :] * Ts
                    #joint_traj_dots = np.vstack((np.diff(joint_traj, axis=0), np.zeros((1, robot_rbdl_model.qdot_size))))
                    #joint_traj_dots *= freq
                    joint_traj_ddots = np.vstack((np.diff(joint_traj_dots, axis=0), np.zeros((1, robot_model.qdot_size))))
                    joint_traj_ddots *= 1./Ts

                elif interpolation_type == 1:
                    q_init = joint_pos_state.copy()
                    arms_init = bigman_env.get_conditions(cond)[:len(bigman_params['joint_ids'][active_joints])]
                    q_init[bigman_params['joint_ids'][active_joints]] = arms_init

                    if active_joints in ['LA', 'BA']:
                        q_interm = robot_model.ik(LH_name, left_hand_base_pose, body_offset=l_soft_hand_offset,
                                                 mask_joints=bigman_params['joint_ids']['TO'],
                                                 joints_limits=bigman_params['joints_limits'], method='optimization')
                        if active_joints == 'BA':
                            # Get configuration only for Right Arm
                            q_interm[bigman_params['joint_ids']['RA']] = robot_model.ik(RH_name, right_hand_base_pose,
                                                                                       body_offset=r_soft_hand_offset,
                                                                                       mask_joints=bigman_params['joint_ids']['TO'],
                                                                                       joints_limits=bigman_params['joints_limits'],
                                                                                       method='optimization')[bigman_params['joint_ids']['RA']]
                    else:
                        q_interm = robot_model.ik(RH_name, right_hand_interm_pose, body_offset=r_soft_hand_offset,
                                                 mask_joints=bigman_params['joint_ids']['TO']+bigman_params['joint_ids']['LA'],
                                                 joints_limits=bigman_params['joints_limits'], method='optimization',
                                                 q_init=q_init)

                    # Interpolation type 1: First joint_space interp, then task_space
                    # ---------------------------------------------------------------
                    print('Create joint_space INTERM trajectory...')
                    joint_traj[:Ninterm, :], joint_traj_dots[:Ninterm, :], joint_traj_ddots[:Ninterm, :] = polynomial5_interpolation(Ninterm,
                                                                                                                                  q_interm,
                                                                                                                                  q_init)
                    joint_traj_dots[:Ninterm, :] *= 1./Ts
                    joint_traj_ddots[:Ninterm, :] *= (1./Ts) * (1./Ts)

                    print('Create task_space INTERM trajectory...')
                    J_left = np.zeros((6, robot_model.qdot_size))
                    J_right = np.zeros((6, robot_model.qdot_size))
                    mask_joints = bigman_params['joint_ids']['TO']
                    for ii in range(Ninterm):
                        if active_joints in ['LA', 'BA']:
                            left_task_space_traj[ii, :] = robot_model.fk(LH_name, q=joint_traj[ii, :],
                                                                         body_offset=l_soft_hand_offset, update_kinematics=True,
                                                                         rotation_rep='quat')
                            if ii > 0:
                                if quaternion_inner(left_task_space_traj[ii, :4], left_task_space_traj[ii-1, :4]) < 0:
                                    left_task_space_traj[ii, :4] *= -1
                            robot_model.update_jacobian(J_left, LH_name, joint_traj[ii, :], l_soft_hand_offset,
                                                        update_kinematics=True)
                            J_left[:, mask_joints] = 0
                            left_task_space_traj_dots[ii, :] = J_left.dot(joint_traj_dots[ii, :])
                        if active_joints in ['RA', 'BA']:
                            right_task_space_traj[ii, :] = robot_model.fk(RH_name, q=joint_traj[ii, :],
                                                                          body_offset=r_soft_hand_offset,
                                                                          update_kinematics=True, rotation_rep='quat')
                            if ii > 0:
                                if quaternion_inner(right_task_space_traj[ii, :4], right_task_space_traj[ii-1, :4]) < 0:
                                    right_task_space_traj[ii, :4] *= -1
                            robot_model.update_jacobian(J_right, RH_name, joint_traj[ii, :], r_soft_hand_offset,
                                                        update_kinematics=True)
                            J_right[:, mask_joints] = 0
                            right_task_space_traj_dots[ii, :] = J_right.dot(joint_traj_dots[ii, :])

                    if active_joints in ['LA', 'BA']:
                        left_task_space_traj_ddots[:Ninterm, :] = np.vstack((np.diff(left_task_space_traj_dots[:Ninterm, :], axis=0), np.zeros((1, 6))))
                        left_task_space_traj_ddots[:Ninterm, :] *= (1./Ts)
                    if active_joints in ['RA', 'BA']:
                        right_task_space_traj_ddots[:Ninterm, :] = np.vstack((np.diff(right_task_space_traj_dots[:Ninterm, :], axis=0), np.zeros((1, 6))))
                        right_task_space_traj_ddots[:Ninterm, :] *= (1./Ts)


            # Interpolation REACH
            interpolation_type = 1
            if interpolation_type == 0:
                if not Tinterm > 0:
                    q_interm = joint_pos_state.copy()

                if active_joints in ['LA', 'BA']:
                    init_left_hand_pose = robot_model.fk(LH_name, q=q_interm, body_offset=l_soft_hand_offset,
                                                         update_kinematics=True, rotation_rep='quat')
                if active_joints in ['RA', 'BA']:
                    init_right_hand_pose = robot_model.fk(RH_name, q=q_interm, body_offset=r_soft_hand_offset,
                                                          update_kinematics=True, rotation_rep='quat')

                # Interpolation type 0: First task_space interp, then joint_space
                # ---------------------------------------------------------------
                print('Create task_space trajectory...')
                if active_joints in ['LA', 'BA']:
                    left_task_space_traj[:, 4:], left_task_space_traj_dots[:, 3:], left_task_space_traj_ddots[:, 3:] = \
                        polynomial5_interpolation(Nreach, left_hand_base_pose[4:], init_left_hand_pose[4:])
                    left_task_space_traj[:, :4], left_task_space_traj_dots[:, :3], left_task_space_traj_ddots[:, :3] = \
                        quaternion_slerp_interpolation(Nreach, left_hand_base_pose[:4], init_left_hand_pose[:4])
                    left_task_space_traj_dots *= 1./Ts
                    left_task_space_traj_ddots *= (1./Ts)**2

                if active_joints in ['RA', 'BA']:
                    right_task_space_traj[:, 4:], right_task_space_traj_dots[:, 3:], right_task_space_traj_ddots[:, 3:] = \
                        polynomial5_interpolation(Nreach, right_hand_base_pose[4:], init_right_hand_pose[4:])
                    right_task_space_traj[:, :4], right_task_space_traj_dots[:, :3], right_task_space_traj_ddots[:, :3] = \
                        quaternion_slerp_interpolation(Nreach, right_hand_base_pose[:4], init_right_hand_pose[:4])
                    right_task_space_traj_dots *= 1./Ts
                    right_task_space_traj_ddots *= 1./Ts**2

                print('Create joint_space trajectory...')
                joint_traj[0, :] = q_interm
                J_left = np.zeros((6, robot_model.qdot_size))
                J_right = np.zeros((6, robot_model.qdot_size))
                mask_joints = bigman_params['joint_ids']['TO']
                rarm_joints = bigman_params['joint_ids']['RA']
                for ii in range(Nreach-1):
                    # Compute the Jacobian matrix
                    robot_model.update_jacobian(J_left, LH_name, joint_traj[ii, :], l_soft_hand_offset,
                                                update_kinematics=True)
                    robot_model.update_jacobian(J_right, RH_name, joint_traj[ii, :], r_soft_hand_offset,
                                                update_kinematics=True)
                    J_left[:, mask_joints] = 0
                    J_right[:, mask_joints] = 0
                    joint_traj_dots[ii, :] = np.linalg.lstsq(J_left, left_task_space_traj_dots[ii, :])[0]
                    joint_traj_dots[ii, rarm_joints] = np.linalg.lstsq(J_left, left_task_space_traj_dots[ii, :])[0][rarm_joints]
                    #joint_traj[ii, :] = robot_model.ik(LH_name, left_task_space_traj[ii, :], body_offset=l_soft_hand_offset,
                    #                                   mask_joints=bigman_params['joint_ids']['TO'],
                    #                                   joints_limits=bigman_params['joints_limits'],
                    #                                   #method='iterative',
                    #                                   method='optimization', regularization_parameter=0.1,
                    #                                   q_init=joint_traj[ii-1, :])
                    joint_traj[ii+1, :] = joint_traj[ii, :] + joint_traj_dots[ii, :] * Ts
                #joint_traj_dots = np.vstack((np.diff(joint_traj, axis=0), np.zeros((1, robot_rbdl_model.qdot_size))))
                #joint_traj_dots *= freq
                joint_traj_ddots = np.vstack((np.diff(joint_traj_dots, axis=0), np.zeros((1, robot_model.qdot_size))))
                joint_traj_ddots *= 1./Ts

            elif interpolation_type == 1:
                if not Tinterm > 0:
                    q_interm = joint_pos_state.copy()
                    #TODO: Assuming joint state first
                    arms_interm = bigman_env.get_conditions(cond)[:len(bigman_params['joint_ids'][active_joints])]
                    q_interm[bigman_params['joint_ids'][active_joints]] = arms_interm

                if active_joints in ['LA', 'BA']:
                    q_reach = robot_model.ik(LH_name, left_hand_base_pose, body_offset=l_soft_hand_offset,
                                             mask_joints=bigman_params['joint_ids']['TO'],
                                             joints_limits=bigman_params['joints_limits'], method='optimization')
                    if active_joints == 'BA':
                        # Get configuration only for Right Arm
                        q_reach[bigman_params['joint_ids']['RA']] = robot_model.ik(RH_name, right_hand_base_pose,
                                                                                   body_offset=r_soft_hand_offset,
                                                                                   mask_joints=bigman_params['joint_ids']['TO'],
                                                                                   joints_limits=bigman_params['joints_limits'],
                                                                                   method='optimization')[bigman_params['joint_ids']['RA']]
                else:
                    q_reach = robot_model.ik(RH_name, right_hand_base_pose, body_offset=r_soft_hand_offset,
                                             mask_joints=bigman_params['joint_ids']['TO']+bigman_params['joint_ids']['LA'],
                                             joints_limits=bigman_params['joints_limits'], method='optimization',
                                             q_init=q_interm)

                # Interpolation type 1: First joint_space interp, then task_space
                # ---------------------------------------------------------------
                print('Create joint_space REACH trajectory...')
                joint_traj[Ninterm:Nreach_acum, :], joint_traj_dots[Ninterm:Nreach_acum, :], joint_traj_ddots[Ninterm:Nreach_acum, :] = polynomial5_interpolation(Nreach,
                                                                                                                              q_reach,
                                                                                                                              q_interm)
                joint_traj_dots[Ninterm:Nreach_acum, :] *= 1./Ts
                joint_traj_ddots[Ninterm:Nreach_acum, :] *= (1./Ts) * (1./Ts)

                print('Create task_space REACH trajectory...')
                J_left = np.zeros((6, robot_model.qdot_size))
                J_right = np.zeros((6, robot_model.qdot_size))
                mask_joints = bigman_params['joint_ids']['TO']
                for ii in range(Ninterm, Nreach_acum):
                    if active_joints in ['LA', 'BA']:
                        left_task_space_traj[ii, :] = robot_model.fk(LH_name, q=joint_traj[ii, :],
                                                                     body_offset=l_soft_hand_offset, update_kinematics=True,
                                                                     rotation_rep='quat')
                        if ii > 0:
                            if quaternion_inner(left_task_space_traj[ii, :4], left_task_space_traj[ii-1, :4]) < 0:
                                left_task_space_traj[ii, :4] *= -1
                        robot_model.update_jacobian(J_left, LH_name, joint_traj[ii, :], l_soft_hand_offset,
                                                    update_kinematics=True)
                        J_left[:, mask_joints] = 0
                        left_task_space_traj_dots[ii, :] = J_left.dot(joint_traj_dots[ii, :])
                    if active_joints in ['RA', 'BA']:
                        right_task_space_traj[ii, :] = robot_model.fk(RH_name, q=joint_traj[ii, :],
                                                                      body_offset=r_soft_hand_offset,
                                                                      update_kinematics=True, rotation_rep='quat')
                        if ii > 0:
                            if quaternion_inner(right_task_space_traj[ii, :4], right_task_space_traj[ii-1, :4]) < 0:
                                right_task_space_traj[ii, :4] *= -1
                        robot_model.update_jacobian(J_right, RH_name, joint_traj[ii, :], r_soft_hand_offset,
                                                    update_kinematics=True)
                        J_right[:, mask_joints] = 0
                        right_task_space_traj_dots[ii, :] = J_right.dot(joint_traj_dots[ii, :])

                if active_joints in ['LA', 'BA']:
                    left_task_space_traj_ddots[Ninterm:Nreach_acum, :] = np.vstack((np.diff(left_task_space_traj_dots[Ninterm:Nreach_acum, :], axis=0), np.zeros((1, 6))))
                    left_task_space_traj_ddots[Ninterm:Nreach_acum, :] *= (1./Ts)
                if active_joints in ['RA', 'BA']:
                    right_task_space_traj_ddots[Ninterm:Nreach_acum, :] = np.vstack((np.diff(right_task_space_traj_dots[Ninterm:Nreach_acum, :], axis=0), np.zeros((1, 6))))
                    right_task_space_traj_ddots[Ninterm:Nreach_acum, :] *= (1./Ts)

            if Tinter > 0:
                if active_joints in ['LA', 'BA']:
                    left_task_space_traj[Nreach_acum:Ninter_acum, :] = left_task_space_traj[Nreach_acum-1, :]
                    left_task_space_traj_dots[Nreach_acum:Ninter_acum, :] = left_task_space_traj_dots[Nreach_acum-1, :]
                    left_task_space_traj_ddots[Nreach_acum:Ninter_acum, :] = left_task_space_traj_ddots[Nreach_acum-1, :]
                if active_joints in ['RA', 'BA']:
                    right_task_space_traj[Nreach_acum:Ninter_acum, :] = right_task_space_traj[Nreach_acum-1, :]
                    right_task_space_traj_dots[Nreach_acum:Ninter_acum, :] = right_task_space_traj_dots[Nreach_acum-1, :]
                    right_task_space_traj_ddots[Nreach_acum:Ninter_acum, :] = right_task_space_traj_ddots[Nreach_acum-1, :]
                joint_traj[Nreach_acum:Ninter_acum, :] = joint_traj[Nreach_acum-1, :]
                joint_traj_dots[Nreach_acum:Ninter_acum, :] = joint_traj_dots[Nreach_acum-1, :]
                joint_traj_ddots[Nreach_acum:Ninter_acum, :] = joint_traj_ddots[Nreach_acum-1, :]

            if Tlift > 0:
                print('Create task_space LIFT trajectory...')
                if active_joints in ['LA', 'BA']:
                    left_task_space_traj[Ninter_acum:Nlift_acum, 4:], left_task_space_traj_dots[Ninter_acum:Nlift_acum, 3:], left_task_space_traj_ddots[Ninter_acum:Nlift_acum, 3:] = \
                        polynomial5_interpolation(Nlift, left_hand_base_pose_lift[4:], left_hand_base_pose[4:])
                    left_task_space_traj[Ninter_acum:Nlift_acum, :4], left_task_space_traj_dots[Ninter_acum:Nlift_acum, :3], left_task_space_traj_ddots[Ninter_acum:Nlift_acum, :3] = \
                        quaternion_slerp_interpolation(Nlift, left_hand_base_pose_lift[:4], left_hand_base_pose[:4])
                    left_task_space_traj_dots[Ninter_acum:Nlift_acum, :] *= 1./Ts
                    left_task_space_traj_ddots[Ninter_acum:Nlift_acum, :] *= (1./Ts)**2

                if active_joints in ['RA', 'BA']:
                    right_task_space_traj[Ninter_acum:Nlift_acum, 4:], right_task_space_traj_dots[Ninter_acum:Nlift_acum, 3:], right_task_space_traj_ddots[Ninter_acum:Nlift_acum, 3:] = \
                        polynomial5_interpolation(Nlift, right_hand_base_pose_lift[4:], right_hand_base_pose[4:])
                    right_task_space_traj[Ninter_acum:Nlift_acum, :4], right_task_space_traj_dots[Ninter_acum:Nlift_acum, :3], right_task_space_traj_ddots[Ninter_acum:Nlift_acum, :3] = \
                        quaternion_slerp_interpolation(Nlift, right_hand_base_pose_lift[:4], right_hand_base_pose[:4])
                    right_task_space_traj_dots[Ninter_acum:Nlift_acum, :] *= 1./Ts
                    right_task_space_traj_ddots[Ninter_acum:Nlift_acum, :] *= 1./Ts**2

                print('Create joint_space LIFT trajectory...')
                joint_traj[Ninter_acum:Nlift_acum, :] = joint_traj[Ninter_acum-1, :]
                J_left = np.zeros((6, robot_model.qdot_size))
                J_right = np.zeros((6, robot_model.qdot_size))
                mask_joints = bigman_params['joint_ids']['TO']
                rarm_joints = bigman_params['joint_ids']['RA']
                for ii in range(Ninter_acum, Nlift_acum):
                    # Compute the Jacobian matrix
                    robot_model.update_jacobian(J_left, LH_name, joint_traj[ii, :], l_soft_hand_offset,
                                                update_kinematics=True)
                    robot_model.update_jacobian(J_right, RH_name, joint_traj[ii, :], r_soft_hand_offset,
                                                update_kinematics=True)
                    J_left[:, mask_joints] = 0
                    J_right[:, mask_joints] = 0
                    joint_traj_dots[ii, :] = np.linalg.lstsq(J_left, left_task_space_traj_dots[ii, :])[0]
                    joint_traj_dots[ii, rarm_joints] = np.linalg.lstsq(J_left, left_task_space_traj_dots[ii, :])[0][rarm_joints]
                    #joint_traj[ii, :] = robot_model.ik(LH_name, left_task_space_traj[ii, :], body_offset=l_soft_hand_offset,
                    #                                   mask_joints=bigman_params['joint_ids']['TO'],
                    #                                   joints_limits=bigman_params['joints_limits'],
                    #                                   #method='iterative',
                    #                                   method='optimization', regularization_parameter=0.1,
                    #                                   q_init=joint_traj[ii-1, :])
                    joint_traj[ii+1, :] = joint_traj[ii, :] + joint_traj_dots[ii, :] * Ts
                #joint_traj_dots = np.vstack((np.diff(joint_traj, axis=0), np.zeros((1, robot_rbdl_model.qdot_size))))
                #joint_traj_dots *= freq
                joint_traj_ddots[Ninter_acum:Nlift_acum, :] = np.vstack((np.diff(joint_traj_dots[Ninter_acum:Nlift_acum, :], axis=0), np.zeros((1, robot_model.qdot_size))))
                joint_traj_ddots[Ninter_acum:Nlift_acum, :] *= 1./Ts

            if Tend > 0:
                left_task_space_traj[Nlift_acum:, :] = left_task_space_traj[Nlift_acum-1, :]
                left_task_space_traj_dots[Nlift_acum:, :] = left_task_space_traj_dots[Nlift_acum-1, :]
                left_task_space_traj_ddots[Nlift_acum:, :] = left_task_space_traj_ddots[Nlift_acum-1, :]
                right_task_space_traj[Nlift_acum:, :] = right_task_space_traj[Nlift_acum-1, :]
                right_task_space_traj_dots[Nlift_acum:, :] = right_task_space_traj_dots[Nlift_acum-1, :]
                right_task_space_traj_ddots[Nlift_acum:, :] = right_task_space_traj_ddots[Nlift_acum-1, :]
                joint_traj[Nlift_acum:, :] = joint_traj[Nlift_acum-1, :]
                joint_traj_dots[Nlift_acum:, :] = joint_traj_dots[Nlift_acum-1, :]
                joint_traj_ddots[Nlift_acum:, :] = joint_traj_ddots[Nlift_acum-1, :]

            # plt.plot(right_task_space_traj[:, :])
            # #plt.plot(joint_traj[:, :])
            # plt.show()

            if noisy:
                noise = generate_noise(Ntotal, dU, noise_hyperparams)
            else:
                noise = np.zeros((Ntotal, dU))

            #plot_multi_info([noise], block=True, cols=3, legend=True, labels=['Ruido'])

            # Create a sample class
            sample = Sample(bigman_env, Ntotal)
            history = [None] * Ntotal
            obs_hist = [None] * Ntotal

            print("Resetting environment...")
            bigman_env.reset(time=2, cond=cond)

            ros_rate = rospy.Rate(int(1/Ts))  # hz
            # Collect history
            sampling_bar = ProgressBar(Ntotal, bar_title='Sampling')
            for t in range(Ntotal):
                sampling_bar.update(t)
                # print("Sample cond:%d | i:%d/%d | t:%d/%d" % (cond, i+1, n_good_samples+n_bad_samples, t+1, Ntotal))
                obs = bigman_env.get_observation()
                state = bigman_env.get_state()

                # Get current(sensed) joints values
                pos_end_ids = len(bigman_params['joint_ids'][active_joints])
                vel_end_ids = len(bigman_params['joint_ids'][active_joints])*2
                current_joint_pos = joint_pos_state.copy()
                current_joint_pos[bigman_params['joint_ids'][active_joints]] = state[:pos_end_ids]
                current_joint_vel = joint_vel_state.copy()
                current_joint_vel[bigman_params['joint_ids'][active_joints]] = state[pos_end_ids:vel_end_ids]

                # Update Jacobian(s)
                if active_joints in ['LA', 'BA']:
                    robot_model.update_jacobian(J_left, LH_name, current_joint_pos, l_soft_hand_offset,
                                                update_kinematics=True)
                    J_left[:, bigman_params['joint_ids']['LB']] = 0
                    J_left[:, bigman_params['joint_ids']['TO']] = 0
                    J_left[:, bigman_params['joint_ids']['RA']] = 0
                if active_joints in ['RA', 'BA']:
                    robot_model.update_jacobian(J_right, RH_name, current_joint_pos, r_soft_hand_offset,
                                                update_kinematics=True)
                    J_right[:, bigman_params['joint_ids']['LB']] = 0
                    J_right[:, bigman_params['joint_ids']['TO']] = 0
                    J_right[:, bigman_params['joint_ids']['LA']] = 0

                # Update gravity forces
                robot_model.update_gravity_forces(g, current_joint_pos)

                # Get J_dot_q_dot(s)
                if active_joints in ['LA', 'BA']:
                    J_left_dot_q_dot = robot_model.jdqd(LH_name, q=current_joint_pos, qdot=current_joint_vel,
                                                        body_offset=l_soft_hand_offset, update_kinematics=True)
                if active_joints in ['RA', 'BA']:
                    J_right_dot_q_dot = robot_model.jdqd(RH_name, q=current_joint_pos, qdot=current_joint_vel,
                                                         body_offset=r_soft_hand_offset, update_kinematics=True)

                # Get current operational point pose(s)
                if active_joints in ['LA', 'BA']:
                    real_left_task_space_traj[t, :] = robot_model.fk(LH_name, q=current_joint_pos, body_offset=l_soft_hand_offset,
                                                                     update_kinematics=True, rotation_rep='quat')
                if active_joints in ['RA', 'BA']:
                    real_right_task_space_traj[t, :] = robot_model.fk(RH_name, q=current_joint_pos, body_offset=r_soft_hand_offset,
                                                                      update_kinematics=True, rotation_rep='quat')
                # Check quaternion inversion
                if t > 0:
                    if active_joints in ['LA', 'BA']:
                        if quaternion_inner(real_left_task_space_traj[t, :4], real_left_task_space_traj[t-1, :4]) < 0:
                            real_left_task_space_traj[t, :4] *= -1
                    if active_joints in ['RA', 'BA']:
                        if quaternion_inner(real_right_task_space_traj[t, :4], real_right_task_space_traj[t-1, :4]) < 0:
                            real_right_task_space_traj[t, :4] *= -1

                # Calculate current task space velocity(ies)
                if active_joints in ['LA', 'BA']:
                    real_left_task_space_traj_dots[t, :] = J_left.dot(joint_vel_state)
                if active_joints in ['RA', 'BA']:
                    real_right_task_space_traj_dots[t, :] = J_right.dot(joint_vel_state)

                # Calculate pose and velocities errors
                if active_joints in ['LA', 'BA']:
                    task_left_pose_error = compute_cartesian_error(left_task_space_traj[t, :], real_left_task_space_traj[t, :])
                    task_left_vel_error = left_task_space_traj_dots[t, :] - real_left_task_space_traj_dots[t, :]
                if active_joints in ['RA', 'BA']:
                    task_right_pose_error = compute_cartesian_error(right_task_space_traj[t, :], real_right_task_space_traj[t, :])
                    task_right_vel_error = right_task_space_traj_dots[t, :] - real_right_task_space_traj_dots[t, :]
                    print(task_right_pose_error)

                # Reference task-space acceleration(s)
                if active_joints in ['LA', 'BA']:
                    x_left_ddot_r = left_task_space_traj_ddots[t, :] + Kp_task.dot(task_left_pose_error) + Kd_task.dot(task_left_vel_error)
                if active_joints in ['RA', 'BA']:
                    x_right_ddot_r = right_task_space_traj_ddots[t, :] + Kp_task.dot(task_right_pose_error) + Kd_task.dot(task_right_vel_error)

                # Update Mass matrix
                if active_joints in ['LA', 'BA']:
                    robot_model.update_inertia_matrix(M_left, current_joint_pos)
                if active_joints in ['RA', 'BA']:
                    robot_model.update_inertia_matrix(M_right, current_joint_pos)

                # Nakanishi: Gauss Controller (Operational Space Controller in Khatib (1987))
                if active_joints in ['LA', 'BA']:
                    M_left_bar[:, :] = np.linalg.inv(J_left.dot(np.linalg.inv(M_left)).dot(J_left.T))
                    J_left_bar = np.linalg.inv(M_left).dot(J_left.T).dot(M_left_bar)
                    q_error_left = np.zeros_like(current_joint_pos)
                if active_joints in ['RA', 'BA']:
                    M_right_bar[:, :] = np.linalg.inv(J_right.dot(np.linalg.inv(M_right)).dot(J_right.T))
                    J_right_bar = np.linalg.inv(M_right).dot(J_right.T).dot(M_right_bar)
                    q_error_right = np.zeros_like(current_joint_pos)
                q0 = joint_traj[t, :]
                if active_joints in ['LA', 'BA']:
                    q_error_left[bigman_params['joint_ids']['LA']] = (current_joint_pos - q0)[bigman_params['joint_ids']['LA']]
                    q_grad_left = Kp_null.dot(q_error_left)
                if active_joints in ['RA', 'BA']:
                    q_error_right[bigman_params['joint_ids']['RA']] = (current_joint_pos - q0)[bigman_params['joint_ids']['RA']]
                    q_grad_right = Kp_null.dot(q_error_right)
                alpha = 1
                if active_joints in ['LA', 'BA']:
                    torque_null_left = -Kd_q.dot(current_joint_vel)*0 - alpha*q_grad_left
                    left_projection_null_times_torque_null = (np.eye(robot_model.qdot_size)
                                                              - J_left.T.dot(J_left_bar.T)).dot(torque_null_left)
                if active_joints in ['RA', 'BA']:
                    torque_null_right = -Kd_q.dot(current_joint_vel)*0 - alpha*q_grad_right
                    right_projection_null_times_torque_null = (np.eye(robot_model.qdot_size)
                                                               - J_right.T.dot(J_right_bar.T)).dot(torque_null_right)

                if active_joints in ['LA', 'BA']:
                    tau_left = M_left.dot(J_left_bar).dot(x_left_ddot_r - J_left_dot_q_dot*0 + J_left.dot(np.linalg.inv(M_left)).dot(g)*0) \
                               + left_projection_null_times_torque_null
                if active_joints in ['RA', 'BA']:
                    tau_right = M_right.dot(J_right_bar).dot(x_right_ddot_r - J_right_dot_q_dot*0 + J_right.dot(np.linalg.inv(M_right)).dot(g)*0) \
                                + right_projection_null_times_torque_null
                # Multitask controller
                if active_joints == 'BA':
                    action = tau_left + tau_right
                elif active_joints == 'LA':
                    action = tau_left
                else:
                    action = tau_right
                    #action = joint_traj[t, :]
                action = action[bigman_params['joint_ids'][active_joints]]
                #if t >= 500:
                #    action = np.zeros_like(action)
                #action = np.zeros_like(action)
                #print(state)
                bigman_env.send_action(action)
                obs_hist[t] = (obs, action)
                history[t] = (state, action)

                ros_rate.sleep()

            sampling_bar.end()

            # Stop environment
            bigman_env.stop()

            all_actions = np.array([hist[1] for hist in history])
            all_states = np.array([hist[0] for hist in history])
            all_obs = np.array([hist[0] for hist in obs_hist])
            sample.set_acts(all_actions)  # Set all actions at the same time
            sample.set_obs(all_obs)  # Set all obs at the same time
            sample.set_states(all_states)  # Set all states at the same time
            sample.set_noise(noise)

            if i < n_good_samples:
                if ask_to_confirm:
                    add_answer = raw_input('Add sample to GOOD sample list? (y/n): ')
                else:
                    add_answer = 'y'
                #add_answer = 'y'
                if add_answer.lower() == 'y':
                    good_demos_samples[cond].append(sample)
                    print('The sample WAS ADDED to the GOOD sample list. Now there are %02d sample(s) for condition %02d'
                          % (len(good_demos_samples[cond]), cond))
                    print("-"*30)
                    hand_offset_loaded = False
                    i += 1
                else:
                    print('The sample WAS NOT ADDED to the GOOD sample list. Sampling again...')
                    print("-"*30)
            else:
                if ask_to_confirm:
                    add_answer = raw_input('Add sample to BAD sample list? (y/n): ')
                else:
                    add_answer = 'y'
                #add_answer = 'y'
                if add_answer.lower() == 'y':
                    bad_demos_samples[cond].append(sample)
                    print('The sample WAS ADDED to the BAD sample list. Now there are %02d sample(s) for condition %02d'
                          % (len(bad_demos_samples[cond]), cond))
                    print("-"*30)
                    hand_offset_loaded = False
                    i += 1
                else:
                    print('The sample WAS NOT ADDED to the BAD sample list. Sampling again...')
                    print("-"*30)

            if i == n_good_samples:
                good_sample_lists = [SampleList(samples) for samples in good_demos_samples]
                good_name_file = data_logger.pickle('good_demos_sample_lists.pkl', good_sample_lists)
                print("#"*30)
                print("Good dual demos sample lists saved in %s" % dir_path+'/'+good_name_file)
                print("#"*30)

    bad_sample_lists = [SampleList(samples) for samples in bad_demos_samples]
    bad_name_file = data_logger.pickle('bad_demos_sample_lists.pkl', bad_sample_lists)
    print("#"*30)
    print("Bad dual demos sample lists saved in %s" % dir_path+'/'+bad_name_file)
    print("#"*30)

    if n_good_samples > 0 and n_bad_samples > 0:
        return good_sample_lists, bad_sample_lists
    elif n_good_samples > 0 and n_bad_samples < 0:
        return good_sample_lists
    elif n_good_samples < 0 and n_bad_samples > 0:
        return bad_sample_lists


def load_task_space_torque_control_demos(dir_path):
    data_logger = DataLogger(dir_path)

    return data_logger.unpickle('demos_sample_lists.pkl')


def load_task_space_torque_control_dual_demos(dir_path):
    """
    
    :param dir_path: 
    :return: good_demos, bad_demos
    """
    data_logger = DataLogger(dir_path)

    return data_logger.unpickle('good_demos_sample_lists.pkl'), data_logger.unpickle('bad_demos_sample_lists.pkl')
