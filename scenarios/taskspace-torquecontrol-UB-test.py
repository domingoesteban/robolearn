import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import rospy
import tf
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import DeleteModel
from geometry_msgs.msg import Pose
from XCM.msg import CommandAdvr
from XCM.msg import JointStateAdvr
import rbdl
from robolearn.utils.trajectory_reproducer import TrajectoryReproducer
from robolearn.utils.iit.iit_robots_params import bigman_params
from robolearn.utils.transformations import *
from robolearn.utils.trajectory_interpolators import polynomial5_interpolation, quaternion_slerp_interpolation
from robolearn.utils.plot_utils import plot_desired_sensed_torque_position
from robolearn.utils.plot_utils import plot_joint_info
from robolearn.utils.plot_utils import plot_desired_sensed_data
from robolearn.utils.plot_utils import plot_joint_multi_info
from robolearn.utils.lift_box_utils import create_box_relative_pose, create_ee_relative_pose

from robolearn.utils.robot_model import RobotModel

np.set_printoptions(precision=4, suppress=True, linewidth=1000)

# Always turn off Gazebo logger
os.system("gz log -d 0")
dir_path = os.path.dirname(os.path.abspath(__file__))
torques_saved_filename = 'torques_init_traj.npy'

# Time
T_init = 5  # Time to move from current position to T_init
T_traj = 10  # Time to execute the trajectory
freq = 100  # Frequency  (1/Ts)

# BOX
box_x = 0.75-0.05
box_y = 0.00
box_z = 0.0184
box_yaw = 0  # Degrees
box_size = [0.4, 0.5, 0.3]
box_relative_pose = create_box_relative_pose(box_x=box_x, box_y=box_y, box_z=box_z, box_yaw=box_yaw)
final_left_hand_pose = create_ee_relative_pose(box_relative_pose, ee_x=0, ee_y=box_size[1]/2-0.02, ee_z=0, ee_yaw=0)
final_left_hand_pose = final_left_hand_pose[[3, 4, 5, 6, 0, 1, 2]]  # First orientation, then position
final_right_hand_pose = create_ee_relative_pose(box_relative_pose, ee_x=0, ee_y=-box_size[1]/2+0.02, ee_z=0, ee_yaw=0)
final_right_hand_pose = final_right_hand_pose[[3, 4, 5, 6, 0, 1, 2]]  # First orientation, then position


# ROBOT MODEL for trying ID
robot_urdf_file = os.environ["ROBOTOLOGY_ROOT"]+'/configs/ADVR_shared/bigman/urdf/bigman.urdf'
robot_rbdl_model = rbdl.loadModel(robot_urdf_file, verbose=False, floating_base=False)
robot_model = RobotModel(robot_urdf_file=robot_urdf_file)
LH_name = 'LWrMot3'
RH_name = 'RWrMot3'
torso_name = 'DWYTorso'
l_soft_hand_offset = np.array([0.000, -0.030, -0.210])
r_soft_hand_offset = np.array([0.000, 0.030, -0.210])
torso_offset = np.array([0.000, 0.000, 0.000])

# Stiffness/Damping gains from Xbot config file
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
Kp_tau = np.eye(robot_rbdl_model.q_size)*default_joint_stiffness/100
Kd_tau = np.eye(robot_rbdl_model.qdot_size)*default_joint_damping/10

# # Joint gains for Joint Space Torque controller
# pd_tau_weights = np.array([0.80,  0.50,  0.80,  0.50,  0.50,  0.20,
#                            0.80,  0.50,  0.50,  0.50,  0.50,  0.20,
#                            0.50,  0.80,  0.50,
#                            0.50,  0.80,  0.50,  0.50,  0.03,  0.20,   0.03,
#                            0.03,  0.03,
#                            0.50,  0.80,  0.50,  0.50,  0.03,  0.20,   0.03])
# Kp_tau = np.eye(robot_rbdl_model.q_size)*(100 * pd_tau_weights)
# Kd_tau = np.eye(robot_rbdl_model.qdot_size)*(2 * pd_tau_weights)


# ROS robot-state
joint_pos_state = np.zeros(robot_rbdl_model.qdot_size)  # Assuming joint state only gives actuated joints state
joint_vel_state = np.zeros(robot_rbdl_model.qdot_size)
joint_effort_state = np.zeros(robot_rbdl_model.qdot_size)
joint_stiffness_state = np.zeros(robot_rbdl_model.qdot_size)
joint_damping_state = np.zeros(robot_rbdl_model.qdot_size)
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

publisher = rospy.Publisher("/xbotcore/bigman/command", CommandAdvr, queue_size=10)
subscriber = rospy.Subscriber("/xbotcore/bigman/joint_states", JointStateAdvr, state_callback, (joint_state_id,
                                                                                                joint_pos_state,
                                                                                                joint_vel_state,
                                                                                                joint_effort_state,
                                                                                                joint_stiffness_state,
                                                                                                joint_damping_state))
rospy.init_node('torquecontrol_example')
pub_rate = rospy.Rate(freq)
des_cmd = CommandAdvr()


# Move ALL joints from current position to INITIAL position in position control mode.
des_cmd.name = bigman_params['joints_names']
q_init = np.zeros(robot_rbdl_model.q_size)
q_init[15] = np.deg2rad(25)
q_init[16] = np.deg2rad(40)
q_init[17] = np.deg2rad(0)
q_init[18] = np.deg2rad(-75)
# ----
q_init[24] = np.deg2rad(25)
q_init[25] = np.deg2rad(-40)
q_init[26] = np.deg2rad(0)
q_init[27] = np.deg2rad(-75)
q_init = np.array([0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,  0.,  0.,  0.,
                   0.,  0.,  0.,
                   #0.0568,  0.2386, -0.2337, -1.6803,  0.2226,  0.0107,  0.5633,
                   0.,  0.,  0.,  -1.5708,  0.,  0., 0.,
                   0.,  0.,
                   #0.0568,  -0.2386, 0.2337, -1.6803,  -0.2226,  0.0107,  -0.5633])
                   0.,  0.,  0.,  -1.5708,  0.,  0., 0.])
N = int(np.ceil(T_init*freq))
joint_init_traj = polynomial5_interpolation(N, q_init, joint_pos_state)[0]
print("Moving to zero configuration with Position control.")
for ii in range(N):
    des_cmd.position = joint_init_traj[ii, :]
    des_cmd.stiffness = default_joint_stiffness
    des_cmd.damping = default_joint_damping
    publisher.publish(des_cmd)
    pub_rate.sleep()

# PAUSE:
print("Sleeping some seconds..")
rospy.sleep(2)

# Move to REACH pose in torque control mode.
# ------------------------------------------
N = int(np.ceil(T_traj*freq))
# joints_to_move = bigman_params['joint_ids']['LA']# + bigman_params['joint_ids']['TO']
joints_to_move = bigman_params['joint_ids']['BA']# + bigman_params['joint_ids']['TO']
# joints_to_move = [bigman_params['joint_ids']['BA'][6]]

# TODO: Temporal, using the current configuration as q_init
q_init = joint_pos_state.copy()
init_left_hand_pose = robot_model.fk(LH_name, q=q_init, body_offset=l_soft_hand_offset, update_kinematics=True,
                                     rotation_rep='quat')
init_right_hand_pose = robot_model.fk(RH_name, q=q_init, body_offset=r_soft_hand_offset, update_kinematics=True,
                                      rotation_rep='quat')

# Preallocate matrices
left_task_space_traj = np.zeros((N, 7))
left_task_space_traj_dots = np.zeros((N, 6))
left_task_space_traj_ddots = np.zeros((N, 6))
right_task_space_traj = np.zeros((N, 7))
right_task_space_traj_dots = np.zeros((N, 6))
right_task_space_traj_ddots = np.zeros((N, 6))
joint_traj = np.zeros((N, robot_rbdl_model.q_size))
joint_traj_dots = np.zeros((N, robot_rbdl_model.qdot_size))
joint_traj_ddots = np.zeros((N, robot_rbdl_model.qdot_size))

# Joint space interpolation
# -------------------------
# final_left_hand_pose = init_left_hand_pose.copy()
# final_right_hand_pose = init_right_hand_pose.copy()
# op_matrix = tf.transformations.quaternion_matrix(final_left_hand_pose[:4])
# op_matrix = op_matrix.dot(tf.transformations.rotation_matrix(np.deg2rad(-90), [0, 0, 1]))
# final_left_hand_pose[:4] = tf.transformations.quaternion_from_matrix(op_matrix)
# op_matrix = tf.transformations.quaternion_matrix(final_right_hand_pose[:4])
# op_matrix = op_matrix.dot(tf.transformations.rotation_matrix(np.deg2rad(90), [0, 0, 1]))
# final_right_hand_pose[:4] = tf.transformations.quaternion_from_matrix(op_matrix)
# final_left_hand_pose[6] += 0.2
# final_right_hand_pose[6] += 0.2
print(init_left_hand_pose)
print(final_left_hand_pose)
print(init_right_hand_pose)
print(final_right_hand_pose)
print('&^%&^%&^%')

q_reach = robot_model.ik(LH_name, final_left_hand_pose, body_offset=l_soft_hand_offset,
                         mask_joints=bigman_params['joint_ids']['TO'], joints_limits=bigman_params['joints_limits'],
                         method='optimization')
# Get configuration only for Right Arm
q_reach[bigman_params['joint_ids']['RA']] = robot_model.ik(RH_name, final_right_hand_pose,
                                                           body_offset=r_soft_hand_offset,
                                                           mask_joints=bigman_params['joint_ids']['TO'],
                                                           joints_limits=bigman_params['joints_limits'],
                                                           method='optimization')[bigman_params['joint_ids']['RA']]
# q_reach[bigman_params['joint_ids']['RA']] = q_reach_right[bigman_params['joint_ids']['RA']]

# # Move to q_reach to check configuration
# N = int(np.ceil(T_init*freq))
# joint_init_traj = polynomial5_interpolation(N, q_reach, joint_pos_state)[0]
# print("Moving to q_reach with Position control.")
# for ii in range(N):
#     des_cmd.position = joint_init_traj[ii, :]
#     des_cmd.stiffness = default_joint_stiffness
#     des_cmd.damping = default_joint_damping
#     publisher.publish(des_cmd)
#     pub_rate.sleep()
# init_left_hand_pose = robot_model.fk(LH_name, q=joint_pos_state, body_offset=l_soft_hand_offset,
#                                      update_kinematics=True, rotation_rep='quat')
# init_right_hand_pose = robot_model.fk(RH_name, q=joint_pos_state, body_offset=r_soft_hand_offset,
#                                       update_kinematics=True, rotation_rep='quat')
# print(init_left_hand_pose)
# print(init_right_hand_pose)
# raw_input('Press a key to continue...')

# -------------
# Interpolation
# -------------
interpolation_type = 0
if interpolation_type == 0:
    # Interpolation type 0: First task_space interp, then joint_space
    # ---------------------------------------------------------------
    print('Create task_space trajectory...')
    left_task_space_traj[:, 4:], left_task_space_traj_dots[:, 3:], left_task_space_traj_ddots[:, 3:] = \
        polynomial5_interpolation(N, final_left_hand_pose[4:], init_left_hand_pose[4:])
    left_task_space_traj[:, :4], left_task_space_traj_dots[:, :3], left_task_space_traj_ddots[:, :3] = \
        quaternion_slerp_interpolation(N, final_left_hand_pose[:4], init_left_hand_pose[:4])
    left_task_space_traj_dots *= freq
    left_task_space_traj_ddots *= freq**2

    right_task_space_traj[:, 4:], right_task_space_traj_dots[:, 3:], right_task_space_traj_ddots[:, 3:] = \
        polynomial5_interpolation(N, final_right_hand_pose[4:], init_right_hand_pose[4:])
    right_task_space_traj[:, :4], right_task_space_traj_dots[:, :3], right_task_space_traj_ddots[:, :3] = \
        quaternion_slerp_interpolation(N, final_right_hand_pose[:4], init_right_hand_pose[:4])
    right_task_space_traj_dots *= freq
    right_task_space_traj_ddots *= freq**2

    print('Create joint_space trajectory...')
    joint_traj[0, :] = joint_pos_state
    J_left = np.zeros((6, robot_rbdl_model.qdot_size))
    J_right = np.zeros((6, robot_rbdl_model.qdot_size))
    mask_joints = bigman_params['joint_ids']['TO']
    rarm_joints = bigman_params['joint_ids']['RA']
    for ii in range(N-1):
        # print('%d/%d' % (ii, N))
        # Compute the Jacobian matrix
        robot_model.update_jacobian(J_left, LH_name, joint_traj[ii, :], l_soft_hand_offset, update_kinematics=True)
        robot_model.update_jacobian(J_right, RH_name, joint_traj[ii, :], r_soft_hand_offset, update_kinematics=True)
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
        joint_traj[ii+1, :] = joint_traj[ii, :] + joint_traj_dots[ii, :] * 1./freq
    #joint_traj_dots = np.vstack((np.diff(joint_traj, axis=0), np.zeros((1, robot_rbdl_model.qdot_size))))
    #joint_traj_dots *= freq
    joint_traj_ddots = np.vstack((np.diff(joint_traj_dots, axis=0), np.zeros((1, robot_rbdl_model.qdot_size))))
    joint_traj_ddots *= freq

    # q_null = joint_pos_state.copy()
    # #q_null[17] = np.deg2rad(-45)
    # #q_null[12] = np.deg2rad(20)
    # #q_null[13] = np.deg2rad(40)
    # joint_traj[:, :], joint_traj_dots[:, :], joint_traj_ddots[:, :] = polynomial5_interpolation(N, q_null, joint_pos_state)
    # joint_traj_dots *= freq
    # joint_traj_ddots *= freq*freq

elif interpolation_type == 1:
    # Interpolation type 1: First joint_space interp, then task_space
    # ---------------------------------------------------------------
    print('Create joint_space trajectory...')
    joint_traj[:, :], joint_traj_dots[:, :], joint_traj_ddots[:, :] = polynomial5_interpolation(N, q_reach, joint_pos_state)
    joint_traj_dots *= freq
    joint_traj_ddots *= freq*freq

    print('Create task_space trajectory...')
    J_left = np.zeros((6, robot_rbdl_model.qdot_size))
    J_right = np.zeros((6, robot_rbdl_model.qdot_size))
    mask_joints = bigman_params['joint_ids']['TO']
    for ii in range(N):
        left_task_space_traj[ii, :] = robot_model.fk(LH_name, q=joint_traj[ii, :], body_offset=l_soft_hand_offset,
                                                     update_kinematics=True, rotation_rep='quat')
        right_task_space_traj[ii, :] = robot_model.fk(RH_name, q=joint_traj[ii, :], body_offset=r_soft_hand_offset,
                                                      update_kinematics=True, rotation_rep='quat')
        if ii > 0:
            if quaternion_inner(left_task_space_traj[ii, :4], left_task_space_traj[ii-1, :4]) < 0:
                left_task_space_traj[ii, :4] *= -1
            if quaternion_inner(right_task_space_traj[ii, :4], right_task_space_traj[ii-1, :4]) < 0:
                right_task_space_traj[ii, :4] *= -1
        robot_model.update_jacobian(J_left, LH_name, joint_traj[ii, :], l_soft_hand_offset, update_kinematics=True)
        robot_model.update_jacobian(J_right, RH_name, joint_traj[ii, :], r_soft_hand_offset, update_kinematics=True)
        J_left[:, mask_joints] = 0
        J_right[:, mask_joints] = 0
        left_task_space_traj_dots[ii, :] = J_left.dot(joint_traj_dots[ii, :])
        right_task_space_traj_dots[ii, :] = J_right.dot(joint_traj_dots[ii, :])
    left_task_space_traj_ddots = np.vstack((np.diff(left_task_space_traj_dots, axis=0), np.zeros((1, 6))))
    right_task_space_traj_ddots = np.vstack((np.diff(right_task_space_traj_dots, axis=0), np.zeros((1, 6))))
    left_task_space_traj_ddots *= freq
    right_task_space_traj_ddots *= freq


# -------------------------
# Task Space Torque Control
# -------------------------
# Preallocate matrices
tau = np.zeros(robot_rbdl_model.qdot_size)
tau_left = np.zeros(robot_rbdl_model.qdot_size)
tau_right = np.zeros(robot_rbdl_model.qdot_size)
taus_cmd_traj = np.zeros((N, robot_rbdl_model.qdot_size))
taus_traj = np.zeros((N, robot_rbdl_model.qdot_size))
multi_taus_traj = np.zeros((5, N, robot_rbdl_model.qdot_size))
qs_traj = np.zeros((N, robot_rbdl_model.q_size))
qdots_traj = np.zeros((N, robot_rbdl_model.q_size))

J_left = np.zeros((6, robot_rbdl_model.qdot_size))
J_right = np.zeros((6, robot_rbdl_model.qdot_size))
J_torso = np.zeros((6, robot_rbdl_model.qdot_size))
M_left = np.zeros((robot_rbdl_model.qdot_size, robot_rbdl_model.qdot_size))
M_right = np.zeros((robot_rbdl_model.qdot_size, robot_rbdl_model.qdot_size))
M_left_bar = np.zeros((6, 6))
M_right_bar = np.zeros((6, 6))
c_plus_g = np.zeros(robot_rbdl_model.qdot_size)
g = np.zeros(robot_rbdl_model.qdot_size)

task_left_pose_errors = np.zeros((N, 6))
task_right_pose_errors = np.zeros((N, 6))
real_left_task_space_traj = np.zeros_like(left_task_space_traj)
real_right_task_space_traj = np.zeros_like(right_task_space_traj)
real_left_task_space_traj_dots = np.zeros_like(left_task_space_traj_dots)
real_right_task_space_traj_dots = np.zeros_like(right_task_space_traj_dots)
left_singu_distances = np.zeros(N)
right_singu_distances = np.zeros(N)

# Only control joints_to_move
des_cmd.name = [bigman_params['joints_names'][idx] for idx in joints_to_move]
des_cmd.position = []

print("Moving to the initial configuration of trajectory with torque control.")
#raw_input("Press a key to continue...")

#Kp_task = np.eye(6)*np.array([1000, 1000, 1000, 50., 50., 50.], dtype=np.float64)
#Kd_task = np.sqrt(Kp_task)
# Nakanishi: high task space gain setting
Kp_task = np.eye(6)*np.array([1000, 1000, 1000, 50., 50., 50.], dtype=np.float64)
Kd_task = np.sqrt(Kp_task)

# Nakanishi: low task space gain setting
Kp_task = np.eye(6)*np.array([500, 500, 500, 25., 25., 25.], dtype=np.float64)
Kd_task = np.sqrt(Kp_task)

# Domingo: similar than low task space gain setting
#K_ori = np.tile(500, 3)
#K_pos = np.tile(150, 3)
#K_ori = np.tile(400, 3)
#K_pos = np.tile(25, 3)
#K_pos = np.tile(100, 3)
K_ori = np.tile(50, 3)
K_pos = np.tile(15, 3)
Kp_task = np.eye(6)*np.r_[K_ori, K_pos]
Kd_task = np.sqrt(Kp_task)

# Joint space Kd gain
Kd_q = Kd_tau#np.eye(robot_model.qdot_size)*0.1

#Kp_null = np.eye(robot_model.qdot_size)*10
#Kp_null = np.eye(robot_model.qdot_size)*2
#Kp_null = np.eye(robot_model.qdot_size)*0.2
Kp_null = np.eye(robot_model.qdot_size)*0.6

# Multitask controller
alpha_left = 1#0.5
alpha_right = 1#0.5

mask_joints = bigman_params['joint_ids']['TO']
inf_limits = np.array([bigman_params['joints_limits'][ii][0] for ii in range(robot_model.qdot_size)])
max_limits = np.array([bigman_params['joints_limits'][ii][1] for ii in range(robot_model.qdot_size)])

des_cmd.position = q_init[joints_to_move]
for ii in range(N):

    # Get current(sensed) joints values
    current_joint_pos = joint_pos_state.copy()
    current_joint_vel = joint_vel_state.copy()
    current_joint_effort = joint_effort_state.copy()

    # Update Jacobian(s)
    robot_model.update_jacobian(J_left, LH_name, current_joint_pos, l_soft_hand_offset, update_kinematics=True)
    robot_model.update_jacobian(J_right, RH_name, current_joint_pos, r_soft_hand_offset, update_kinematics=True)
    robot_model.update_jacobian(J_torso, torso_name, current_joint_pos, torso_offset, update_kinematics=True)
    J_left[:, bigman_params['joint_ids']['LB']] = 0
    J_left[:, bigman_params['joint_ids']['TO']] = 0
    J_left[:, bigman_params['joint_ids']['RA']] = 0
    J_right[:, bigman_params['joint_ids']['LB']] = 0
    J_right[:, bigman_params['joint_ids']['TO']] = 0
    J_right[:, bigman_params['joint_ids']['LA']] = 0

    # Update Non linear Effects (Coriolis + gravity forces)
    robot_model.update_nonlinear_forces(c_plus_g, current_joint_pos, current_joint_vel)
    # robot_model.update_nonlinear_forces(c_plus_g, joint_traj[ii, :], joint_traj_dots[ii, :])
    # Update gravity forces
    robot_model.update_gravity_forces(g, current_joint_pos)
    # robot_model.update_gravity_forces(g, joint_traj[ii, :])

    # Get J_dot_q_dot(s)
    J_left_dot_q_dot = robot_model.jdqd(LH_name, q=current_joint_pos, qdot=current_joint_vel,
                                        body_offset=l_soft_hand_offset, update_kinematics=True)
    J_right_dot_q_dot = robot_model.jdqd(RH_name, q=current_joint_pos, qdot=current_joint_vel,
                                         body_offset=r_soft_hand_offset, update_kinematics=True)

    # Get current operational point pose(s)
    real_left_task_space_traj[ii, :] = robot_model.fk(LH_name, q=current_joint_pos, body_offset=l_soft_hand_offset,
                                                      update_kinematics=True, rotation_rep='quat')
    real_right_task_space_traj[ii, :] = robot_model.fk(RH_name, q=current_joint_pos, body_offset=r_soft_hand_offset,
                                                       update_kinematics=True, rotation_rep='quat')
    # Check quaternion inversion
    if ii > 0:
        if quaternion_inner(real_left_task_space_traj[ii, :4], real_left_task_space_traj[ii-1, :4]) < 0:
            real_left_task_space_traj[ii, :4] *= -1
        if quaternion_inner(real_right_task_space_traj[ii, :4], real_right_task_space_traj[ii-1, :4]) < 0:
            real_right_task_space_traj[ii, :4] *= -1

    # Calculate current task space velocity(ies)
    real_left_task_space_traj_dots[ii, :] = J_left.dot(joint_vel_state)
    real_right_task_space_traj_dots[ii, :] = J_right.dot(joint_vel_state)

    # Calculate pose and velocities errors
    task_left_pose_error = compute_cartesian_error(left_task_space_traj[ii, :], real_left_task_space_traj[ii, :])
    task_right_pose_error = compute_cartesian_error(right_task_space_traj[ii, :], real_right_task_space_traj[ii, :])
    task_left_vel_error = left_task_space_traj_dots[ii, :] - real_left_task_space_traj_dots[ii, :]
    task_right_vel_error = right_task_space_traj_dots[ii, :] - real_right_task_space_traj_dots[ii, :]

    # Reference task-space acceleration(s)
    x_left_ddot_r = left_task_space_traj_ddots[ii, :] + Kp_task.dot(task_left_pose_error) + Kd_task.dot(task_left_vel_error)
    x_right_ddot_r = right_task_space_traj_ddots[ii, :] + Kp_task.dot(task_right_pose_error) + Kd_task.dot(task_right_vel_error)

    # Update Mass matrix
    # robot_model.update_inertia_matrix(M, joint_traj[ii, :])
    robot_model.update_inertia_matrix(M_left, current_joint_pos)
    robot_model.update_inertia_matrix(M_right, current_joint_pos)
    #M_left[bigman_params['joint_ids']['LB'], bigman_params['joint_ids']['LB']] = 0
    #M_left[bigman_params['joint_ids']['TO'], bigman_params['joint_ids']['TO']] = 0
    #M_left[bigman_params['joint_ids']['RA'], bigman_params['joint_ids']['RA']] = 0
    #M_right[bigman_params['joint_ids']['LB'], bigman_params['joint_ids']['LB']] = 0
    #M_right[bigman_params['joint_ids']['TO'], bigman_params['joint_ids']['TO']] = 0
    #M_right[bigman_params['joint_ids']['LA'], bigman_params['joint_ids']['LA']] = 0

    # #F = M_left_bar.dot(x_left_ddot_r)
    # rbdl.CompositeRigidBodyAlgorithm(robot_rbdl_model, current_joint_pos, M, update_kinematics=True)
    # M_left_bar[:, :] = np.linalg.inv(J_left.dot(np.linalg.inv(M)).dot(J_left.T))
    # rbdl.NonlinearEffects(robot_rbdl_model, current_joint_pos, current_joint_vel, c_plus_g)
    # tau_left = J_left.T.dot(M_left_bar).dot(x_left_ddot_r) + c_plus_g
    # # Null space
    # #u_null = Kp_null.dot(joint_traj[ii, :] - current_joint_pos)
    # u_null = Kp_null.dot(q_reach - current_joint_pos)
    # #u_null = Kp_null.dot(joint_traj[0, :] - current_joint_pos)
    # #u_null = 0.5*(q0 - current_joint_pos)*Kp_null.dot(q0 - current_joint_pos)
    # # Method1: Nakanishi
    # J_left_bar = np.linalg.inv(M).dot(J_left.T).dot(M_left_bar)
    # torque_null = (np.eye(robot_model.qdot_size) - J_left.T.dot(J_left_bar.T)).dot(u_null)
    # ## Method2: DeWolf
    # #J_left_bar_T = M_left_bar.dot(J_left).dot(np.linalg.inv(M))
    # #torque_null = (np.eye(robot_model.qdot_size) - J_left.T.dot(J_left_bar_T)).dot(u_null)
    # tau_left += torque_null

    # Nakanishi: Gauss Controller (Operational Space Controller in Khatib (1987))
    M_left_bar[:, :] = np.linalg.inv(J_left.dot(np.linalg.inv(M_left)).dot(J_left.T))
    M_right_bar[:, :] = np.linalg.inv(J_right.dot(np.linalg.inv(M_right)).dot(J_right.T))
    J_left_bar = np.linalg.inv(M_left).dot(J_left.T).dot(M_left_bar)
    J_right_bar = np.linalg.inv(M_right).dot(J_right.T).dot(M_right_bar)
    q_error_left = np.zeros_like(current_joint_pos)
    q_error_right = np.zeros_like(current_joint_pos)
    q0 = q_reach
    #q0 = joint_traj[ii, :]
    #q0 = q_init
    #q0 = np.array(current_joint_pos.copy())
    #q0[17] = joint_traj[ii, 17]
    #q0[18] = joint_traj[ii, 18]
    #q0[12] = joint_traj[ii, 12]; q0[13] = joint_traj[ii, 13]; q0[14] = joint_traj[ii, 14]
    #q0[bigman_params['joint_ids']['TO']] = joint_traj[ii, bigman_params['joint_ids']['TO']]
    ##q0[12] = 0; q0[13] = 0; q0[14] = 0
    ##q0 = np.zeros_like(current_joint_pos)
    q_error_left[bigman_params['joint_ids']['LA']] = (current_joint_pos - q0)[bigman_params['joint_ids']['LA']]
    q_error_right[bigman_params['joint_ids']['RA']] = (current_joint_pos - q0)[bigman_params['joint_ids']['RA']]
    q_grad_left = Kp_null.dot(q_error_left)
    q_grad_right = Kp_null.dot(q_error_right)
    #q_grad = Kp_null.dot(current_joint_pos - (inf_limits+max_limits)/2)
    alpha = 1
    #torque_null_left = -Kd_q.dot(current_joint_vel)*0 - alpha*q_grad
    #torque_null_right = -Kd_q.dot(current_joint_vel)*0 - alpha*q_grad
    #left_projection_null_times_torque_null = (np.eye(robot_model.qdot_size) - J_left.T.dot(J_left_bar.T)).dot(torque_null_left)
    #right_projection_null_times_torque_null = (np.eye(robot_model.qdot_size) - J_right.T.dot(J_right_bar.T)).dot(torque_null_right)
    #torque_null_left = M.dot(J_right_bar).dot(x_right_ddot_r - J_right_dot_q_dot*0) + right_projection_null_times_torque_null
    #torque_null_right = M.dot(J_left_bar).dot(x_left_ddot_r - J_left_dot_q_dot*0) + left_projection_null_times_torque_null
    torque_null_left = -Kd_q.dot(current_joint_vel)*0 - alpha*q_grad_left
    torque_null_right = -Kd_q.dot(current_joint_vel)*0 - alpha*q_grad_right
    left_projection_null_times_torque_null = (np.eye(robot_model.qdot_size) - J_left.T.dot(J_left_bar.T)).dot(torque_null_left)
    right_projection_null_times_torque_null = (np.eye(robot_model.qdot_size) - J_right.T.dot(J_right_bar.T)).dot(torque_null_right)

    tau_left = M_left.dot(J_left_bar).dot(x_left_ddot_r - J_left_dot_q_dot*0 + J_left.dot(np.linalg.inv(M_left)).dot(g)*0)\
               + left_projection_null_times_torque_null*0
    tau_right = M_right.dot(J_right_bar).dot(x_right_ddot_r - J_right_dot_q_dot*0 + J_right.dot(np.linalg.inv(M_right)).dot(g)*0)\
                + right_projection_null_times_torque_null*0
    # Multitask controller
    tau = alpha_left*tau_left + alpha_right*tau_right + c_plus_g
    #tau = alpha_left*tau_left + alpha_right*tau_right + g

    # # Nakanishi: Dynamical Decoupling Controller Variation 2
    # # (With Null Space Pre-multiplication of M, and Compensation of C and g in Joint Space)
    # rbdl.CompositeRigidBodyAlgorithm(robot_rbdl_model, current_joint_pos, M, update_kinematics=True)
    # M_left_bar[:, :] = np.linalg.inv(J_left.dot(np.linalg.inv(M)).dot(J_left.T))
    # J_left_bar = np.linalg.inv(M).dot(J_left.T).dot(M_left_bar)
    # q0 = q_reach
    # #q0 = np.zeros_like(current_joint_pos)
    # q_grad = Kp_null.dot(current_joint_pos - q0)
    # alpha = 1
    # q_ddot_0 = M.dot(-Kd_q.dot(current_joint_vel)*0 - alpha*q_grad)
    # torque_null = (np.eye(robot_model.qdot_size) - J_left_bar.dot(J_left)).dot(q_ddot_0)
    # tau_left = M.dot(J_left_bar.dot(x_left_ddot_r - J_left_dot_q_dot*0) + torque_null*0) + c_plus_g


    # Modugno: Unified Framework (UF)


    # # Del Prete: Sentis' WBC
    # J_1 = J_left
    # J_2 = J_right
    # x_ddot_1 = x_left_ddot_r
    # x_ddot_2 = x_right_ddot_r
    # J_1_dot_q_dot = J_right_dot_q_dot
    # J_2_dot_q_dot = J_right_dot_q_dot
    # Lambda_p_1 = np.linalg.pinv(J_1.dot(np.linalg.inv(M)).dot(J_1.T))
    # Lambda_p_2 = np.linalg.pinv(J_2.dot(np.linalg.inv(M)).dot(J_2.T))
    # #h = c_plus_g
    # h = g
    # #sum_F_p_0 = np.zeros_like(h)
    # #F_p_0 = Lambda_p_0.dot(x_ddot_0 - J_0_dot_q_dot*0 + J_0.dot(np.linalg.inv(M)).dot(h - sum_F_p_0)*0)
    # #sum_J_p_0 = np.zeros((robot_model.qdot_size, robot_model.qdot_size))
    # #J_p_0 = J_0.dot(np.eye(robot_model.qdot_size) - sum_J_p_0)
    # #dyn_consist_J_pseudo_0 = np.linalg.inv(M).dot(J_0.T).dot(Lambda_p_0)
    # #torque_0 = J_p_0.T.dot(F_p_0)
    # sum_F_p_1 = np.zeros_like(h)
    # F_p_1 = Lambda_p_1.dot(x_ddot_1 - J_1_dot_q_dot*0 + J_1.dot(np.linalg.inv(M)).dot(h - sum_F_p_1)*0)
    # sum_J_p_1 = np.zeros((robot_model.qdot_size, robot_model.qdot_size))
    # J_p_1 = J_1.dot(np.eye(robot_model.qdot_size) - sum_J_p_1)
    # dyn_consist_J_pseudo_1 = np.linalg.inv(M).dot(J_1.T).dot(Lambda_p_1)
    # torque_1 = J_p_1.T.dot(F_p_1)
    # sum_F_p_2 = J_p_1.T.dot(F_p_1)
    # F_p_2 = Lambda_p_2.dot(x_ddot_2 - J_2_dot_q_dot*0 + J_2.dot(np.linalg.inv(M)).dot(h - sum_F_p_2)*0)
    # sum_J_p_2 = dyn_consist_J_pseudo_1.dot(J_p_1)
    # J_p_2 = J_2.dot(np.eye(robot_model.qdot_size) - sum_J_p_2)
    # dyn_consist_J_pseudo_2 = np.linalg.inv(M).dot(J_2.T).dot(Lambda_p_2)
    # torque_2 = J_p_2.T.dot(F_p_2)
    # q_error = np.zeros(robot_model.qdot_size)
    # #q_error[bigman_params['joint_ids']['TO']] = (q_init - current_joint_pos)[bigman_params['joint_ids']['TO']]
    # q_error = q_init - current_joint_pos
    # q_ddot_3 = Kp_null.dot(q_error) - Kd_q.dot(current_joint_vel)*0
    # # torque_3 = (np.eye(robot_model.qdot_size) - J_1.T.dot(dyn_consist_J_pseudo_1.T)).dot(q_ddot_3)  # Works for no torque_2
    # sum_J_p_3 = dyn_consist_J_pseudo_2.dot(J_p_2) + sum_J_p_2
    # J_p_3 = np.eye(robot_model.qdot_size) - sum_J_p_3
    # torque_3 = J_p_3.T.dot(q_ddot_3)
    # torque = torque_1 + torque_2 + torque_3 + h
    # print(torque_1)
    # print(torque_2)
    # print(torque_3)
    # print(tau_left)
    # print(torque)
    # #raw_input('---')
    # tau = torque
    # #tau = tau_left

    # # Joint_space Torque control
    # rbdl.InverseDynamics(robot_rbdl_model, joint_traj[ii, :], joint_traj_dots[ii, :], joint_traj_ddots[ii, :], tau)
    # pd_tau = Kp_tau.dot(joint_traj[ii, :] - joint_pos_state) + Kd_tau.dot(joint_traj_dots[ii, :] - joint_vel_state)
    # tau += pd_tau

    # Distance from singularities
    U_l, s_l, V_l = np.linalg.svd(J_left[:, bigman_params['joint_ids']['LA']], full_matrices=False)
    U_r, s_r, V_r = np.linalg.svd(J_right[:, bigman_params['joint_ids']['RA']], full_matrices=False)
    # # \mu = sqrt(|J J^T|)
    # singu_distance_left = np.sqrt(np.linalg.det(J_left.dot(J_left.T)))
    # singu_distance_right = np.sqrt(np.linalg.det(J_right.dot(J_right.T)))
    # print('%.4f -- %.4f' % (singu_distance_left, singu_distance_right))
    # \mu = \prod(\sigma_i) where \sigma_i is the i singular value
    singu_distance_left = np.prod(s_l)
    singu_distance_right = np.prod(s_r)
    print('%.4f -- %.4f' % (singu_distance_left, singu_distance_right))
    singu_distance_left = np.min(s_l)
    singu_distance_right = np.min(s_r)
    print('%.4f -- %.4f' % (singu_distance_left, singu_distance_right))
    print('---')
    left_singu_distances[ii] = singu_distance_left
    right_singu_distances[ii] = singu_distance_right
    #raw_input('nonononono')

    # Uncomment to send torque references
    des_cmd.position = []
    des_cmd.effort = tau[joints_to_move]
    des_cmd.stiffness = np.zeros_like(tau[joints_to_move])
    des_cmd.damping = np.zeros_like(tau[joints_to_move])

    # # Uncomment to send position references
    # des_cmd.position = joint_traj[ii, joints_to_move]
    # des_cmd.stiffness = default_joint_stiffness[joints_to_move]
    # des_cmd.damping = default_joint_damping[joints_to_move]

    publisher.publish(des_cmd)
    taus_cmd_traj[ii, :] = tau
    taus_traj[ii, :] = joint_effort_state
    qs_traj[ii, :] = joint_pos_state
    qdots_traj[ii, :] = joint_vel_state
    task_left_pose_errors[ii, :] = task_left_pose_error
    task_right_pose_errors[ii, :] = task_right_pose_error
    multi_taus_traj[0, ii, :] = tau
    multi_taus_traj[1, ii, :] = tau_left
    multi_taus_traj[2, ii, :] = tau_right
    multi_taus_traj[3, ii, :] = c_plus_g
    multi_taus_traj[4, ii, :] = g
    pub_rate.sleep()

# Return to position control
print("Changing to position control!")
des_cmd.position = joint_pos_state[joints_to_move]
for ii in range(50):
    des_cmd.stiffness = default_joint_stiffness[joints_to_move]
    des_cmd.damping = default_joint_damping[joints_to_move]
    publisher.publish(des_cmd)
    pub_rate.sleep()


# ##### #
# PLOTS #
# ##### #
#joints_to_plot = bigman_params['joint_ids']['LA']# + bigman_params['joint_ids']['TO']
joints_to_plot = joints_to_move
cols = 3
task_names = ['qx', 'qy', 'qz', 'qw', 'x', 'y', 'z']
task_error_names = ['omegax', 'omegay', 'omegaz', 'x', 'y', 'z']
joint_names = [bigman_params['joints_names'][idx] for idx in joints_to_plot]
print("Plotting...")
#plot_desired_sensed_data(range(7), left_task_space_traj, real_left_task_space_traj, task_names, data_type='pose',
#                         block=False, legend=False)
plot_desired_sensed_data(range(7), right_task_space_traj, real_right_task_space_traj, task_names, data_type='pose',
                         block=False, legend=False)
plot_desired_sensed_data(range(6), task_left_pose_errors, task_right_pose_errors, task_error_names,
                         data_type='pose-error', block=False, legend=False)
plot_desired_sensed_data(joints_to_plot, joint_traj, qs_traj, joint_names, data_type='position',
                         limits=bigman_params['joints_limits'], block=False, legend=False)
#plot_desired_sensed_data(joints_to_plot, np.tile(q_init, (N, 1)), qs_traj, joint_names, data_type='position', block=False)
#plot_desired_sensed_data(joints_to_plot, joint_traj_dots*0, qdots_traj, joint_names, data_type='velocity', block=False)
plot_desired_sensed_data(joints_to_plot, taus_cmd_traj, taus_traj, joint_names, data_type='torque', block=False)

plot_joint_multi_info(joints_to_plot, multi_taus_traj,  joint_names, data='torque', block=True, cols=3, legend=True,
                      labels=['total', 'left', 'right', 'c+g', 'g'])

# plt.figure()
# plt.plot(left_singu_distances)
# plt.plot(right_singu_distances)
# plt.show()
# plot_desired_sensed_torque_position(joints_to_plot, taus_cmd_traj, taus_traj,
#                                     joint_traj, qs_traj, joint_names, block=True, cols=cols)

print("Saving sensed torques in %s" % torques_saved_filename)
np.save(torques_saved_filename, taus_traj)
raw_input('Press a key to close the script!!')
sys.exit()
