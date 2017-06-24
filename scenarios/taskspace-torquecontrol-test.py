import sys
import numpy as np
import scipy
import math
import os
import rospy
import matplotlib.pyplot as plt
import tf
from XCM.msg import CommandAdvr
from XCM.msg import JointStateAdvr
from robolearn.utils.trajectory_reproducer import TrajectoryReproducer
from robolearn.utils.iit.iit_robots_params import bigman_params
from robolearn.utils.transformations import *
from robolearn.utils.trajectory_interpolators import polynomial5_interpolation, quaternion_slerp_interpolation
from robolearn.utils.plot_utils import plot_desired_sensed_torque_position
from robolearn.utils.plot_utils import plot_joint_info
from robolearn.utils.plot_utils import plot_desired_sensed_data
from robolearn.utils.lift_box_utils import create_box_relative_pose, create_ee_relative_pose
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import DeleteModel
from geometry_msgs.msg import Pose
import rbdl

from robolearn.utils.robot_model import RobotModel

np.set_printoptions(precision=4, suppress=True, linewidth=1000)

# Always turn off Gazebo logger
os.system("gz log -d 0")

#current_path = os.path.abspath(__file__)
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

torques_saved_filename = 'torques_init_traj.npy'

T_init = 5  # Time to move from current position to T_init
T_traj = 5  # Time to execute the trajectory
freq = 100

# BOX
box_x = 0.75-0.05
box_y = 0.00
box_z = 0.0184
box_yaw = 0  # Degrees
box_size = [0.4, 0.5, 0.3]
box_relative_pose = create_box_relative_pose(box_x=box_x, box_y=box_y, box_z=box_z, box_yaw=box_yaw)
final_left_ee_pose = create_ee_relative_pose(box_relative_pose, ee_x=0, ee_y=box_size[1]/2-0.02, ee_z=0, ee_yaw=0)
final_left_ee_pose = final_left_ee_pose[[3, 4, 5, 6, 0, 1, 2]]


# ROBOT MODEL for trying ID
robot_urdf_file = os.environ["ROBOTOLOGY_ROOT"]+'/configs/ADVR_shared/bigman/urdf/bigman.urdf'
robot_rbdl_model = rbdl.loadModel(robot_urdf_file, verbose=False, floating_base=False)
robot_model = RobotModel(robot_urdf_file=robot_urdf_file)
LH_name = 'LWrMot3'
RH_name = 'RWrMot3'
l_soft_hand_offset = np.array([0.000, -0.030, -0.210])
r_soft_hand_offset = np.array([0.000, 0.030, -0.210])

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
# pd_tau_weights = np.array([0.80,  0.50,  0.80,  0.50,  0.50,  0.20,
#                            0.80,  0.50,  0.50,  0.50,  0.50,  0.20,
#                            0.50,  0.80,  0.50,
#                            0.50,  0.80,  0.50,  0.50,  0.10,  0.20,   0.03,
#                            0.03,  0.03,
#                            0.50,  0.80,  0.50,  0.50,  0.10,  0.20,   0.03])
pd_tau_weights = np.array([0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                           0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                           0.0,  0.0,  0.0,
                           0.50,  0.80,  0.50,  0.50,  0.20,  0.20,   0.03,
                           0.0,  0.0,
                           0.50,  0.80,  0.50,  0.50,  0.20,  0.20,   0.03])
Kp_tau = np.eye(robot_rbdl_model.q_size)*(100 * pd_tau_weights)
Kd_tau = np.eye(robot_rbdl_model.qdot_size)*(2 * pd_tau_weights)
Kp_tau = np.eye(robot_rbdl_model.q_size)*np.array([0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                                                   0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                                                   0.0,  0.0,  0.0,
                                                   #50,  80,  50,  50,  100,  200,   -7.7036e05,
                                                   500,  0,  0,  0,  0,  0,   50000,
                                                   0.0,  0.0,
                                                   0.0,  0.0,  0.0,  0.0,  0.0,  0.0,   0.0])
Kd_tau = np.eye(robot_rbdl_model.qdot_size)*np.array([0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                                                      0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                                                      0.0,  0.0,  0.0,
                                                      #1.,  1.,  1,  1,  1.,  1.,   4.72e02,
                                                      10.,  0.,  0,  0,  0.,  0.,   500,
                                                      0.0,  0.0,
                                                      0.0,  0.0,  0.0,  0.0,  0.0,  0.0,   0.0])


joint_pos_state = np.zeros(robot_rbdl_model.q_size)
joint_vel_state = np.zeros(robot_rbdl_model.qdot_size)
joint_effort_state = np.zeros(robot_rbdl_model.qdot_size)
joint_stiffness_state = np.zeros(robot_rbdl_model.qdot_size)
joint_damping_state = np.zeros(robot_rbdl_model.qdot_size)
joint_state_id = []


def callback(data, params):
    joint_ids = params[0]
    joint_pos_state = params[1]
    joint_effort_state = params[2]
    #if not joint_ids:
    #    joint_ids[:] = [bigman_params['joints_names'].index(name) for name in data.name]
    joint_ids[:] = [bigman_params['joints_names'].index(name) for name in data.name]
    joint_pos_state[joint_ids] = data.link_position
    joint_effort_state[joint_ids] = data.effort
    joint_stiffness_state[joint_ids] = data.stiffness
    joint_damping_state[joint_ids] = data.damping
    joint_vel_state[joint_ids] = data.link_velocity

publisher = rospy.Publisher("/xbotcore/bigman/command", CommandAdvr, queue_size=10)
subscriber = rospy.Subscriber("/xbotcore/bigman/joint_states", JointStateAdvr, callback, (joint_state_id, joint_pos_state, joint_effort_state))
rospy.init_node('torquecontrol_example')
pub_rate = rospy.Rate(freq)
des_cmd = CommandAdvr()
des_cmd.name = bigman_params['joints_names']

q_init = np.zeros(robot_rbdl_model.q_size)
q_init[16] = np.deg2rad(90)
q_init[17] = np.deg2rad(90)
N = int(np.ceil(T_init*freq))
joint_init_traj = polynomial5_interpolation(N, q_init, joint_pos_state)[0]
print("Moving to zero configuration with Position control.")
for ii in range(N):
    des_cmd.position = joint_init_traj[ii, :]
    des_cmd.stiffness = default_joint_stiffness
    des_cmd.damping = default_joint_damping
    publisher.publish(des_cmd)
    pub_rate.sleep()

N = int(np.ceil(T_traj*freq))
joints_to_move = bigman_params['joint_ids']['BA'][:7]# + bigman_params['joint_ids']['TO']
#joints_to_move = [bigman_params['joint_ids']['BA'][6]]



# PAUSE:
print("Sleeping some seconds..")
rospy.sleep(1)

init_left_ee_pose = robot_model.fk(LH_name, q=q_init, body_offset=l_soft_hand_offset,
                                   update_kinematics=True,
                                   rotation_rep='quat')

task_space_traj = np.zeros((N, 7))
task_space_traj_dots = np.zeros((N, 6))
task_space_traj_ddots = np.zeros((N, 6))

joint_traj = np.zeros((N, robot_rbdl_model.q_size))
joint_traj_dots = np.vstack((np.diff(joint_traj, axis=0), np.zeros((1, robot_rbdl_model.qdot_size))))
joint_traj_ddots = np.vstack((np.diff(joint_traj_dots, axis=0), np.zeros((1, robot_rbdl_model.qdot_size))))

#final_left_ee_pose = init_left_ee_pose.copy()
q_reach = final_left_ee_pose
q_reach = robot_model.ik(LH_name, final_left_ee_pose, body_offset=l_soft_hand_offset,
                         mask_joints=bigman_params['joint_ids']['TO'], joints_limits=bigman_params['joints_limits'],
                         method='optimization')
joint_traj, joint_traj_dots, joint_traj_ddots = polynomial5_interpolation(N, q_reach, joint_pos_state)

joint_traj_dots *= freq
joint_traj_ddots *= freq*freq

task_space_traj[:, 4:], task_space_traj_dots[:, 3:], task_space_traj_ddots[:, 3:] = \
    polynomial5_interpolation(N, final_left_ee_pose[4:], init_left_ee_pose[4:])

task_space_traj[:, :4], task_space_traj_dots[:, :3], task_space_traj_ddots[:, :3] = \
    quaternion_slerp_interpolation(N, final_left_ee_pose[:4], init_left_ee_pose[:4])
task_space_traj_dots *= freq
task_space_traj_ddots *= freq*freq

J = np.zeros((6, robot_rbdl_model.qdot_size))
#for ii in range(N):
#    task_space_traj[ii, :] = robot_model.fk(LH_name, q=joint_traj[ii, :], body_offset=l_soft_hand_offset,
#                                            update_kinematics=True, rotation_rep='quat')
#    robot_model.update_jacobian(J, LH_name, joint_traj[ii, :], l_soft_hand_offset, update_kinematics=True)
#    task_space_traj_dots[ii, :] = J.dot(joint_traj_dots[ii, :])
#task_space_traj_ddots = np.vstack((np.diff(task_space_traj_ddots, axis=0), np.zeros((1, 6))))
#task_space_traj_ddots *= freq

# joint_traj_dots *= freq
# joint_traj_ddots *= freq*freq


tau = np.zeros(robot_rbdl_model.qdot_size)
a = np.zeros(robot_rbdl_model.qdot_size)
des_cmd.name = [bigman_params['joints_names'][idx] for idx in joints_to_move]
des_cmd.position = []
qs_traj = np.zeros((N, robot_rbdl_model.q_size))
qdots_traj = np.zeros((N, robot_rbdl_model.q_size))
taus_cmd_traj = np.zeros((N, robot_rbdl_model.qdot_size))
taus_traj = np.zeros((N, robot_rbdl_model.qdot_size))

J = np.zeros((6, robot_rbdl_model.qdot_size))
M = np.zeros((robot_rbdl_model.qdot_size, robot_rbdl_model.qdot_size))
M_bar = np.zeros((6, 6))
C_bar = np.zeros(6)
g = np.zeros(robot_rbdl_model.qdot_size)
task_pose_errors = np.zeros((N, 6))

real_task_space_traj = np.zeros_like(task_space_traj)
real_task_space_traj_dots = np.zeros_like(task_space_traj_dots)



print("Moving to the initial configuration of trajectory with torque control.")
#raw_input("Press a key to continue...")

Kp_ik = np.eye(6)*np.array([1., 1., 1., 4., 5., 5.], dtype=np.float64)
Kd_ik = np.eye(6)*np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.3])

#Kp_task = np.eye(6)*np.array([1000, 1000, 1000, 50., 50., 50.], dtype=np.float64)
#Kd_task = np.sqrt(Kp_task)
# Nakanishi: high task space gain setting
Kp_task = np.eye(6)*np.array([1000, 1000, 1000, 50., 50., 50.], dtype=np.float64)
Kd_task = np.sqrt(Kp_task)

# Nakanishi: low task space gain setting
Kp_task = np.eye(6)*np.array([500, 500, 500, 25., 25., 25.], dtype=np.float64)
Kd_task = np.sqrt(Kp_task)

## Domingo: low task space gain setting
#Kp_task = np.eye(6)*np.array([500.00, 500.00, 500.00, 200., 200., 200.], dtype=np.float64)
#Kd_task = 2*np.sqrt(Kp_task)

#real_task_space_traj[0, :] = robot_model.fk(LH_name, q=joint_pos_state, body_offset=l_soft_hand_offset,
#                                             update_kinematics=True, rotation_rep='quat')[[3, 4, 5, 6, 0, 1, 2]]
des_cmd.position = q_init[joints_to_move]
for ii in range(N):
    # robot_model.update_jacobian(J, LH_name, joint_pos_state, l_soft_hand_offset, update_kinematics=True)
    # real_task_space_traj[ii, :] = robot_model.fk(LH_name, q=joint_pos_state, body_offset=l_soft_hand_offset,
    #                                              update_kinematics=True, rotation_rep='quat')
    # real_task_space_traj_dots[ii, :] = J.dot(joint_vel_state)
    # task_error = compute_cartesian_error(task_space_traj[ii, :], real_task_space_traj[ii, :])
    # v = task_space_traj_dots[ii, :] + Kp_ik.dot(task_error)
    # #joint_traj_dots[ii, :] = np.linalg.pinv(J).dot(v)
    # joint_traj_dots[ii, :] = scipy.sparse.linalg.lsqr(J, v)[0]
    # #des_cmd.position += joint_traj_dots[ii, joints_to_move]/freq
    # ##des_cmd.position = joint_traj[ii, joints_to_move]
    # #des_cmd.stiffness = default_joint_stiffness[joints_to_move]
    # #des_cmd.damping = default_joint_damping[joints_to_move]

    # rbdl.InverseDynamics(robot_rbdl_model, joint_traj[ii, :], joint_traj_dots[ii, :], joint_traj_ddots[ii, :], tau)
    # pd_tau = Kp_tau.dot(joint_traj[ii, :] - joint_pos_state) + \
    #          Kd_tau.dot(joint_traj_dots[ii, :] - joint_vel_state)
    # tau += pd_tau

    current_pos = joint_pos_state.copy()
    current_vel = joint_vel_state.copy()
    current_effort = joint_effort_state.copy()

    robot_model.update_jacobian(J, LH_name, current_pos, l_soft_hand_offset, update_kinematics=True)
    rbdl.CompositeRigidBodyAlgorithm(robot_rbdl_model, current_pos, M, update_kinematics=True)
    M_bar[:, :] = np.linalg.inv(J.dot(np.linalg.inv(M)).dot(J.T))

    real_task_space_traj[ii, :] = robot_model.fk(LH_name, q=current_pos, body_offset=l_soft_hand_offset,
                                                 update_kinematics=True, rotation_rep='quat')
    real_task_space_traj_dots[ii, :] = J.dot(joint_vel_state)
    task_pose_error = compute_cartesian_error(task_space_traj[ii, :], real_task_space_traj[ii, :])
    task_vel_error = task_space_traj_dots[ii, :]*0 - real_task_space_traj_dots[ii, :]

    x_ddot_r = task_space_traj_ddots[ii, :]*0 + Kp_task.dot(task_pose_error) + Kd_task.dot(task_vel_error)
    #F = M_bar.dot(x_ddot_r)
    #J_bar = np.linalg.inv(M).dot(J.T).dot(M_bar)
    rbdl.NonlinearEffects(robot_rbdl_model, current_pos, current_vel, g)

    tau = J.T.dot(M_bar).dot(x_ddot_r) + g

    # rbdl.NonlinearEffects(robot_rbdl_model, joint_pos_state, joint_vel_state*0, g)
    # rbdl.CompositeRigidBodyAlgorithm(robot_rbdl_model, joint_pos_state, M, update_kinematics=True)
    # u_torque = M.dot(Kp_tau.dot(pos_error) + Kd_tau.dot(vel_error) + g)

    # current_pos = joint_pos_state.copy()
    # current_vel = joint_vel_state.copy()
    # current_effort = joint_effort_state.copy()
    # rbdl.CompositeRigidBodyAlgorithm(robot_rbdl_model, current_pos, M, update_kinematics=True)
    # pos_error = q_init - current_pos
    # vel_error = np.zeros_like(current_vel) - current_vel
    # u_torque = Kp_tau.dot(pos_error) + Kd_tau.dot(vel_error)
    # #u_torque = Kd_tau.dot(vel_error)# + Kd_tau.dot(vel_error)
    # rbdl.InverseDynamics(robot_rbdl_model, current_pos, current_vel, u_torque*0, tau)
    # rbdl.InverseDynamics(robot_rbdl_model, joint_pos_state, joint_vel_state, u_torque, tau)
    #rbdl.NonlinearEffects(robot_rbdl_model, current_pos, current_vel, g)
    #print(repr(g[joints_to_move]))
    #rbdl.InverseDynamics(robot_rbdl_model, q_init, joint_vel_state*0, u_torque*0, tau)
    #rbdl.NonlinearEffects(robot_rbdl_model, joint_pos_state, joint_vel_state*0, g)
    # #tau = M.dot(J_bar).dot()
    # #tau = J[:, joints_to_move].T.dot(F) + g[joints_to_move]
    # tau = J.T.dot(F) + g
    #tau = g# + Kp_tau.dot(q_init - joint_pos_state)

    #taus_traj[ii, :] = joint_effort_state
    #print(joint_traj[ii, joints_to_move] - joint_pos_state[joints_to_move])
    #rbdl.NonlinearEffects(robot_rbdl_model, joint_pos_state, joint_vel_state*0, g)
    #a = joint_traj_ddots[ii, :] + \
    #    default_joint_damping*0 * (joint_traj_dots[ii, :] - joint_vel_state) + \
    #    default_joint_stiffness*0.0 * (joint_traj[ii, :] - joint_pos_state)
    #a = default_joint_damping * (joint_traj_dots[ii, :] - joint_vel_state)
    #rbdl.CompositeRigidBodyAlgorithm(robot_rbdl_model, joint_pos_state, M, update_kinematics=True)
    #rbdl.InverseDynamics(robot_rbdl_model, joint_pos_state, joint_vel_state/freq, joint_traj_ddots[ii, :]/(freq*freq), tau)
    #rbdl.InverseDynamics(robot_rbdl_model, joint_traj[ii, :], joint_traj_dots[ii, :], joint_traj_ddots[ii, :], tau)
    #tau += M.dot(a)
    des_cmd.position = []
    des_cmd.effort = tau[joints_to_move]
    des_cmd.stiffness = np.zeros_like(tau[joints_to_move])
    des_cmd.damping = np.zeros_like(tau[joints_to_move])

    publisher.publish(des_cmd)
    taus_traj[ii, :] = joint_effort_state
    taus_cmd_traj[ii, :] = tau
    qs_traj[ii, :] = joint_pos_state
    qdots_traj[ii, :] = joint_vel_state
    task_pose_errors[ii, :] = task_pose_error
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
joints_to_plot = bigman_params['joint_ids']['LA']# + bigman_params['joint_ids']['TO']
cols = 3
task_names = ['qx', 'qy', 'qz', 'qw', 'x', 'y', 'z']
task_error_names = ['omegax', 'omegay', 'omegaz', 'x', 'y', 'z']
joint_names = [bigman_params['joints_names'][idx] for idx in joints_to_plot]
print("Plotting...")
plot_desired_sensed_data(range(7), task_space_traj, real_task_space_traj, task_names, data_type='pose', block=False)
#plot_desired_sensed_data(range(6), task_pose_errors, task_pose_errors, task_error_names, data_type='pose', block=False)
#plot_desired_sensed_data(joints_to_plot, joint_traj, qs_traj, joint_names, data_type='position', block=False)
#plot_desired_sensed_data(joints_to_plot, np.tile(q_init, (N, 1)), qs_traj, joint_names, data_type='position', block=False)
#plot_desired_sensed_data(joints_to_plot, joint_traj_dots*0, qdots_traj, joint_names, data_type='velocity', block=False)
plot_desired_sensed_data(joints_to_plot, taus_cmd_traj, taus_traj, joint_names, data_type='torque', block=True)
# plot_desired_sensed_torque_position(joints_to_plot, taus_cmd_traj, taus_traj,
#                                     joint_traj, qs_traj, joint_names, block=True, cols=cols)

print("Saving sensed torques in %s" % torques_saved_filename)
np.save(torques_saved_filename, taus_traj)
raw_input('Press a key to close the script!!')
sys.exit()
