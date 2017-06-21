import sys
import numpy as np
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

T_init = 3
T_traj = 5
T_impedance_zero = 10
freq = 100

# BOX
box_x = 0.75-0.05
box_y = 0.00
box_z = 0.0184
box_yaw = 0  # Degrees
box_size = [0.4, 0.5, 0.3]
box_relative_pose = create_box_relative_pose(box_x=box_x, box_y=box_y, box_z=box_z, box_yaw=box_yaw)
final_left_ee_pose = create_ee_relative_pose(box_relative_pose, ee_x=0, ee_y=box_size[1]/2-0.02, ee_z=0, ee_yaw=0)

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
pd_tau_weights = np.array([0.80,  0.50,  0.80,  0.50,  0.50,  0.20,
                           0.80,  0.50,  0.50,  0.50,  0.50,  0.20,
                           0.50,  0.80,  0.50,
                           0.50,  0.80,  0.30,  0.50,  0.10,  0.20,   0.03,
                           0.03,  0.03,
                           0.50,  0.80,  0.50,  0.50,  0.10,  0.20,   0.03])
Kp_tau = 100 * pd_tau_weights
Kd_tau = 2 * pd_tau_weights


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
joints_to_move = bigman_params['joint_ids']['BA'][:7]
#joints_to_move = [bigman_params['joint_ids']['BA'][6]]


init_left_ee_pose = robot_model.fk(LH_name, q=q_init, body_offset=l_soft_hand_offset,
                                   update_kinematics=True,
                                   rotation_rep='quat')

task_space_traj = np.zeros((N, 7))
task_space_traj_dots = np.zeros((N, 6))
task_space_traj_ddots = np.zeros((N, 6))
joint_traj = np.zeros((N, robot_rbdl_model.q_size))
#task_space_traj_dots = np.zeros((N, 7))
#task_space_traj_ddots = np.zeros((N, 7))
task_space_traj[:, 4:] = polynomial5_interpolation(N, final_left_ee_pose[4:], init_left_ee_pose[4:])[0]
linspace_interp = np.linspace(0, 1, N)
for ii in range(N):
    task_space_traj[ii, :3] = tf.transformations.quaternion_slerp(init_left_ee_pose[:3], final_left_ee_pose[:3],
                                                                  linspace_interp[ii])

#joint_traj_dots = np.vstack((np.diff(joint_traj, axis=0), np.zeros((1, robot_rbdl_model.qdot_size))))*freq
#joint_traj_ddots = np.vstack((np.diff(joint_traj_dots, axis=0), np.zeros((1, robot_rbdl_model.qdot_size))))*freq*freq
task_space_traj_dots[:, 3:] = np.vstack((np.diff(task_space_traj[:, 4:], axis=0), np.zeros((1, 3))))

#TODO: Calculate a correct angular velocity
for ii in range(N-1):
    task_space_traj_dots[ii, :3] = quat_difference(task_space_traj[ii+1, :4], task_space_traj[ii, :4])

task_space_traj_ddots[:, :] = np.vstack((np.diff(task_space_traj_dots, axis=0), np.zeros((1, 6))))
task_space_traj_dots *= freq
task_space_traj_ddots *= freq*freq

joint_traj_dots = np.vstack((np.diff(joint_traj, axis=0), np.zeros((1, robot_rbdl_model.qdot_size))))
joint_traj_ddots = np.vstack((np.diff(joint_traj_dots, axis=0), np.zeros((1, robot_rbdl_model.qdot_size))))

# q_end = np.zeros(robot_rbdl_model.q_size)
# q_end[16] = np.deg2rad(90)
# q_end[joints_to_move] += np.deg2rad(-20)
# joint_traj, joint_traj_dots, joint_traj_ddots = polynomial5_interpolation(N, q_end, joint_pos_state)
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

print("Moving to the initial configuration of trajectory with torque control.")
raw_input("Press a key to continue...")
for ii in range(N):
    robot_model.update_jacobian(J, LH_name, joint_pos_state, l_soft_hand_offset, update_kinematics=True)
    rbdl.CompositeRigidBodyAlgorithm(robot_rbdl_model, joint_pos_state, M, update_kinematics=True)
    M_bar[:, :] = np.linalg.inv(J.dot(np.linalg.inv(M)).dot(np.transpose(J)))
    print(repr(M_bar))
    print("--")

    #des_cmd.position = joint_traj[ii, joints_to_move]
    #des_cmd.stiffness = default_joint_stiffness[joints_to_move]
    #des_cmd.damping = default_joint_damping[joints_to_move]
    rbdl.InverseDynamics(robot_rbdl_model, joint_traj[ii, :], joint_traj_dots[ii, :], joint_traj_ddots[ii, :], tau)
    #taus_traj[ii, :] = joint_effort_state
    #print(joint_traj[ii, joints_to_move] - joint_pos_state[joints_to_move])
    #rbdl.NonlinearEffects(robot_rbdl_model, joint_pos_state, joint_vel_state, g)
    #rbdl.NonlinearEffects(robot_rbdl_model, joint_pos_state, joint_vel_state*0, g)
    #a = joint_traj_ddots[ii, :] + \
    #    default_joint_damping*0 * (joint_traj_dots[ii, :] - joint_vel_state) + \
    #    default_joint_stiffness*0.0 * (joint_traj[ii, :] - joint_pos_state)
    pd_tau = Kp_tau * (joint_traj[ii, :] - joint_pos_state) + \
             Kd_tau * (joint_traj_dots[ii, :] - joint_vel_state)
    #pd_tau = default_joint_stiffness * (joint_traj[ii, :] - joint_pos_state) + \
    #         default_joint_damping * (joint_traj_dots[ii, :] - joint_vel_state)
    tau += pd_tau
    #rbdl.InverseDynamics(robot_rbdl_model, joint_pos_state, joint_vel_state, a, tau)
    #rbdl.NonlinearEffects(robot_rbdl_model, joint_traj[ii, :], joint_vel_state*0, tau)
    #tau = np.ones(robot_rbdl_model.qdot_size)*-0.5
    #a = default_joint_damping * (joint_traj_dots[ii, :] - joint_vel_state)
    #rbdl.CompositeRigidBodyAlgorithm(robot_rbdl_model, joint_pos_state, M, update_kinematics=True)
    #rbdl.InverseDynamics(robot_rbdl_model, joint_pos_state, joint_vel_state/freq, joint_traj_ddots[ii, :]/(freq*freq), tau)
    #rbdl.InverseDynamics(robot_rbdl_model, joint_traj[ii, :], joint_traj_dots[ii, :], joint_traj_ddots[ii, :], tau)
    #tau += M.dot(a)
    # des_cmd.position = []
    # des_cmd.effort = tau[joints_to_move]
    # des_cmd.stiffness = np.zeros_like(tau[joints_to_move])
    # des_cmd.damping = np.zeros_like(tau[joints_to_move])

    publisher.publish(des_cmd)
    taus_traj[ii, :] = joint_effort_state
    taus_cmd_traj[ii, :] = tau
    qs_traj[ii, :] = joint_pos_state
    qdots_traj[ii, :] = joint_vel_state
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
joints_to_plot = bigman_params['joint_ids']['LA']
cols = 3
joint_names = [bigman_params['joints_names'][idx] for idx in joints_to_plot]
print("Plotting...")
plot_desired_sensed_data(joints_to_plot, joint_traj, qs_traj, joint_names, data_type='position', block=False)
plot_desired_sensed_data(joints_to_plot, joint_traj_dots, qdots_traj, joint_names, data_type='velocity', block=False)
plot_desired_sensed_data(joints_to_plot, taus_cmd_traj, taus_traj, joint_names, data_type='torque', block=True)
# plot_desired_sensed_torque_position(joints_to_plot, taus_cmd_traj, taus_traj,
#                                     joint_traj, qs_traj, joint_names, block=True, cols=cols)

print("Saving sensed torques in %s" % torques_saved_filename)
np.save(torques_saved_filename, taus_traj)
sys.exit()
