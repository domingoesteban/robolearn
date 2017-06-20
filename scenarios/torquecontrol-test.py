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
from robolearn.utils.trajectory_interpolators import polynomial5_interpolation
from robolearn.utils.plot_utils import plot_desired_sensed_torque_position
from robolearn.utils.plot_utils import plot_joint_info
from robolearn.utils.plot_utils import plot_desired_sensed_data
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

load_torques = False
torques_saved_filename = 'torques_init_traj.npy'

T_init = 3
T_traj = 5
T_impedance_zero = 10
freq = 100

# ROBOT MODEL for trying ID
#robot_urdf_file = '/home/domingo/robotology-superbuild/robots/iit-bigman-ros-pkg/bigman_urdf/urdf/bigman.urdf'
robot_urdf_file = '/home/domingo/robotology-superbuild/configs/ADVR_shared/bigman/urdf/bigman.urdf'
robot_model = rbdl.loadModel(robot_urdf_file, verbose=False, floating_base=False)
#LH_name = 'LWrMot3'
#RH_name = 'RWrMot3'
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
                           0.50,  0.80,  0.50,  0.50,  0.10,  0.20,   0.03,
                           0.03,  0.03,
                           0.50,  0.80,  0.50,  0.50,  0.10,  0.20,   0.03])
Kp_tau = 100 * pd_tau_weights
Kd_tau = 2 * pd_tau_weights


joint_pos_state = np.zeros(robot_model.q_size)
joint_vel_state = np.zeros(robot_model.qdot_size)
joint_effort_state = np.zeros(robot_model.qdot_size)
joint_stiffness_state = np.zeros(robot_model.qdot_size)
joint_damping_state = np.zeros(robot_model.qdot_size)
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

q_init = np.zeros(robot_model.q_size)
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

q_end = np.zeros(robot_model.q_size)
joints_to_move = bigman_params['joint_ids']['BA'][:7]
#joints_to_move = [bigman_params['joint_ids']['BA'][1]]
q_end[16] = np.deg2rad(90)
q_end[joints_to_move] += np.deg2rad(-20)
N = int(np.ceil(T_traj*freq))
joint_traj, joint_traj_dots, joint_traj_ddots = polynomial5_interpolation(N, q_end, joint_pos_state)
joint_traj_dots *= freq
joint_traj_ddots *= freq*freq
#joint_traj_dots = np.vstack((np.diff(joint_traj, axis=0), np.zeros((1, robot_model.qdot_size))))*freq
#joint_traj_ddots = np.vstack((np.diff(joint_traj_dots, axis=0), np.zeros((1, robot_model.qdot_size))))*freq*freq

tau = np.zeros(robot_model.qdot_size)
a = np.zeros(robot_model.qdot_size)
M = np.zeros((robot_model.qdot_size, robot_model.qdot_size))
des_cmd.name = [bigman_params['joints_names'][idx] for idx in joints_to_move]
des_cmd.position = []
qs_traj = np.zeros((N, robot_model.q_size))
qdots_traj = np.zeros((N, robot_model.q_size))
taus_cmd_traj = np.zeros((N, robot_model.qdot_size))

if load_torques:
    taus_traj = np.load(torques_saved_filename)
else:
    taus_traj = np.zeros((N, robot_model.qdot_size))

print("Moving to the initial configuration of trajectory with torque control.")
raw_input("Press a key to continue...")
for ii in range(N):
    if load_torques:
        print("Reproducing previous torques!")
        des_cmd.effort = taus_traj[ii, joints_to_move]
        des_cmd.stiffness = np.zeros_like(tau[joints_to_move])
        des_cmd.damping = np.zeros_like(tau[joints_to_move])
    else:
        des_cmd.position = joint_traj[ii, joints_to_move]
        des_cmd.stiffness = default_joint_stiffness[joints_to_move]
        des_cmd.damping = default_joint_damping[joints_to_move]
        rbdl.InverseDynamics(robot_model, joint_traj[ii, :], joint_traj_dots[ii, :], joint_traj_ddots[ii, :], tau)
        #taus_traj[ii, :] = joint_effort_state
        #print(joint_traj[ii, joints_to_move] - joint_pos_state[joints_to_move])
        #rbdl.NonlinearEffects(robot_model, joint_pos_state, joint_vel_state, g)
        #rbdl.NonlinearEffects(robot_model, joint_pos_state, joint_vel_state*0, g)
        print(repr(joint_traj_ddots[ii, joints_to_move]))
        print(repr(joint_traj_ddots[ii, joints_to_move]/(freq*freq)))
        print("##")
        print(repr(joint_traj_dots[ii, joints_to_move]))
        print(repr(joint_vel_state[joints_to_move]))
        print("--")
        #a = joint_traj_ddots[ii, :] + \
        #    default_joint_damping*0 * (joint_traj_dots[ii, :] - joint_vel_state) + \
        #    default_joint_stiffness*0.0 * (joint_traj[ii, :] - joint_pos_state)
        pd_tau = Kp_tau * (joint_traj[ii, :] - joint_pos_state) + \
                 Kd_tau * (joint_traj_dots[ii, :] - joint_vel_state)
        #pd_tau = default_joint_stiffness * (joint_traj[ii, :] - joint_pos_state) + \
        #         default_joint_damping * (joint_traj_dots[ii, :] - joint_vel_state)
        tau += pd_tau
        #rbdl.InverseDynamics(robot_model, joint_pos_state, joint_vel_state, a, tau)
        #rbdl.NonlinearEffects(robot_model, joint_traj[ii, :], joint_vel_state*0, tau)
        #tau = np.ones(robot_model.qdot_size)*-0.5
        #a = default_joint_damping * (joint_traj_dots[ii, :] - joint_vel_state)
        #rbdl.CompositeRigidBodyAlgorithm(robot_model, joint_pos_state, M, update_kinematics=True)
        #rbdl.InverseDynamics(robot_model, joint_pos_state, joint_vel_state/freq, joint_traj_ddots[ii, :]/(freq*freq), tau)
        #rbdl.InverseDynamics(robot_model, joint_traj[ii, :], joint_traj_dots[ii, :], joint_traj_ddots[ii, :], tau)
        #tau += M.dot(a)
        print(repr(tau[joints_to_move]))
        print(repr(joint_effort_state[joints_to_move]))
        print("++")
        des_cmd.position = []
        des_cmd.effort = tau[joints_to_move]
        des_cmd.stiffness = np.zeros_like(tau[joints_to_move])
        des_cmd.damping = np.zeros_like(tau[joints_to_move])
    publisher.publish(des_cmd)
    taus_traj[ii, :] = joint_effort_state
    taus_cmd_traj[ii, :] = tau
    qs_traj[ii, :] = joint_pos_state
    qdots_traj[ii, :] = joint_vel_state
    pub_rate.sleep()

# Return to position control
print("Changing to position control!")
for ii in range(50):
    des_cmd.position = joint_pos_state[joints_to_move]
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
#plot_desired_sensed_data(joints_to_plot, joint_traj_dots, qdots_traj, joint_names, data_type='velocity', block=False)
plot_desired_sensed_torque_position(joints_to_plot, taus_cmd_traj, taus_traj,
                                    joint_traj, qs_traj, joint_names, block=True, cols=cols)

print("Saving sensed torques in %s" % torques_saved_filename)
np.save(torques_saved_filename, taus_traj)
sys.exit()
