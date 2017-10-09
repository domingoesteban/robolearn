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
from robolearn.utils.transformations_utils import *
from robolearn.utils.trajectory_interpolators import polynomial5_interpolation
from robolearn.utils.plot_utils import plot_desired_sensed_torque_position
from robolearn.utils.plot_utils import plot_joint_info
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import DeleteModel
from geometry_msgs.msg import Pose
import rbdl

from robolearn.utils.robot_model import RobotModel

np.set_printoptions(precision=4, suppress=True, linewidth=1000)

# Always turn off logger
os.system("gz log -d 0")

#current_path = os.path.abspath(__file__)
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

load_torques = False
torques_saved_filename = 'torques_init_traj.npy'

T_init = 3
T_init_traj = 5
T_impedance_zero = 10
T_sleep = 2.
remove_spawn_new_box = False
freq = 100
box_position = np.array([0.75,
                         0.00,
                         0.0184])
box_size = [0.4, 0.5, 0.3]
box_yaw = 0  # Degrees
box_orient = tf.transformations.rotation_matrix(np.deg2rad(box_yaw), [0, 0, 1])
box_matrix = homogeneous_matrix(rot=box_orient, pos=box_position)

reach_method = 0
lift_method = 2

#traj_files = ['trajectories/traj1'+'_x'+str(box_position[0])+'_y'+str(box_position[1])+'_Y'+str(box_yaw)+'_m'+str(reach_method)+'_reach.npy',
#              'trajectories/traj1'+'_x'+str(box_position[0])+'_y'+str(box_position[1])+'_Y'+str(box_yaw)+'_m'+str(lift_method)+'_lift.npy']
traj_files = ['trajectories/traj1'+'_x'+str(box_position[0])+'_y'+str(box_position[1])+'_Y'+str(box_yaw)+'_m'+str(reach_method)+'_reach.npy']
traj_rep = TrajectoryReproducer(traj_files)

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

# ROBOT MODEL for trying ID
robot_urdf = '/home/domingo/robotology-superbuild/robots/iit-bigman-ros-pkg/bigman_urdf/urdf/bigman.urdf'
robot_model = RobotModel(robot_urdf)
#LH_name = 'LWrMot3'
#RH_name = 'RWrMot3'

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
rospy.init_node('traj_example')
pub_rate = rospy.Rate(freq)
des_cmd = CommandAdvr()
des_cmd.name = bigman_params['joints_names']

q_init = traj_rep.get_data(0)*0
N = int(np.ceil(T_init*freq))
joint_init_traj = polynomial5_interpolation(N, q_init, joint_pos_state)[0]
print("Moving to zero configuration")
for ii in range(N):
    des_cmd.position = joint_init_traj[ii, :]
    des_cmd.stiffness = default_joint_stiffness
    des_cmd.damping = default_joint_damping
    publisher.publish(des_cmd)
    pub_rate.sleep()

q_init = traj_rep.get_data(0)
N = int(np.ceil(T_init_traj*freq))
joint_init_traj = polynomial5_interpolation(N, q_init, joint_pos_state)[0]
joint_init_traj_dots = np.vstack((np.diff(joint_init_traj, axis=0), np.zeros((1, traj_rep.dim))))*freq
joint_init_traj_ddots = np.vstack((np.diff(joint_init_traj_dots, axis=0), np.zeros((1, traj_rep.dim))))*freq*freq

tau = np.zeros(robot_model.qdot_size)
a = np.zeros(robot_model.qdot_size)
M = np.zeros((robot_model.qdot_size, robot_model.qdot_size))
joints_to_move = bigman_params['joint_ids']['BA'][:7]
#joints_to_move = [bigman_params['joint_ids']['BA'][6]]
des_cmd.name = [bigman_params['joints_names'][idx] for idx in joints_to_move]
#raw_input("Press key for moving to the initial configuration of trajectory")
des_cmd.position = []
qs_init_traj = np.zeros((N, robot_model.q_size))
taus_cmd_init_traj = np.zeros((N, robot_model.qdot_size))

if load_torques:
    taus_init_traj = np.load(torques_saved_filename)
else:
    taus_init_traj = np.zeros((N, robot_model.qdot_size))

print("Moving to the initial configuration of trajectory")
raw_input("Press a key to continue...")
for ii in range(N):
    if load_torques:
        print("Reproducing previous torques!")
        des_cmd.effort = taus_init_traj[ii, joints_to_move]
        des_cmd.stiffness = np.zeros_like(tau[joints_to_move])
        des_cmd.damping = np.zeros_like(tau[joints_to_move])
    else:
        #des_cmd.position = joint_init_traj[ii, joints_to_move]
        #des_cmd.stiffness = default_joint_stiffness[joints_to_move]
        #des_cmd.damping = default_joint_damping[joints_to_move]
        #taus_init_traj[ii, :] = joint_effort_state
        #print(joint_init_traj[ii, joints_to_move] - joint_pos_state[joints_to_move])
        #robot_model.update_torque(tau, joint_init_traj[ii, :], joint_init_traj_dots[ii, :]*freq,
        #                          joint_init_traj_ddots[ii, :]*freq*freq)
        #robot_model.update_coriolis_forces(tau, joint_pos_state, joint_vel_state)
        #robot_model.update_coriolis_forces(tau, joint_pos_state, joint_vel_state*0)

        #a = joint_init_traj_ddots[ii, :] + \
        a = default_joint_damping * (joint_init_traj_dots[ii, :] - joint_vel_state) + \
            default_joint_stiffness * (joint_init_traj[ii, :] - joint_pos_state)
        robot_model.update_inertia_matrix(M, joint_pos_state)
        robot_model.update_torque(tau, joint_pos_state, joint_vel_state,
                                  joint_init_traj_ddots[ii, :])
        tau += M.dot(a)
        des_cmd.effort = tau[joints_to_move]
        des_cmd.stiffness = np.zeros_like(tau[joints_to_move])
        des_cmd.damping = np.zeros_like(tau[joints_to_move])
    publisher.publish(des_cmd)
    taus_init_traj[ii, :] = joint_effort_state
    taus_cmd_init_traj[ii, :] = tau
    pub_rate.sleep()
joints_to_plot = bigman_params['joint_ids']['LA']
cols = 3
joint_names = [bigman_params['joints_names'][idx] for idx in joints_to_plot]
plot_desired_sensed_torque_position(joints_to_plot, taus_cmd_init_traj, taus_init_traj,
                                    joint_init_traj, qs_init_traj, joint_names, block=True, cols=cols)

print("Saving sensed torques in %s" % torques_saved_filename)
np.save(torques_saved_filename, taus_init_traj)
sys.exit()

des_cmd.stiffness = []
des_cmd.damping = []

if remove_spawn_new_box:
    f = open('/home/domingo/robotlearning-superbuild/catkin_ws/src/robolearn_gazebo_env/models/cardboard_cube_box/model.sdf', 'r')
    sdf_box = f.read()
    f = open('/home/domingo/robotlearning-superbuild/catkin_ws/src/robolearn_gazebo_env/models/big_support/model.sdf', 'r')
    sdf_box_support = f.read()
    box_pose = Pose()
    box_pose.position.x = box_position[0]
    box_pose.position.y = box_position[1]
    box_pose.position.z = 1.014
    box_quat = tf.transformations.quaternion_from_matrix(box_matrix)
    box_pose.orientation.x = box_quat[0]
    box_pose.orientation.y = box_quat[1]
    box_pose.orientation.z = box_quat[2]
    box_pose.orientation.w = box_quat[3]
    box_support_pose = Pose()
    box_support_pose.position.x = box_position[0]
    box_support_pose.position.y = box_position[1]
    box_support_pose.position.z = 0
    box_support_pose.orientation = box_pose.orientation
    rospy.wait_for_service('gazebo/delete_model')
    delete_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
    print("Deleting previous box...")
    #raw_input("Press for delete box_support")
    try:
        delete_model_prox("box_support")
    except rospy.ServiceException as exc:
        print("/gazebo/delete_model service call failed: %s" % str(exc))
    try:
        delete_model_prox("box")
    except rospy.ServiceException as exc:
        print("/gazebo/delete_model service call failed: %s" % str(exc))
    rospy.wait_for_service('gazebo/spawn_sdf_model')
    spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
    print("Spawning new box...")
    try:
        spawn_model_prox("box_support", sdf_box_support, "box_support", box_support_pose, "world")
    except rospy.ServiceException as exc:
        print("/gazebo/spawn_sdf_model service call failed: %s" % str(exc))
    try:
        spawn_model_prox("box", sdf_box, "box", box_pose, "world")
    except rospy.ServiceException as exc:
        print("/gazebo/spawn_sdf_model service call failed: %s" % str(exc))



qs = traj_rep.traj.copy()
sensed_qs = np.zeros_like(qs)
qdots = np.vstack((np.diff(qs, axis=0), np.zeros((1, traj_rep.dim))))
qddots = np.vstack((np.diff(qdots, axis=0), np.zeros((1, traj_rep.dim))))
taus = np.zeros_like(qdots)
sensed_taus = np.zeros_like(taus)

q = np.zeros(robot_model.q_size)
qdot = np.zeros(robot_model.qdot_size)
qddot = np.zeros(robot_model.qdot_size)
tau = np.zeros(robot_model.qdot_size)

print("Sleeping for %.2f secs" % T_sleep)
rospy.sleep(T_sleep)

print('joint_stiffness = %s' % repr(joint_stiffness_state))
print('joint_damping = %s' % repr(joint_damping_state))
#raw_input()

## Set impedance to zero in order to perform pure torque control
#print("Setting impedance to zero...")
#Timp = 0.5
#for ii in range(int(Timp*freq)):
#    #des_cmd.position = joint_init_traj[ii, :]
#    des_cmd.stiffness = np.zeros_like(qdot)
#    des_cmd.damping = np.zeros_like(qdot)
#    publisher.publish(des_cmd)
#    pub_rate.sleep()

#raw_input("Press key for reproducing trajectory")
print("Reproducing trajectory (ONLY ARMS)...")
joints_to_move = bigman_params['joint_ids']['BA'][5:6]
joints_to_move = [bigman_params['joint_ids']['BA'][6]]
des_cmd.name = [bigman_params['joints_names'][idx] for idx in joints_to_move]
#robot_model.update_torque(tau, joint_pos_state, joint_vel_state, qddot)
robot_model.update_torque(tau, joint_pos_state, qdot, qddot)
print(tau[joints_to_move])
print(joint_effort_state[joints_to_move])
des_cmd.position = []
des_cmd.effort = tau[joints_to_move]
#des_cmd.effort = joint_effort_state[bigman_params['joint_ids']['BA'][:4]]
des_cmd.stiffness = np.zeros_like(default_joint_stiffness[joints_to_move])
des_cmd.damping = np.zeros_like(default_joint_damping[joints_to_move])
qs2 = np.tile(joint_pos_state[:], (T_impedance_zero*freq, 1))
taus2 = np.tile(tau[:], (T_impedance_zero*freq, 1))
sensed_qs2 = np.zeros((T_impedance_zero*freq, robot_model.q_size))
sensed_taus2 = np.zeros((T_impedance_zero*freq, robot_model.qdot_size))

temp_tau = np.zeros((T_impedance_zero*freq, robot_model.q_size))
temp_stiff = np.zeros((T_impedance_zero*freq, robot_model.q_size))
temp_damp = np.zeros((T_impedance_zero*freq, robot_model.q_size))
for qq in range(robot_model.q_size):
    temp_tau[:, qq] = np.linspace(joint_effort_state[qq], 0, T_impedance_zero*freq)
    temp_stiff[:, qq] = np.linspace(default_joint_stiffness[qq], 0, T_impedance_zero*freq)
    temp_damp[:, qq] = np.linspace(default_joint_damping[qq], 0, T_impedance_zero*freq)

os.system("gz log -d 1")
raw_input("Press a key for setting impedance to zero")
#for ii in range(T_impedance_zero*freq):
#    print("Decreasing zero stiffness and damping...")
#    des_cmd.effort = temp_tau[-ii-1, joints_to_move]
#    des_cmd.stiffness = temp_stiff[ii, joints_to_move]
#    des_cmd.damping = temp_damp[ii, joints_to_move]
#    publisher.publish(des_cmd)
#    pub_rate.sleep()
for ii in range(T_impedance_zero*freq):
    print("Sending zero stiffness and damping...")
    robot_model.update_torque(tau, qs[0,:], qdot, qddot)
    #robot_model.update_coriolis_forces(tau, joint_pos_state, joint_vel_state)
    #robot_model.update_coriolis_forces(tau, joint_pos_state, joint_vel_state*0)
    des_cmd.effort = tau[joints_to_move]
    sensed_taus2[ii, ] = joint_effort_state
    sensed_qs2[ii, :] = joint_pos_state
    publisher.publish(des_cmd)
    pub_rate.sleep()
os.system("gz log -d 0")
joints_to_plot = bigman_params['joint_ids']['LA']
cols = 3
joint_names = [bigman_params['joints_names'][idx] for idx in joints_to_plot]
plot_desired_sensed_torque_position(joints_to_plot, taus2, sensed_taus2, qs2, sensed_qs2, joint_names, block=False, cols=cols)
#sys.exit()
raw_input("AA")


des_cmd.position = []
des_cmd.effort = []
des_cmd.stiffness = []
des_cmd.damping = []
raw_input("Press a key for sending commands")
for ii in range(traj_rep.data_points):
#for ii in range(20):
    #print("Sending LIFTING cmd...")
    #error = joint_lift_trajectory[ii, :] - joint_pos_state
    #print(error[bigman_params['joint_ids']['BA']])
    #des_cmd.position += K*error
    q[joints_to_move] = joint_effort_state[joints_to_move]
    qdot[joints_to_move] = qdots[ii, joints_to_move]*freq
    qddot[joints_to_move] = qddots[ii, joints_to_move]*freq*freq
    robot_model.update_torque(tau, q, qdot, qddot)
    taus[ii, :] = tau
    sensed_taus[ii, ] = joint_effort_state
    sensed_qs[ii, :] = joint_pos_state
    print("joint_names: %s" % [bigman_params['joints_names'][id] for id in joints_to_move])
    print("q: %s" % repr(q[joints_to_move]))
    print("tau: %s" % repr(tau[joints_to_move]))
    print("tau_state: %s" % repr(joint_effort_state[joints_to_move]))
    print("--")
    #des_cmd.position = traj_rep.get_data(ii)[bigman_params['joint_ids']['BA'][joints_to_move]]
    #des_cmd.position = q[joints_to_move]
    des_cmd.effort = tau[joints_to_move]
    des_cmd.stiffness = np.zeros_like(qdot[joints_to_move])
    des_cmd.damping = np.zeros_like(qdot[joints_to_move])
    #des_cmd.stiffness = default_joint_stiffness[bigman_params['joint_ids']['BA'][joints_to_move]]
    #des_cmd.damping = default_joint_damping[bigman_params['joint_ids']['BA'][joints_to_move]]
    publisher.publish(des_cmd)
    pub_rate.sleep()

#fig, axs = plt.subplots(dU/cols+1, cols)
#fig.canvas.set_window_title("Positions")
#fig.set_facecolor((0.5, 0.5, 0.5))
#for ii in range(dU):
#    #plt.subplot(dU/cols+1, cols, ii+1)
#    print(ii/cols)
#    print(ii%cols)
#    print("-")
#    axs[ii/cols, ii % cols].set_title("Position %d: %s" % (ii+1, bigman_params['joints_names'][joints_to_plot[ii]]))
#    print(joints_to_plot[ii])
#    print(bigman_params['joints_names'][joints_to_plot[ii]])
#    axs[ii/cols, ii % cols].plot(qs[:, joints_to_plot[ii]], 'b')
#    axs[ii/cols, ii % cols].plot(sensed_qs[:, joints_to_plot[ii]], 'r')
#plt.show(block=False)

joints_to_plot = bigman_params['joint_ids']['LA']
cols = 3
joint_names = [bigman_params['joints_names'][idx] for idx in joints_to_plot]
plot_desired_sensed_torque_position(joints_to_plot, taus, sensed_taus, qs, sensed_qs, joint_names, block=False, cols=cols)
#plot_joint_info(joints_to_plot, taus, joint_names, data='torque')


raw_input("Plotting..")
