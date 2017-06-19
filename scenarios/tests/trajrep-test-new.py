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

# Always turn off logger
os.system("gz log -d 0")

#current_path = os.path.abspath(__file__)
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

load_torques = False
torques_saved_filename = 'torques_init_traj.npy'
joints_to_move = bigman_params['joint_ids']['BA'][:14]
#joints_to_move = [bigman_params['joint_ids']['BA'][6]]
control_mode = 'torque'  # 'position'

T_init = 3
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
#pd_tau_weights = np.array([0.80,  0.50,  0.80,  0.50,  0.50,  0.20,
#                           0.80,  0.50,  0.50,  0.50,  0.50,  0.20,
#                           0.50,  0.80,  0.50,
#                           0.50,  0.80,  0.50,  0.50,  0.10,  0.20,   0.03,
#                           0.03,  0.03,
#                           0.50,  0.80,  0.50,  0.50,  0.10,  0.20,   0.03])
pd_tau_weights = np.array([0.80,  0.50,  0.80,  0.50,  0.50,  0.20,
                           0.80,  0.50,  0.50,  0.50,  0.50,  0.20,
                           0.50,  0.80,  0.50,
                           0.50,  0.80,  0.50,  0.50,  0.10,  0.20,   0.03,
                           0.03,  0.03,
                           0.50,  0.80,  0.50,  0.50,  0.10,  0.20,   0.03])
Kp_tau = 100*pd_tau_weights
#Kd_tau = 2 * np.sqrt(Kp_tau)
Kd_tau = 2 * pd_tau_weights

#Kp_tau = np.array([80,  50,  80,  50,  50,  20,
#                   80,  50,  50,  50,  50,  20,
#                   50,  80,  50,
#                   50,  80,  50,  50,  10,  20,   300,
#                   3,  3,
#                   50,  80,  50,  50,  10,  20,   300])
#Kd_tau = np.array([1.60,  1.00,  1.60,  1.00,  1.00,  0.40,
#                   1.60,  1.00,  1.00,  1.00,  1.00,  0.40,
#                   1.00,  1.60,  1.00,
#                   1.00,  1.60,  1.00,  1.00,  0.20,  0.40,   0.00,
#                   0.06,  0.06,
#                   1.00,  1.60,  1.00,  1.00,  0.20,  0.40,   0.00])
#Kd_tau = default_joint_damping

#Kp_tau = 100 * default_joint_stiffness/1000
#Kd_tau = 2 * default_joint_damping/100

# ROBOT MODEL for trying ID
robot_urdf_file = '/home/domingo/robotology-superbuild/robots/iit-bigman-ros-pkg/bigman_urdf/urdf/bigman.urdf'
robot_urdf_file = '/home/domingo/robotology-superbuild/configs/ADVR_shared/bigman/urdf/bigman.urdf'
robot_model = RobotModel(robot_urdf_file)
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

Ninit = int(np.ceil(T_init*freq))
joint_init_traj = polynomial5_interpolation(Ninit, traj_rep.get_data(0), joint_pos_state)[0]

print("Moving to initial configuration with Position control.")
for ii in range(Ninit):
    des_cmd.position = joint_init_traj[ii, :]
    des_cmd.stiffness = default_joint_stiffness
    des_cmd.damping = default_joint_damping
    publisher.publish(des_cmd)
    pub_rate.sleep()



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


N = traj_rep.data_points

tau = np.zeros(robot_model.qdot_size)
a = np.zeros(robot_model.qdot_size)
M = np.zeros((robot_model.qdot_size, robot_model.qdot_size))
des_cmd.name = [bigman_params['joints_names'][idx] for idx in joints_to_move]

qs = traj_rep.traj
qdots = np.vstack((np.diff(qs, axis=0), np.zeros((1, robot_model.qdot_size))))*freq
qddots = np.vstack((np.diff(qdots, axis=0), np.zeros((1, robot_model.qdot_size))))*freq*freq
taus = np.zeros((N, robot_model.qdot_size))
sensed_taus = np.zeros((N, robot_model.qdot_size))
sensed_qs = np.zeros((N, robot_model.q_size))
sensed_qdots = np.zeros((N, robot_model.qdot_size))

des_cmd.position = []
des_cmd.effort = []
des_cmd.stiffness = []
des_cmd.damping = []
#raw_input("Press a key for sending %s commands" % control_mode.lower())
for ii in range(N):
    print("Sending LIFTING %s cmd %d/%d..." % (control_mode.lower(), ii+1, N))
    if control_mode.lower() == 'position':
        des_cmd.position = qs[ii, joints_to_move]
        des_cmd.stiffness = default_joint_stiffness[joints_to_move]
        des_cmd.damping = default_joint_damping[joints_to_move]

    elif control_mode.lower() == 'torque':
        #rbdl.InverseDynamics(robot_model.model, qs[ii, :], qdots[ii, :], qddots[ii, :], tau)
        #taus_traj[ii, :] = joint_effort_state
        #print(joint_traj[ii, joints_to_move] - joint_pos_state[joints_to_move])
        #rbdl.NonlinearEffects(robot_model, joint_pos_state, joint_vel_state, g)
        #rbdl.NonlinearEffects(robot_model, joint_pos_state, joint_vel_state*0, g)
        #a = joint_traj_ddots[ii, :] + \
        #    default_joint_damping*0 * (joint_traj_dots[ii, :] - joint_vel_state) + \
        #    default_joint_stiffness*0.0 * (joint_traj[ii, :] - joint_pos_state)

        # Computed Torque Control
        a = qddots[ii, :] + \
            Kd_tau * (qdots[ii, :] - joint_vel_state) + \
            Kp_tau * (qs[ii, :] - joint_pos_state)
        robot_model.update_torque(tau, joint_pos_state, joint_vel_state, a)

        ## FeedForward + PD compensation
        #robot_model.update_torque(tau, qs[ii, :], qdots[ii, :], qddots[ii, :])
        #pd_tau = Kp_tau * (qs[ii, :] - joint_pos_state) + \
        #         Kd_tau * (qdots[ii, :] - joint_vel_state)
        #tau += pd_tau


        #pd_tau = default_joint_stiffness * (qs[ii, :] - joint_pos_state) + \
        #         default_joint_damping * (qdots[ii, :] - joint_vel_state)
        #rbdl.InverseDynamics(robot_model, joint_pos_state, joint_vel_state, a, tau)
        #rbdl.NonlinearEffects(robot_model, qs[ii, :], joint_vel_state*0, tau)
        #tau = np.ones(robot_model.qdot_size)*-0.5
        #a = default_joint_damping * (joint_traj_dots[ii, :] - joint_vel_state)
        #rbdl.CompositeRigidBodyAlgorithm(robot_model, joint_pos_state, M, update_kinematics=True)
        #rbdl.InverseDynamics(robot_model, joint_pos_state, joint_vel_state/freq, qddots[ii, :]/(freq*freq), tau)
        #rbdl.InverseDynamics(robot_model, qs[ii, :], qdots[ii, :], qddots[ii, :], tau)
        #tau += M.dot(a)

        des_cmd.position = []
        des_cmd.effort = tau[joints_to_move]
        if ii <= 100:
            des_cmd.stiffness = np.zeros_like(tau[joints_to_move])
            des_cmd.damping = np.zeros_like(tau[joints_to_move])
        else:
            des_cmd.stiffness = []
            des_cmd.damping = []
    else:
        raise ValueError("Wrong control mode option: %s" % control_mode)

    publisher.publish(des_cmd)
    sensed_taus[ii, :] = joint_effort_state
    taus[ii, :] = tau
    sensed_qs[ii, :] = joint_pos_state
    sensed_qdots[ii, :] = joint_vel_state
    pub_rate.sleep()

# Return to position control
print("Changing to position control!")
for ii in range(50):
    des_cmd.position = joint_pos_state[joints_to_move]
    des_cmd.stiffness = default_joint_stiffness[joints_to_move]
    des_cmd.damping = default_joint_damping[joints_to_move]
    publisher.publish(des_cmd)
    pub_rate.sleep()


joints_to_plot = bigman_params['joint_ids']['LA']
cols = 3
joint_names = [bigman_params['joints_names'][idx] for idx in joints_to_plot]
print("Ploting...")
#plot_desired_sensed_data(joints_to_plot, qs, sensed_qs, joint_names, data_type='position', block=False)
#plot_desired_sensed_data(joints_to_plot, qdots, sensed_qdots, joint_names, data_type='velocity', block=False)
plot_desired_sensed_torque_position(joints_to_plot, taus, sensed_taus,
                                    qs, sensed_qs, joint_names, block=True, cols=cols)
raw_input("Press a key to finish the script..")
