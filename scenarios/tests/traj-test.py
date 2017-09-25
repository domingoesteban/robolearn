import os
import numpy as np
import matplotlib.pyplot as plt
from robolearn.utils.iit.iit_robots_params import *
from robolearn.utils.trajectory_interpolators import polynomial5_interpolation
from robolearn.utils.trajectory_interpolators import spline_interpolation
from robolearn.utils.trajectory_interpolators import quaternion_slerp_interpolation
from robolearn.utils.robot_model import *
from robolearn.utils.iit.robot_poses.bigman.poses import *
from robolearn.utils.transformations import *
from std_srvs.srv import Empty
from robolearn.utils.reach_drill_utils import create_drill_relative_pose, reset_bigman_drill_gazebo, create_hand_relative_pose, spawn_drill_gazebo
import rospy
import tf
from XCM.msg import CommandAdvr
from XCM.msg import JointStateAdvr
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import DeleteModel
from geometry_msgs.msg import Pose

np.set_printoptions(precision=4, suppress=True, linewidth=1000)

# Script parameters
#box_position = np.array([0.75, 0.0, 0.0184])
box_position = np.array([0.75-0.05,
                         0.00,
                         0.0184])
box_position = np.array([0.64, -0.03-0.3, 0.0173+0.2])  # drill
box_position = np.array([0.64, 0., -0.1327])  # beer
box_position = np.array([0.64, 0., -0.1327])  # beer relative

box_position = create_drill_relative_pose(drill_x=0.86, drill_y=-0.1776-0.05, drill_z=-0.1327+0.17, drill_yaw=0)[-3:]

box_size = [0.4, 0.5, 0.3]
box_size = [0.1, 0.1, 0.3]  # drill
box_size = [0.11, 0.11, 0.3]  # beer
box_yaw = 0  # Degrees
#box_orient = tf.transformations.rotation_matrix(np.deg2rad(15), [1, 0, 0])  # For the EEs is rotation in X
box_orient = tf.transformations.rotation_matrix(np.deg2rad(box_yaw), [0, 0, 1])
box_matrix = homogeneous_matrix(rot=box_orient, pos=box_position)
freq = 100
T_init = 2
T_reach = 6
T_lift = 1

# Save/Load file name
file_name = 'trajectories/traj1'+'_x'+str(box_position[0])+'_y'+str(box_position[1])+'_Y'+str(box_yaw)
load_reach_traj = False
load_lift_traj = False
#load_reach_traj = True
#load_lift_traj = True
save_reach_traj = False
save_lift_traj = False

remove_spawn_new_box = False

reach_option = 3
#reach_option 0: IK desired final pose, interpolate in joint space
#reach_option 1: Trajectory in EEs, then IK whole trajectory
#reach_option 2: Trajectory in EEs, IK with Jacobians

lift_option = 0
#lift_option 0: IK desired final pose, interpolate the others
#lift_option 1: Trajectory in EEs, then IK whole trajectory
#lift_option 2: Trajectory in EEs, IK with Jacobians

regularization_parameter = 0.01  # For IK optimization algorithm


q_init = np.zeros(31)
# q_init[16] = np.deg2rad(50)
# q_init[25] = np.deg2rad(-50)
#q_init[24] = np.deg2rad(20)
#q_init[25] = np.deg2rad(-35)
#q_init[26] = np.deg2rad(0)
#q_init[27] = np.deg2rad(-95)
#q_init[28] = np.deg2rad(0)
#q_init[29] = np.deg2rad(0)
#q_init[30] = np.deg2rad(0)
q_init[24] = np.deg2rad(-30)
q_init[25] = np.deg2rad(-65)
q_init[26] = np.deg2rad(20)
q_init[27] = np.deg2rad(-95)
q_init[28] = np.deg2rad(20)
q_init[29] = np.deg2rad(0)
q_init[30] = np.deg2rad(0)



print(q_init[24:])
print(bigman_params['joints_limits'][24:])
#q_init = np.deg2rad(np.array(bigman_Apose))
#q_init = np.deg2rad(np.array(bigman_Fpose))
#q_init = np.deg2rad(np.array(bigman_Tpose))

# Robot Model
robot_urdf = '/home/domingo/robotology-superbuild/robots/iit-bigman-ros-pkg/bigman_urdf/urdf/bigman.urdf'
robot_urdf = os.environ["ROBOTOLOGY_ROOT"]+'/robots/iit-bigman-ros-pkg/bigman_urdf/urdf/bigman.urdf'
robot_model = RobotModel(robot_urdf)
LH_name = 'LWrMot3'
RH_name = 'RWrMot3'
l_soft_hand_offset = np.array([0.000, -0.030, -0.210])
r_soft_hand_offset = np.array([0.000, 0.030, -0.210])
left_sign = np.array([1, 1, 1, 1, 1, 1, 1])
right_sign = np.array([1, -1, -1, 1, -1, 1, -1])

actual_RH_pose = robot_model.fk(RH_name, q=q_init, body_offset=r_soft_hand_offset, update_kinematics=True)
#desired_RH_reach_pose = polynomial5_interpolation(N, RH_reach_pose, actual_RH_pose)[0]
desired_RH_pose = actual_RH_pose.copy()
desired_RH_pose[-3] -= 0.01
desired_RH_pose[-2] += 0.10
desired_RH_pose[-1] -= 0.12

drill_x = 0.70
drill_y = 0.00
drill_z = -0.1327
drill_yaw = 0  # Degrees
drill_pose3 = create_drill_relative_pose(drill_x=drill_x+0.16, drill_y=drill_y-0.2276, drill_z=drill_z, drill_yaw=drill_yaw)
drill_size = [0.1, 0.1, 0.3]
hand_y = -drill_size[1]/2-0.02
hand_z = drill_size[2]/2+0.02
desired_RH_pose = create_hand_relative_pose(drill_pose3, hand_x=0.0, hand_y=hand_y, hand_z=hand_z, hand_yaw=0)

torso_joints = bigman_params['joint_ids']['TO']
q_reach2 = robot_model.ik(RH_name, desired_RH_pose, body_offset=r_soft_hand_offset,
                          mask_joints=torso_joints, joints_limits=bigman_params['joints_limits'],
                          method='optimization')

print(repr(actual_RH_pose))
print(repr(desired_RH_pose))
print(np.rad2deg(q_reach2))
raw_input("BORRAME")


# ###########
# REACH BOX #
# ###########
if not load_reach_traj:
    print("\033[5mGenerating reaching trajectory...")
    ## Orientation
    ##des_orient = homogeneous_matrix(rot=rot)
    #des_orient = tf.transformations.rotation_matrix(np.deg2rad(-90), [0, 1, 0])
    ###des_orient = des_orient.dot(tf.transformations.rotation_matrix(np.deg2rad(-5), [1, 0, 0]))
    ##des_orient = des_orient.dot(tf.transformations.rotation_matrix(np.deg2rad(-3), [1, 0, 0]))
    ##des_orient = des_orient.dot(tf.transformations.rotation_matrix(np.deg2rad(5), [0, 0, 1]))
    ##des_orient = des_orient.dot(tf.transformations.rotation_matrix(np.deg2rad(10), [0, 1, 0]))
    #des_orient = des_orient.dot(box_orient)
    box_LH_position = np.array([0.00,
                                box_size[1]/2. - 0.00,
                                -0.05])
    box_LH_matrix = homogeneous_matrix(pos=box_LH_position)
    LH_reach_matrix = box_matrix.dot(box_LH_matrix)
    LH_reach_matrix = LH_reach_matrix.dot(tf.transformations.rotation_matrix(np.deg2rad(-90), [0, 1, 0]))
    LH_reach_pose = np.zeros(7)
    LH_reach_pose[4:] = tf.transformations.translation_from_matrix(LH_reach_matrix)
    LH_reach_pose[:4] = tf.transformations.quaternion_from_matrix(LH_reach_matrix)

    box_RH_position = np.array([0.00,
                                -box_size[1]/2. + 0.00,
                                -0.05])
    box_RH_matrix = homogeneous_matrix(pos=box_RH_position)
    RH_reach_matrix = box_matrix.dot(box_RH_matrix)
    # Rotate HAND
    RH_reach_matrix = RH_reach_matrix.dot(tf.transformations.rotation_matrix(np.deg2rad(-90), [0, 1, 0]))
    RH_reach_pose = np.zeros(7)
    RH_reach_pose[4:] = tf.transformations.translation_from_matrix(RH_reach_matrix)
    RH_reach_pose[:4] = tf.transformations.quaternion_from_matrix(RH_reach_matrix)

    N = int(np.ceil(T_reach*freq))
    torso_joints = bigman_params['joint_ids']['TO']

    if reach_option == 0:
        q_reach = robot_model.ik(LH_name, LH_reach_pose, body_offset=l_soft_hand_offset,
                                 mask_joints=torso_joints, joints_limits=bigman_params['joints_limits'],
                                 method='optimization')
        print("TODO: NOT MOVING LEFT ARM")
        q_reach = q_init.copy()
        q_reach2 = robot_model.ik(RH_name, RH_reach_pose, body_offset=r_soft_hand_offset,
                                  mask_joints=torso_joints, joints_limits=bigman_params['joints_limits'],
                                  method='optimization')

        q_reach[bigman_params['joint_ids']['RA']] = q_reach2[bigman_params['joint_ids']['RA']]

        # Trajectory
        joint_reach_trajectory = polynomial5_interpolation(N, q_reach, q_init)[0]

        print("TODO: UNCOMMENT BELOW")
        # save_reach_traj = raw_input("Save reach trajectory before visualize? (y/yes): ")
        # if save_reach_traj.lower() in ['y', 'yes']:
        #     np.save(file_name + '_m' + str(reach_option) + '_reach.npy', joint_reach_trajectory)

    elif reach_option == 1:
        q = q_init.copy()
        actual_LH_pose = robot_model.fk(LH_name, q=q, body_offset=l_soft_hand_offset, update_kinematics=True)
        actual_RH_pose = robot_model.fk(RH_name, q=q, body_offset=r_soft_hand_offset, update_kinematics=True)
        desired_LH_reach_pose = polynomial5_interpolation(N, LH_reach_pose, actual_LH_pose)[0]
        desired_RH_reach_pose = polynomial5_interpolation(N, RH_reach_pose, actual_RH_pose)[0]

        viapoint_LH_reach = np.empty(3)

        quatLH_interpolation = quaternion_slerp_interpolation(N, LH_reach_pose[:4], actual_LH_pose[:4])
        quatRH_interpolation = quaternion_slerp_interpolation(N, RH_reach_pose[:4], actual_RH_pose[:4])
        desired_LH_reach_pose[:, :4] = quatLH_interpolation
        desired_RH_reach_pose[:, :4] = quatRH_interpolation

        joint_reach_trajectory = np.zeros((desired_LH_reach_pose.shape[0], robot_model.q_size))
        joint_reach_trajectory[0, :] = q

        q_reach = np.empty(robot_model.q_size)
        q_reach2 = np.empty(robot_model.q_size)
        for ii in range(desired_LH_reach_pose.shape[0]-1):
            print("%d/%d " % (ii+1, N))
            #print("%d/%d " % (ii+1, N))
            q_reach[:] = robot_model.ik(LH_name, desired_LH_reach_pose[ii+1, :], body_offset=l_soft_hand_offset,
                                     q_init=joint_reach_trajectory[ii, :], joints_limits=bigman_params['joints_limits'],
                                     mask_joints=torso_joints, method='optimization',
                                     regularization_parameter=regularization_parameter)
            q_reach2[:] = robot_model.ik(RH_name, desired_RH_reach_pose[ii+1, :], body_offset=r_soft_hand_offset,
                                      q_init=joint_reach_trajectory[ii, :], joints_limits=bigman_params['joints_limits'],
                                      mask_joints=torso_joints, method='optimization',
                                      regularization_parameter=regularization_parameter)
            q_reach[bigman_params['joint_ids']['RA']] = q_reach2[bigman_params['joint_ids']['RA']]
            joint_reach_trajectory[ii+1, :] = q_reach
            #print(joint_reach_trajectory[ii+1, :]-joint_reach_trajectory[ii, :])
            print(sum(joint_reach_trajectory[ii+1, :]-joint_reach_trajectory[ii, :]))

        save_reach_traj = raw_input("Save reach trajectory before visualize? (y/yes): ")
        if save_reach_traj.lower() in ['y', 'yes']:
            np.save(file_name + '_m' + str(reach_option) + '_reach.npy', joint_reach_trajectory)

    elif reach_option == 2:
        q = q_init.copy()
        actual_LH_pose = robot_model.fk(LH_name, q=q, body_offset=l_soft_hand_offset, update_kinematics=True)
        actual_RH_pose = robot_model.fk(RH_name, q=q, body_offset=r_soft_hand_offset, update_kinematics=True)

        print("TODO: TEMPORALY CHANGING DESIRED LH/RH POSES")
        LH_reach_pose = actual_LH_pose
        RH_reach_pose = desired_RH_pose

        desired_LH_reach_pose = polynomial5_interpolation(N, LH_reach_pose, actual_LH_pose)[0]
        desired_RH_reach_pose = polynomial5_interpolation(N, RH_reach_pose, actual_RH_pose)[0]

        quatLH_interpolation = quaternion_slerp_interpolation(N, LH_reach_pose[:4], actual_LH_pose[:4])[0]
        quatRH_interpolation = quaternion_slerp_interpolation(N, RH_reach_pose[:4], actual_RH_pose[:4])[0]
        desired_LH_reach_pose[:, :4] = quatLH_interpolation
        desired_RH_reach_pose[:, :4] = quatRH_interpolation

        #for ii in range(desired_LH_reach_pose.shape[0]):
        #    print(desired_LH_reach_pose[ii, 4])
        #raw_input("CUCU")

        q_reach = robot_model.ik(LH_name, LH_reach_pose, body_offset=l_soft_hand_offset,
                                 mask_joints=torso_joints, joints_limits=bigman_params['joints_limits'],
                                 method='optimization')
        q_reach2 = robot_model.ik(RH_name, RH_reach_pose, body_offset=r_soft_hand_offset,
                                  mask_joints=torso_joints, joints_limits=bigman_params['joints_limits'],
                                  method='optimization')
        q_reach[bigman_params['joint_ids']['RA']] = q_reach2[bigman_params['joint_ids']['RA']]

        J1 = np.zeros((6, robot_model.qdot_size))
        J2 = np.zeros((6, robot_model.qdot_size))
        K = 500
    elif reach_option == 3:
        pass
    else:
        raise ValueError("Wrong reach_option %d" % reach_option)
    print("\033[31mDONE!! \033[0m")

    #RH_reach_pose = robot_model.fk(RH_name, q=np.zeros(robot_model.q_size), body_offset=r_soft_hand_offset)
    #RH_reach_pose[4:] = LH_reach_pose[4:]
    #RH_reach_pose[5] = box_position[1] - box_size[1]/2. + 0.02
    #des_orient = tf.transformations.rotation_matrix(np.deg2rad(-90), [0, 1, 0])
    #des_orient = des_orient.dot(box_orient)
    #RH_reach_pose[:4] = tf.transformations.quaternion_from_matrix(des_orient)

else:
    print("\n\033[5mLoading reaching trajectory...")
    joint_reach_trajectory = np.load(file_name + '_m' + str(reach_option) + '_reach.npy')
    if reach_option == 2:
        q_reach = joint_reach_trajectory[-1, :]
        J1 = np.zeros((6, robot_model.qdot_size))
        J2 = np.zeros((6, robot_model.qdot_size))
        desired_LH_reach_pose = np.load(file_name+'_reach_LH_EE.npy')
        desired_RH_reach_pose = np.load(file_name+'_reach_RH_EE.npy')
    print("\033[31mDONE!! \033[0m")



# ######## #
# LIFT BOX #
# ######## #
"""
if not load_lift_traj:
    print("\033[5mGenerating lifting trajectory...")
    LH_lift_pose = LH_reach_pose.copy()
    LH_lift_pose[6] += 0.3
    RH_lift_pose = RH_reach_pose.copy()
    RH_lift_pose[6] += 0.3

    N = int(np.ceil(T_lift*freq))

    #final_LH_lift_pose = actual_LH_lift_pose.copy()
    #final_LH_lift_pose[-1] += 0.3
    #final_LH_lift_pose[-2] -= 0.005
    #final_RH_lift_pose = actual_RH_lift_pose.copy()
    #final_RH_lift_pose[-1] += 0.3
    ##des_orient = des_orient.dot(tf.transformations.rotation_matrix(np.deg2rad(-4), [1, 0, 0]))
    #des_orient = des_orient.dot(tf.transformations.rotation_matrix(np.deg2rad(-9), [1, 0, 0]))
    #des_orient = des_orient.dot(tf.transformations.rotation_matrix(np.deg2rad(-3), [0, 0, 1]))
    #final_LH_lift_pose[:4] = tf.transformations.quaternion_from_matrix(des_orient)

    if lift_option == 0:
        q_lift = robot_model.ik(LH_name, LH_lift_pose, body_offset=l_soft_hand_offset,
                                 mask_joints=torso_joints, joints_limits=bigman_params['joints_limits'],
                                 method='optimization')
        q_lift2 = robot_model.ik(RH_name, RH_lift_pose, body_offset=r_soft_hand_offset,
                                  mask_joints=torso_joints, joints_limits=bigman_params['joints_limits'],
                                  method='optimization')
        q_lift[bigman_params['joint_ids']['RA']] = q_lift2[bigman_params['joint_ids']['RA']]
        joint_lift_trajectory = polynomial5_interpolation(N, q_lift, q_reach)[0]

        #if save_lift_traj:
        #    np.save(file_name+'_lift.npy', joint_lift_trajectory)
        print("TODO: UNCOMMENT BELOW")
        # save_lift_traj = raw_input("Save lift trajectory before visualize? (y/yes): ")
        # if save_lift_traj.lower() in ['y', 'yes']:
        #     np.save(file_name + '_m' + str(lift_option) + '_lift.npy', joint_lift_trajectory)

    elif lift_option == 1:
        q = q_reach.copy()
        actual_LH_pose = robot_model.fk(LH_name, q=q, body_offset=l_soft_hand_offset, update_kinematics=True)
        actual_RH_pose = robot_model.fk(RH_name, q=q, body_offset=r_soft_hand_offset, update_kinematics=True)
        desired_LH_lift_pose = polynomial5_interpolation(N, LH_lift_pose, actual_LH_pose)[0]
        desired_RH_lift_pose = polynomial5_interpolation(N, RH_lift_pose, actual_RH_pose)[0]

        joint_lift_trajectory = np.zeros((N, robot_model.q_size))
        joint_lift_trajectory[0, :] = q
        q_lift = np.empty(robot_model.q_size)
        q_lift2 = np.empty(robot_model.q_size)
        for ii in range(N-1):
            print("%d/%d " % (ii+1, N))
            #print("%d/%d " % (ii+1, N))
            q_lift[:] = robot_model.ik(LH_name, desired_LH_lift_pose[ii+1, :], body_offset=l_soft_hand_offset,
                                       q_init=joint_lift_trajectory[ii, :], joints_limits=bigman_params['joints_limits'],
                                       mask_joints=torso_joints, method='optimization',
                                       regularization_parameter=regularization_parameter)
            q_lift2[:] = robot_model.ik(RH_name, desired_RH_lift_pose[ii+1, :], body_offset=r_soft_hand_offset,
                                        q_init=joint_lift_trajectory[ii, :], joints_limits=bigman_params['joints_limits'],
                                        mask_joints=torso_joints, method='optimization',
                                        regularization_parameter=regularization_parameter)
            q_lift[bigman_params['joint_ids']['RA']] = q_lift2[bigman_params['joint_ids']['RA']]
            joint_lift_trajectory[ii+1, :] = q_lift

        #if save_lift_traj:
        #    np.save(file_name+'_lift.npy', joint_lift_trajectory)
        print("TODO: UNCOMMENT THE BELOW")
        # save_lift_traj = raw_input("Save lift trajectory before visualize? (y/yes): ")
        # if save_lift_traj.lower() in ['y', 'yes']:
        #     np.save(file_name + '_m' + str(lift_option) + '_lift.npy', joint_lift_trajectory)

    elif lift_option == 2:
        T_lift = 2
        N = int(np.ceil(T_lift*freq))

        q = q_reach.copy()
        actual_LH_pose = robot_model.fk(LH_name, q=q, body_offset=l_soft_hand_offset, update_kinematics=True)
        actual_RH_pose = robot_model.fk(RH_name, q=q, body_offset=r_soft_hand_offset, update_kinematics=True)
        desired_LH_lift_pose = polynomial5_interpolation(N, LH_lift_pose, actual_LH_pose)[0]
        desired_RH_lift_pose = polynomial5_interpolation(N, RH_lift_pose, actual_RH_pose)[0]

        J1 = np.zeros((6, robot_model.qdot_size))
        J2 = np.zeros((6, robot_model.qdot_size))
        K = 500
    else:
        raise ValueError("Wrong lift_option %d" % lift_option)
    print("\n\033[31mDONE!! \033[0m")

else:
    print("\n\033[5mLoading lifting trajectory...")
    joint_lift_trajectory = np.load(file_name + '_m' + str(lift_option) + '_lift.npy')
    if lift_option == 2:
        J1 = np.zeros((6, robot_model.qdot_size))
        J2 = np.zeros((6, robot_model.qdot_size))
        desired_LH_lift_pose = np.load(file_name+'_lift_LH_EE.npy')
        desired_RH_lift_pose = np.load(file_name+'_lift_RH_EE.npy')
    print("\033[31mDONE!! \033[0m")
"""

print("Waiting for ROS..."),
while rospy.is_shutdown():
    pass
print("ROS OK")
# ROS Stuff
#raw_input("Press for ROS related stuff")
joint_state = np.zeros(robot_model.q_size)
joint_state_id = []
def callback(data, params):
    joint_state = params[0]
    joint_state_id = params[1]
    if not joint_state_id:
        joint_state_id[:] = [bigman_params['joints_names'].index(name) for name in data.name]
    joint_state[joint_state_id] = data.link_position
    #print params[0]
    #rospy.loginfo("I heard %s", data.data)
publisher = rospy.Publisher("/xbotcore/bigman/command", CommandAdvr, queue_size=10)
subscriber = rospy.Subscriber("/xbotcore/bigman/joint_states", JointStateAdvr, callback, (joint_state, joint_state_id))
rospy.init_node('traj_example')
pub_rate = rospy.Rate(freq)
des_cmd = CommandAdvr()
des_cmd.name = bigman_params['joints_names']

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

# ##################### #
# INITIAL CONFIGURATION #
# ##################### #
N = int(np.ceil(T_init*freq))
print(q_init)
joint_init_traj = polynomial5_interpolation(N, q_init, joint_state)[0]
raw_input("Press key for moving to INIT")
for ii in range(N):
    des_cmd.position = joint_init_traj[ii, :]
    publisher.publish(des_cmd)
    pub_rate.sleep()

# RESETING DRILL
print("Reset drill")
#bigman_drill_pose = create_drill_relative_pose(drill_x=0.86, drill_y=-0.1776-0.05, drill_z=-0.1327+0.15, drill_yaw=0)
bigman_drill_pose = create_drill_relative_pose(drill_x=drill_x+0.16, drill_y=drill_y-0.2276, drill_z=drill_z+0.15, drill_yaw=drill_yaw)
#spawn_drill_gazebo(bigman_drill_pose, drill_size=drill_size)
#raw_input('press a key')
print(bigman_drill_pose)
print('+'*10)
#reset_bigman_drill_gazebo(bigman_drill_pose, drill_size=None)

#temp_count = 0
#des_cmd.position = q_init
#while temp_count < 100:
#    publisher.publish(des_cmd)
#    pub_rate.sleep()
#    temp_count += 1
# rospy.wait_for_service('/gazebo/reset_world')
# reset_srv = rospy.ServiceProxy('/gazebo/reset_world', Empty)
# try:
#     reset_srv()
# except rospy.ServiceException as exc:
#     print("/gazebo/reset_world service call failed: %s" % str(exc))

# Send Commands
raw_input("Press key for REACHING")
#K = 0.03
#des_cmd.position = joint_state.copy()
if reach_option == 0 or reach_option == 1:
    for ii in range(joint_reach_trajectory.shape[0]):
        #print("Sending REACHING cmd...")
        #error = joint_trajectory[ii, :] - joint_state
        #print(error[bigman_params['joint_ids']['BA']])
        #des_cmd.position += K*error
        des_cmd.position = joint_reach_trajectory[ii, :]
        publisher.publish(des_cmd)
        pub_rate.sleep()

elif reach_option == 2:
    joint_reach_trajectory = np.empty((desired_LH_reach_pose.shape[0], robot_model.q_size))
    q = q_init.copy()
    joint_reach_trajectory[0, :] = q[:]
    for ii in range(desired_LH_reach_pose.shape[0]-1):
        #for ii in range(N-1):
        print("Sending REACHING cmd...")
        #error1 = compute_cartesian_error(desired_LH_lift_pose[ii, :], actual_LH_lift_pose, rotation_rep='quat')
        #error2 = compute_cartesian_error(desired_RH_lift_pose[ii, :], actual_RH_lift_pose, rotation_rep='quat')

        xdot = compute_cartesian_error(desired_LH_reach_pose[ii+1, :], desired_LH_reach_pose[ii, :], rotation_rep='quat')#error1
        xdot2 = compute_cartesian_error(desired_RH_reach_pose[ii+1, :], desired_RH_reach_pose[ii, :], rotation_rep='quat')#error1

        # Compute the jacobian matrix
        #rbdl.CalcPointJacobian6D(robot_model.model, q, model.GetBodyId(LH_name), np.zeros(0), J1, True)
        robot_model.update_jacobian(J1, LH_name, q=q, body_offset=l_soft_hand_offset, update_kinematics=True)
        robot_model.update_jacobian(J2, RH_name, q=q, body_offset=r_soft_hand_offset, update_kinematics=True)
        J1[:, bigman_params['joint_ids']['TO']] = 0
        J2[:, bigman_params['joint_ids']['TO']] = 0
        #print(J1)

        qdot = np.linalg.lstsq(J1, xdot)[0]
        qdot2 = np.linalg.lstsq(J2, xdot2)[0]

        #qdot[bigman_params['joint_ids']['RA']] = qdot[bigman_params['joint_ids']['LA']]*right_sign
        qdot[bigman_params['joint_ids']['RA']] = qdot2[bigman_params['joint_ids']['RA']]

        q[:] += qdot

        actual_LH_pose = robot_model.fk(LH_name, q=q, body_offset=l_soft_hand_offset, update_kinematics=True)
        actual_RH_pose = robot_model.fk(RH_name, q=q, body_offset=r_soft_hand_offset, update_kinematics=True)

        des_cmd.position = q
        publisher.publish(des_cmd)

        joint_reach_trajectory[ii+1, :] = q[:]

        pub_rate.sleep()

elif reach_option == 3:
    N = int(np.ceil(T_reach*freq))
    q_reach2 = robot_model.ik(RH_name, desired_RH_pose, body_offset=r_soft_hand_offset,
                              mask_joints=torso_joints, joints_limits=bigman_params['joints_limits'],
                              method='optimization')
    joint_init_traj = polynomial5_interpolation(N, q_reach2, joint_state)[0]
    raw_input("Press key for moving to INIT")
    for ii in range(N):
        des_cmd.position = joint_init_traj[ii, :]
        publisher.publish(des_cmd)
        pub_rate.sleep()


raw_input("Press key for LIFTING")
if lift_option == 0 or lift_option == 1:
    for ii in range(joint_lift_trajectory.shape[0]):
        #print("Sending LIFTING cmd...")
        #error = joint_lift_trajectory[ii, :] - joint_state
        #print(error[bigman_params['joint_ids']['BA']])
        #des_cmd.position += K*error
        des_cmd.position = joint_lift_trajectory[ii, :]
        publisher.publish(des_cmd)
        pub_rate.sleep()

elif lift_option == 2:
    joint_lift_trajectory = np.empty((desired_LH_lift_pose.shape[0], robot_model.q_size))
    q = q_reach.copy()
    joint_lift_trajectory[0, :] = q[:]
    for ii in range(desired_LH_lift_pose.shape[0]-1):
    #for ii in range(N-1):
        print("Sending LIFTING cmd...")
        #error1 = compute_cartesian_error(desired_LH_lift_pose[ii, :], actual_LH_lift_pose, rotation_rep='quat')
        #error2 = compute_cartesian_error(desired_RH_lift_pose[ii, :], actual_RH_lift_pose, rotation_rep='quat')

        xdot = compute_cartesian_error(desired_LH_lift_pose[ii+1, :], desired_LH_lift_pose[ii, :], rotation_rep='quat')#error1
        xdot2 = compute_cartesian_error(desired_RH_lift_pose[ii+1, :], desired_RH_lift_pose[ii, :], rotation_rep='quat')#error1

        # Compute the jacobian matrix
        #rbdl.CalcPointJacobian6D(robot_model.model, q, model.GetBodyId(LH_name), np.zeros(0), J1, True)
        robot_model.update_jacobian(J1, LH_name, q=q, body_offset=l_soft_hand_offset, update_kinematics=True)
        robot_model.update_jacobian(J2, RH_name, q=q, body_offset=r_soft_hand_offset, update_kinematics=True)
        J1[:, bigman_params['joint_ids']['TO']] = 0
        J2[:, bigman_params['joint_ids']['TO']] = 0
        #print(J1)

        qdot = np.linalg.lstsq(J1, xdot)[0]
        qdot2 = np.linalg.lstsq(J2, xdot2)[0]

        #qdot[bigman_params['joint_ids']['RA']] = qdot[bigman_params['joint_ids']['LA']]*right_sign
        qdot[bigman_params['joint_ids']['RA']] = qdot2[bigman_params['joint_ids']['RA']]

        q[:] += qdot

        actual_LH_pose = robot_model.fk(LH_name, q=q, body_offset=l_soft_hand_offset, update_kinematics=True)
        actual_RH_pose = robot_model.fk(RH_name, q=q, body_offset=r_soft_hand_offset, update_kinematics=True)

        des_cmd.position = q
        publisher.publish(des_cmd)

        joint_lift_trajectory[ii+1, :] = q[:]
        pub_rate.sleep()


save_reach_traj = raw_input("Save reach trajectory? (y/yes): ")
if save_reach_traj.lower() in ['y', 'yes']:
    np.save(file_name + '_m' + str(reach_option) + '_reach.npy', joint_reach_trajectory)
    if reach_option == 2:
        np.save(file_name+'_reach_LH_EE.npy', desired_LH_reach_pose)
        np.save(file_name+'_reach_RH_EE.npy', desired_RH_reach_pose)



save_lift_traj = raw_input("Save lift trajectory? (y/yes): ")
if save_lift_traj.lower() in ['y', 'yes']:
    np.save(file_name + '_m' + str(lift_option) + '_lift.npy', joint_lift_trajectory)
    if lift_option == 2:
        np.save(file_name+'_lift_LH_EE.npy', desired_LH_lift_pose)
        np.save(file_name+'_lift_RH_EE.npy', desired_RH_lift_pose)

#plt.plot(desired_LH_lift_pose[:, -1], 'r')
#plt.show()
