import os
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from robolearn.utils.iit.iit_robots_params import *
from robolearn.utils.trajectory_interpolators import polynomial5_interpolation
from robolearn.utils.trajectory_interpolators import spline_interpolation
from robolearn.utils.trajectory_interpolators import quaternion_interpolation
from robolearn.utils.robot_model import *
from robolearn.utils.iit.robot_poses.bigman.poses import *
from robolearn.utils.transformations import *
from robolearn.utils.plot_utils import *
import tf

np.set_printoptions(precision=4, suppress=True, linewidth=1000)

# Script parameters
#box_position = np.array([0.75, 0.0, 0.0184])
box_mass = 0.71123284
box_position = np.array([0.75,
                         0.00,
                         0.0184])
box_size = [0.4, 0.5, 0.3]
box_yaw = 0  # Degrees
#box_orient = tf.transformations.rotation_matrix(np.deg2rad(15), [1, 0, 0])  # For the EEs is rotation in X
box_orient = tf.transformations.rotation_matrix(np.deg2rad(box_yaw), [0, 0, 1])
box_matrix = homogeneous_matrix(rot=box_orient, pos=box_position)
freq = 100
T_init = 1
T_reach = 10
T_lift = 10

# Save/Load file name
file_name = 'trajectories/traj1'+'_x'+str(box_position[0])+'_y'+str(box_position[1])+'_Y'+str(box_yaw)
load_reach_traj = False
load_lift_traj = False
#load_reach_traj = True
#load_lift_traj = True
save_reach_traj = True
save_lift_traj = True

plot_at_the_end = True

reach_option = 0
#reach_option 0: IK desired final pose, interpolate in joint space
#reach_option 1: Trajectory in EEs, then IK whole trajectory
#reach_option 2: Trajectory in EEs, IK with Jacobians

lift_option = 2
#lift_option 0: IK desired final pose, interpolate the others
#lift_option 1: Trajectory in EEs, then IK whole trajectory
#lift_option 2: Trajectory in EEs, IK with Jacobians

regularization_parameter = 0.02  # For IK optimization algorithm
ik_method = 'iterative' #iterative / optimization

q_init = np.zeros(31)
q_init[16] = np.deg2rad(50)
q_init[25] = np.deg2rad(-50)
#q_init = np.deg2rad(np.array(bigman_Apose))
#q_init = np.deg2rad(np.array(bigman_Fpose))
#q_init = np.deg2rad(np.array(bigman_Tpose))

# Robot Model
robot_urdf = os.environ["ROBOTOLOGY_ROOT"]+'/robots/iit-bigman-ros-pkg/bigman_urdf/urdf/bigman.urdf'
robot_model = RobotModel(robot_urdf)
LH_name = 'LWrMot3'
RH_name = 'RWrMot3'
l_soft_hand_offset = np.array([0.000, -0.030, -0.210])
r_soft_hand_offset = np.array([0.000, 0.030, -0.210])


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
    box_LH_position = np.array([0.05,
                                box_size[1]/2. - 0.00,
                                -0.05])
    box_LH_matrix = homogeneous_matrix(pos=box_LH_position)
    LH_reach_matrix = box_matrix.dot(box_LH_matrix)
    LH_reach_matrix = LH_reach_matrix.dot(tf.transformations.rotation_matrix(np.deg2rad(-90), [0, 1, 0]))
    LH_reach_pose = np.zeros(7)
    LH_reach_pose[4:] = tf.transformations.translation_from_matrix(LH_reach_matrix)
    LH_reach_pose[:4] = tf.transformations.quaternion_from_matrix(LH_reach_matrix)

    box_RH_position = np.array([0.05,
                                -box_size[1]/2. + 0.00,
                                -0.05])
    box_RH_matrix = homogeneous_matrix(pos=box_RH_position)
    RH_reach_matrix = box_matrix.dot(box_RH_matrix)
    RH_reach_matrix = RH_reach_matrix.dot(tf.transformations.rotation_matrix(np.deg2rad(-90), [0, 1, 0]))
    RH_reach_pose = np.zeros(7)
    RH_reach_pose[4:] = tf.transformations.translation_from_matrix(RH_reach_matrix)
    RH_reach_pose[:4] = tf.transformations.quaternion_from_matrix(RH_reach_matrix)

    N = int(np.ceil(T_reach*freq))
    torso_joints = bigman_params['joint_ids']['TO']

    if reach_option == 0:
        q_reach = robot_model.ik(LH_name, LH_reach_pose, body_offset=l_soft_hand_offset,
                                 mask_joints=torso_joints, joints_limits=bigman_params['joints_limits'],
                                 method=ik_method)
        q_reach2 = robot_model.ik(RH_name, RH_reach_pose, body_offset=r_soft_hand_offset,
                                  mask_joints=torso_joints, joints_limits=bigman_params['joints_limits'],
                                  method=ik_method)

        q_reach[bigman_params['joint_ids']['RA']] = q_reach2[bigman_params['joint_ids']['RA']]

        # Trajectory
        joint_reach_trajectory = polynomial5_interpolation(N, q_reach, q_init)[0]

    elif reach_option == 1:
        q = q_init.copy()
        actual_LH_pose = robot_model.fk(LH_name, q=q, body_offset=l_soft_hand_offset, update_kinematics=True)
        actual_RH_pose = robot_model.fk(RH_name, q=q, body_offset=r_soft_hand_offset, update_kinematics=True)
        desired_LH_reach_pose = polynomial5_interpolation(N, LH_reach_pose, actual_LH_pose)[0]
        desired_RH_reach_pose = polynomial5_interpolation(N, RH_reach_pose, actual_RH_pose)[0]

        viapoint_LH_reach = np.empty(3)

        quatLH_interpolation = quaternion_interpolation(N, LH_reach_pose[:4], actual_LH_pose[:4])
        quatRH_interpolation = quaternion_interpolation(N, RH_reach_pose[:4], actual_RH_pose[:4])
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
                                     mask_joints=torso_joints, method=ik_method,
                                     regularization_parameter=regularization_parameter)
            q_reach2[:] = robot_model.ik(RH_name, desired_RH_reach_pose[ii+1, :], body_offset=r_soft_hand_offset,
                                      q_init=joint_reach_trajectory[ii, :], joints_limits=bigman_params['joints_limits'],
                                      mask_joints=torso_joints, method=ik_method,
                                      regularization_parameter=regularization_parameter)
            q_reach[bigman_params['joint_ids']['RA']] = q_reach2[bigman_params['joint_ids']['RA']]
            joint_reach_trajectory[ii+1, :] = q_reach
            #print(joint_reach_trajectory[ii+1, :]-joint_reach_trajectory[ii, :])
            print(sum(joint_reach_trajectory[ii+1, :]-joint_reach_trajectory[ii, :]))

    elif reach_option == 2:
        q = q_init.copy()
        actual_LH_pose = robot_model.fk(LH_name, q=q, body_offset=l_soft_hand_offset, update_kinematics=True)
        actual_RH_pose = robot_model.fk(RH_name, q=q, body_offset=r_soft_hand_offset, update_kinematics=True)
        desired_LH_reach_pose = polynomial5_interpolation(N, LH_reach_pose, actual_LH_pose)[0]
        desired_RH_reach_pose = polynomial5_interpolation(N, RH_reach_pose, actual_RH_pose)[0]

        quatLH_interpolation = quaternion_interpolation(N, LH_reach_pose[:4], actual_LH_pose[:4])
        quatRH_interpolation = quaternion_interpolation(N, RH_reach_pose[:4], actual_RH_pose[:4])
        desired_LH_reach_pose[:, :4] = quatLH_interpolation
        desired_RH_reach_pose[:, :4] = quatRH_interpolation

        q_reach = robot_model.ik(LH_name, LH_reach_pose, body_offset=l_soft_hand_offset,
                                 mask_joints=torso_joints, joints_limits=bigman_params['joints_limits'],
                                 method=ik_method)
        q_reach2 = robot_model.ik(RH_name, RH_reach_pose, body_offset=r_soft_hand_offset,
                                  mask_joints=torso_joints, joints_limits=bigman_params['joints_limits'],
                                  method=ik_method)
        q_reach[bigman_params['joint_ids']['RA']] = q_reach2[bigman_params['joint_ids']['RA']]

        #J1 = np.zeros((6, robot_model.qdot_size))
        #J2 = np.zeros((6, robot_model.qdot_size))
        J = np.zeros((12, robot_model.qdot_size))
        xdot = np.zeros(12)
        qdot = np.zeros(robot_model.qdot_size)
        K = 500
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
        #J1 = np.zeros((6, robot_model.qdot_size))
        #J2 = np.zeros((6, robot_model.qdot_size))
        J = np.zeros((12, robot_model.qdot_size))
        xdot = np.zeros(12)
        qdot = np.zeros(robot_model.qdot_size)
        desired_LH_reach_pose = np.load(file_name+'_reach_LH_EE.npy')
        desired_RH_reach_pose = np.load(file_name+'_reach_RH_EE.npy')
    print("\033[31mDONE!! \033[0m")



# ######## #
# LIFT BOX #
# ######## #
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
                                 method=ik_method)
        q_lift2 = robot_model.ik(RH_name, RH_lift_pose, body_offset=r_soft_hand_offset,
                                  mask_joints=torso_joints, joints_limits=bigman_params['joints_limits'],
                                  method=ik_method)
        q_lift[bigman_params['joint_ids']['RA']] = q_lift2[bigman_params['joint_ids']['RA']]
        joint_lift_trajectory = polynomial5_interpolation(N, q_lift, q_reach)[0]

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
                                       mask_joints=torso_joints, method=ik_method,
                                       regularization_parameter=regularization_parameter)
            q_lift2[:] = robot_model.ik(RH_name, desired_RH_lift_pose[ii+1, :], body_offset=r_soft_hand_offset,
                                        q_init=joint_lift_trajectory[ii, :], joints_limits=bigman_params['joints_limits'],
                                        mask_joints=torso_joints, method=ik_method,
                                        regularization_parameter=regularization_parameter)
            q_lift[bigman_params['joint_ids']['RA']] = q_lift2[bigman_params['joint_ids']['RA']]
            joint_lift_trajectory[ii+1, :] = q_lift

    elif lift_option == 2:
        T_lift = 2
        N = int(np.ceil(T_lift*freq))

        q = q_reach.copy()
        actual_LH_pose = robot_model.fk(LH_name, q=q, body_offset=l_soft_hand_offset, update_kinematics=True)
        actual_RH_pose = robot_model.fk(RH_name, q=q, body_offset=r_soft_hand_offset, update_kinematics=True)
        desired_LH_lift_pose = polynomial5_interpolation(N, LH_lift_pose, actual_LH_pose)[0]
        desired_RH_lift_pose = polynomial5_interpolation(N, RH_lift_pose, actual_RH_pose)[0]

        #J1 = np.zeros((6, robot_model.qdot_size))
        #J2 = np.zeros((6, robot_model.qdot_size))
        J = np.zeros((12, robot_model.qdot_size))
        xdot = np.zeros(12)
        qdot = np.zeros(robot_model.qdot_size)
        K = 500
    else:
        raise ValueError("Wrong lift_option %d" % lift_option)
    print("\n\033[31mDONE!! \033[0m")

else:
    print("\n\033[5mLoading lifting trajectory...")
    joint_lift_trajectory = np.load(file_name + '_m' + str(lift_option) + '_lift.npy')
    if lift_option == 2:
        #J1 = np.zeros((6, robot_model.qdot_size))
        #J2 = np.zeros((6, robot_model.qdot_size))
        J = np.zeros((12, robot_model.qdot_size))
        xdot = np.zeros(12)
        qdot = np.zeros(robot_model.qdot_size)
        desired_LH_lift_pose = np.load(file_name+'_lift_LH_EE.npy')
        desired_RH_lift_pose = np.load(file_name+'_lift_RH_EE.npy')
    print("\033[31mDONE!! \033[0m")

# Send Commands
if reach_option == 2:
    joint_reach_trajectory = np.empty((desired_LH_reach_pose.shape[0], robot_model.q_size))
    q = q_init.copy()
    joint_reach_trajectory[0, :] = q[:]
    for ii in range(desired_LH_reach_pose.shape[0]-1):
        #for ii in range(N-1):
        print("Generating REACHING %d/%d..." % (ii+1, desired_LH_reach_pose.shape[0]))
        #error1 = compute_cartesian_error(desired_LH_lift_pose[ii, :], actual_LH_lift_pose, rotation_rep='quat')
        #error2 = compute_cartesian_error(desired_RH_lift_pose[ii, :], actual_RH_lift_pose, rotation_rep='quat')

        xdot[:6] = compute_cartesian_error(desired_LH_reach_pose[ii+1, :], desired_LH_reach_pose[ii, :], rotation_rep='quat')#error1
        xdot[6:] = compute_cartesian_error(desired_RH_reach_pose[ii+1, :], desired_RH_reach_pose[ii, :], rotation_rep='quat')#error1

        ## Compute the jacobian matrix
        ##rbdl.CalcPointJacobian6D(robot_model.model, q, model.GetBodyId(LH_name), np.zeros(0), J1, True)
        #robot_model.update_jacobian(J1, LH_name, q=q, body_offset=l_soft_hand_offset, update_kinematics=True)
        #robot_model.update_jacobian(J2, RH_name, q=q, body_offset=r_soft_hand_offset, update_kinematics=True)
        #J1[:, bigman_params['joint_ids']['TO']] = 0
        #J2[:, bigman_params['joint_ids']['TO']] = 0
        #qdot = np.linalg.lstsq(J1, xdot)[0]
        #qdot2 = np.linalg.lstsq(J2, xdot2)[0]
        #qdot[bigman_params['joint_ids']['RA']] = qdot2[bigman_params['joint_ids']['RA']]

        # Compute the jacobian matrix
        #rbdl.CalcPointJacobian6D(robot_model.model, q, model.GetBodyId(LH_name), np.zeros(0), J1, True)
        robot_model.update_jacobian(J[:6, :], LH_name, q=q, body_offset=l_soft_hand_offset, update_kinematics=True)
        robot_model.update_jacobian(J[6:, :], RH_name, q=q, body_offset=r_soft_hand_offset, update_kinematics=True)
        #Note: lstsq is faster than pinv and then dot
        qdot[:] = np.linalg.lstsq(J, xdot)[0]

        q[:] += qdot

        actual_LH_pose = robot_model.fk(LH_name, q=q, body_offset=l_soft_hand_offset, update_kinematics=True)
        actual_RH_pose = robot_model.fk(RH_name, q=q, body_offset=r_soft_hand_offset, update_kinematics=True)

        joint_reach_trajectory[ii+1, :] = q[:]

if lift_option == 2:
    joint_lift_trajectory = np.empty((desired_LH_lift_pose.shape[0], robot_model.q_size))
    q = q_reach.copy()
    joint_lift_trajectory[0, :] = q[:]
    for ii in range(desired_LH_lift_pose.shape[0]-1):
    #for ii in range(N-1):
        print("Generating LIFTING %d/%d..." % (ii+1, desired_LH_lift_pose.shape[0]))
        #error1 = compute_cartesian_error(desired_LH_lift_pose[ii, :], actual_LH_lift_pose, rotation_rep='quat')
        #error2 = compute_cartesian_error(desired_RH_lift_pose[ii, :], actual_RH_lift_pose, rotation_rep='quat')

        xdot[:6] = compute_cartesian_error(desired_LH_lift_pose[ii+1, :], desired_LH_lift_pose[ii, :], rotation_rep='quat')#error1
        xdot[6:] = compute_cartesian_error(desired_RH_lift_pose[ii+1, :], desired_RH_lift_pose[ii, :], rotation_rep='quat')#error1

        ## Compute the jacobian matrix
        ##rbdl.CalcPointJacobian6D(robot_model.model, q, model.GetBodyId(LH_name), np.zeros(0), J1, True)
        #robot_model.update_jacobian(J1, LH_name, q=q, body_offset=l_soft_hand_offset, update_kinematics=True)
        #robot_model.update_jacobian(J2, RH_name, q=q, body_offset=r_soft_hand_offset, update_kinematics=True)
        #J1[:, bigman_params['joint_ids']['TO']] = 0
        #J2[:, bigman_params['joint_ids']['TO']] = 0
        #qdot = np.linalg.lstsq(J1, xdot)[0]
        #qdot2 = np.linalg.lstsq(J2, xdot2)[0]
        #qdot[bigman_params['joint_ids']['RA']] = qdot2[bigman_params['joint_ids']['RA']]

        # Compute the jacobian matrix
        #rbdl.CalcPointJacobian6D(robot_model.model, q, model.GetBodyId(LH_name), np.zeros(0), J1, True)
        robot_model.update_jacobian(J[:6, :], LH_name, q=q, body_offset=l_soft_hand_offset, update_kinematics=True)
        robot_model.update_jacobian(J[6:, :], RH_name, q=q, body_offset=r_soft_hand_offset, update_kinematics=True)
        #Note: lstsq is faster than pinv and then dot
        qdot[:] = np.linalg.lstsq(J, xdot)[0]
        #qdot = np.linalg.pinv(J).dot(xdot)

        q[:] += qdot

        actual_LH_pose = robot_model.fk(LH_name, q=q, body_offset=l_soft_hand_offset, update_kinematics=True)
        actual_RH_pose = robot_model.fk(RH_name, q=q, body_offset=r_soft_hand_offset, update_kinematics=True)

        joint_lift_trajectory[ii+1, :] = q[:]


if save_reach_traj:
    np.save(file_name + '_m' + str(reach_option) + '_reach.npy', joint_reach_trajectory)
    if reach_option == 2:
        np.save(file_name+'_reach_LH_EE.npy', desired_LH_reach_pose)
        np.save(file_name+'_reach_RH_EE.npy', desired_RH_reach_pose)

if save_lift_traj:
    np.save(file_name + '_m' + str(lift_option) + '_lift.npy', joint_lift_trajectory)
    if lift_option == 2:
        np.save(file_name+'_lift_LH_EE.npy', desired_LH_lift_pose)
        np.save(file_name+'_lift_RH_EE.npy', desired_RH_lift_pose)

#plt.plot(desired_LH_lift_pose[:, -1], 'r')
#plt.show()
if plot_at_the_end:
    joints_to_plot = bigman_params['joint_ids']['LA']
    cols = 3
    joint_names = [bigman_params['joints_names'][idx] for idx in joints_to_plot]
    #plot_joint_info(joints_to_plot, joint_reach_trajectory, joint_names, data='position', block=False)
    #qdots_reach = np.vstack((np.diff(joint_reach_trajectory, axis=0), np.zeros((1, robot_model.qdot_size))))
    #plot_joint_info(joints_to_plot, qdots_reach*freq, joint_names, data='velocity', block=False)
    #qddots_reach = np.vstack((np.diff(qdots_reach, axis=0), np.zeros((1, robot_model.qdot_size))))
    #plot_joint_info(joints_to_plot, qddots_reach*freq*freq, joint_names, data='acceleration', block=False)

    plot_joint_info(joints_to_plot, joint_lift_trajectory, joint_names, data='position', block=False)
    qdots_lift = np.vstack((np.diff(joint_lift_trajectory, axis=0), np.zeros((1, robot_model.qdot_size))))
    plot_joint_info(joints_to_plot, qdots_lift*freq, joint_names, data='velocity', block=False)
    qddots_lift = np.vstack((np.diff(qdots_lift, axis=0), np.zeros((1, robot_model.qdot_size))))
    plot_joint_info(joints_to_plot, qddots_lift*freq*freq, joint_names, data='acceleration', block=False)

    raw_input("Press a key to close the script")


