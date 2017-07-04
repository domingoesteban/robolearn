import numpy as np
import tf


def multiply_quat(quat1, quat2):
    w = quat1[3]*quat2[3] - quat1[0]*quat2[0] - quat1[1]*quat2[1] - quat1[2]*quat2[2]
    x = quat1[3]*quat2[0] + quat1[0]*quat2[3] + quat1[1]*quat2[2] - quat1[2]*quat2[1]
    y = quat1[3]*quat2[1] + quat1[1]*quat2[3] + quat1[2]*quat2[0] - quat1[0]*quat2[2]
    z = quat1[3]*quat2[2] + quat1[2]*quat2[3] + quat1[0]*quat2[1] - quat1[1]*quat2[0]
    return [x, y, z, w]


def quaternion_inner(q1, q2):
    q1 = np.array(q1)
    q2 = np.array(q2)
    return q1.dot(q2)
    #return quaternion_multiply(q1, q2)


def quaternion_multiply(q1, q2):
    q1 = np.array(q1)
    q2 = np.array(q2)
    w1 = q1[-1]
    w2 = q2[-1]
    v1 = q1[:3]
    v2 = q2[:3]
    return np.r_[w1*v2+w2*v1 + np.cross(v1, v2), w1*w2-v1.dot(v2)]


def inv_quat(quat):
    return [-quat[0], -quat[1], -quat[2], quat[3]]


def quat_vector_cross(quat_vec):
    return np.array([[0, -quat_vec[2], quat_vec[1]],
                     [quat_vec[2], 0, -quat_vec[0]],
                     [-quat_vec[1], quat_vec[0], 0]])


def quat_difference(final_quat, init_quat):
    return init_quat[3]*final_quat[:3] - final_quat[3]*init_quat[:3] - np.cross(final_quat[:3], init_quat[:3])  # Previous
    #return final_quat[3]*init_quat[:3] - init_quat[3]*final_quat[:3] + quat_vector_cross(final_quat[:3]).dot(init_quat[:3])  # From Nakanishi


def homogeneous_matrix(rot=np.identity(3), pos=np.zeros(3)):
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = rot[:3, :3]
    transform_matrix[:3, -1] = pos[:3]
    return transform_matrix


def compute_cartesian_error(des, current, rotation_rep='quat'):
    """
    Compute the cartesian error between two poses
    :param des: Desired cartesian pose (orientation+position)
    :param current: Actual cartesian pose (orientation+position)
    :param rotation_rep: Orientation units:
    :return: 
    """
    position_error = des[-3:] - current[-3:]
    if rotation_rep == 'quat':
        #orientation_error = current[3]*des[:3] - des[3]*current[:3] - np.cross(des[:3], current[:3])  # Previous
        #orientation_error = des[3]*current[:3] - current[3]*des[:3] + quat_vector_cross(des[:3]).dot(current[:3])  # From Nakanishi
        orientation_error = quat_difference(des[:4], current[:4])
    elif rotation_rep == 'rpy':
        orientation_error = des[:3] - current[:3]
    else:
        raise NotImplementedError("Only quaternion has been implemented")

    return np.concatenate((orientation_error, position_error))


def create_quat_pose(pos_x=0, pos_y=0, pos_z=0, rot_roll=0, rot_pitch=0, rot_yaw=0):
    """
    Rotation assuming first yaw, then pitch, and then yaw.
    :param pos_x: 
    :param pos_y: 
    :param pos_z: 
    :param rot_roll: 
    :param rot_pith: 
    :param rot_yaw: 
    :return: 
    """
    pose = np.zeros(7)
    pose[:4] = tf.transformations.quaternion_from_matrix(tf.transformations.euler_matrix(rot_roll, rot_pitch, rot_yaw))
    pose[4] = pos_x
    pose[5] = pos_y
    pose[6] = pos_z
    return pose


def pose_transform(frame_pose, relative_pose):
    frame_matrix = tf.transformations.quaternion_matrix(frame_pose[:4])
    frame_matrix[:3, -1] = frame_pose[4:]
    relative_matrix = tf.transformations.quaternion_matrix(relative_pose[:4])
    relative_matrix[:3, -1] = relative_pose[4:]
    transform_matrix = frame_matrix.dot(relative_matrix)
    pose = np.zeros(7)
    pose[4:] = transform_matrix[:3, -1]
    pose[:4] = tf.transformations.quaternion_from_matrix(transform_matrix)
    return pose

