import numpy as np


def multiply_quat(quat1, quat2):
    w = quat1[3]*quat2[3] - quat1[0]*quat2[0] - quat1[1]*quat2[1] - quat1[2]*quat2[2]
    x = quat1[3]*quat2[0] + quat1[0]*quat2[3] + quat1[1]*quat2[2] - quat1[2]*quat2[1]
    y = quat1[3]*quat2[1] + quat1[1]*quat2[3] + quat1[2]*quat2[0] - quat1[0]*quat2[2]
    z = quat1[3]*quat2[2] + quat1[2]*quat2[3] + quat1[0]*quat2[1] - quat1[1]*quat2[0]
    return [x, y, z, w]


def inv_quat(quat):
    return [-quat[0], -quat[1], -quat[2], quat[3]]


def quat_vector_cross(quat_vec):
    return np.array([[0, -quat_vec[2], quat_vec[1]],
                     [quat_vec[2], 0, -quat_vec[0]],
                     [-quat_vec[1], quat_vec[0], 0]])


def quat_difference(final_quat, init_quat):
    #return init_quat[3]*final_quat[:3] - final_quat[3]*init_quat[:3] - np.cross(final_quat[:3], init_quat[:3])  # Previous
    return final_quat[3]*init_quat[:3] - init_quat[3]*final_quat[:3] + quat_vector_cross(final_quat[:3]).dot(init_quat[:3])  # From Nakanishi


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

