import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.derivations.eulerangles import x_rotation, y_rotation, z_rotation
# import tf


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


def compute_cartesian_error(des, current, rotation_rep='quat', first='ori'):
    """
    Compute the cartesian error between two poses: error = des-current
    :param des: Desired cartesian pose (orientation+position)
    :param current: Actual cartesian pose (orientation+position)
    :param rotation_rep: Orientation units:
    :param first: 'ori' or 'pos'
    :return: 
    """
    if first == 'pos':
        position_error = des[:3] - current[:3]
    else:
        position_error = des[-3:] - current[-3:]

    if rotation_rep == 'quat':
        #orientation_error = current[3]*des[:3] - des[3]*current[:3] - np.cross(des[:3], current[:3])  # Previous
        #orientation_error = des[3]*current[:3] - current[3]*des[:3] + quat_vector_cross(des[:3]).dot(current[:3])  # From Nakanishi
        if first == 'pos':
            orientation_error = quat_difference(des[-4:], current[-4:])
        else:
            orientation_error = quat_difference(des[:4], current[:4])
    elif rotation_rep == 'rpy':
        if first == 'pos':
            orientation_error = des[-3:] - current[-3:]
        else:
            orientation_error = des[:3] - current[:3]
    else:
        raise NotImplementedError("Only quaternion has been implemented")

    if first == 'pos':
        return np.concatenate((position_error, orientation_error))
    else:
        return np.concatenate((orientation_error, position_error))


def create_quat_pose(pos_x=0, pos_y=0, pos_z=0, rot_roll=0, rot_pitch=0,
                     rot_yaw=0, first='ori', order='xyzw'):
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
    quat = create_quat(rot_roll=rot_roll, rot_pitch=rot_pitch,
                       rot_yaw=rot_yaw, order=order)
    if first == 'ori':
        pose[:4] = quat
        pose[4] = pos_x
        pose[5] = pos_y
        pose[6] = pos_z
    else:
        pose[0] = pos_x
        pose[1] = pos_y
        pose[2] = pos_z
        pose[3:] = quat

    return pose


def create_quat(rot_roll=0, rot_pitch=0, rot_yaw=0, order='xyzw'):
    # pose[:4] = tf.transformations.quaternion_from_matrix(tf.transformations.euler_matrix(rot_roll, rot_pitch, rot_yaw))
    # rot = x_rotation(rot_roll)*y_rotation(rot_pitch)*z_rotation(rot_yaw)
    rot = euler2mat(rot_roll, rot_pitch, rot_yaw, 'sxyz')
    quat = mat2quat(np.array(rot).astype(float))
    if order == 'wxyz':
        return quat
    elif order == 'xyzw':
        return np.take(quat, [1, 2, 3, 0])
    else:
        raise AttributeError('Wrong order option')


def euler_from_quat(quat, order='xyzw'):
    """

    :param quat:
    :param order:
    :return: [rot_x, rot_y, rot_z]
    """
    if order == 'xyzw':
        quat = np.take(quat, [3, 0, 1, 2])
    elif order == 'wxyz':
        pass
    else:
        raise AttributeError('Wrong order option')

    return mat2euler(np.array(quat2mat(quat)).astype(float))


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


def normalize_angle(angle, range='pi'):
    """

    :param angle:
    :param range: 'pi' or '2pi'
    :return:
    """
    if range == 'pi':
        # reduce the angle
        angle = angle % np.pi

        # Force it to be the positive remainder, so that 0 <= angle < 360
        angle = (angle + np.pi) % np.pi

        # Force into the minimum absolute value residue class, so that
        # -180 < angle <= 180
        if angle > np.pi/2:
            angle -= np.pi

        return angle

    else:
        raise NotImplementedError('Only implemented with -pi/pi')
