import numpy as np

def multiply_quat(quat1, quat2):
    w = quat1[3]*quat2[3] - quat1[0]*quat2[0] - quat1[1]*quat2[1] - quat1[2]*quat2[2]
    x = quat1[3]*quat2[0] + quat1[0]*quat2[3] + quat1[1]*quat2[2] - quat1[2]*quat2[1]
    y = quat1[3]*quat2[1] + quat1[1]*quat2[3] + quat1[2]*quat2[0] - quat1[0]*quat2[2]
    z = quat1[3]*quat2[2] + quat1[2]*quat2[3] + quat1[0]*quat2[1] - quat1[1]*quat2[0]
    return (x, y, z, w)


def inv_quat(quat):
    return (-quat[0], -quat[1], -quat[2], quat[3])


def homogeneous_matrix(rot=np.identity(3), pos=np.zeros(3)):
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = rot[:3, :3]
    transform_matrix[:3, -1] = pos[:3]
    return transform_matrix


def compute_cartesian_error(ref, actual, rotation_rep='quat'):
    """
    Compute the cartesian error between two poses
    :param ref: Desired cartesian pose (orientation+position)
    :param actual: Actual cartesian pose (orientation+position)
    :param rotation_rep: Orientation units:
    :return: 
    """
    position_error = ref[-3:] - actual[-3:]
    if rotation_rep == 'quat':
        orientation_error = actual[3]*ref[:3] - ref[3]*actual[:3] - np.cross(ref[:3], actual[:3])
    elif rotation_rep == 'rpy':
        orientation_error = ref[:3] - actual[:3]
    else:
        raise NotImplementedError("Only quaternion has been implemented")

    return np.concatenate((orientation_error, position_error))
