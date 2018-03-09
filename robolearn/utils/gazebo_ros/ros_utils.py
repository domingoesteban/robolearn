import numpy as np
import socket
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from robolearn_gazebo_env.msg import RelativePose
from XBotCore.msg import JointStateAdvr


# Joint state fields
joint_state_fields = ['position',
                      'velocity',
                      'effort']

# FT sensor fields (geometry_msgs/WrenchStamped)
ft_sensor_fields = ['force', 'torque']

ft_sensor_dof = {'force': 3,  # x, y, z
                 'torque': 3}  # x, y, z

# IMU sensor fields (sensor_msgs/Imu)
imu_sensor_fields = ['orientation',  # x, y, z, w
                     'angular_velocity',  # x, y, z
                     'linear_acceleration']  # x, y, z

imu_sensor_dof = {'orientation': 4,  # x, y, z, w
                  'angular_velocity': 3,  # x, y, z
                  'linear_acceleration': 3}  # x, y, z

# ROS OPTITRACK (robolearn_gazebo_envs PACKAGE)
optitrack_fields = ['position', 'orientation']

optitrack_dof = {'position': 3,  # x, y, z
                 'orientation': 4}  # x, y, z, w


def get_indexes_from_list(list_to_check, values):
    """
    Get the indexes of matching values
    :param list_to_check: List whose values we want to look at.
    :param values: Values we want to find.
    :return:
    """
    return [list_to_check.index(a) for a in values]


def get_sensor_data(obs_msg, sensor_field):
    """
    Get the values from a ROS message for a specified field.
    :param obs_msg:
    :param sensor_field:
    :return:
    """
    if hasattr(obs_msg, sensor_field):
        data_field = getattr(obs_msg, sensor_field)
    else:
        raise ValueError("Wrong field option for ADVR sensor. | type:%s | obs_msg_type:%s" % (sensor_field,
                                                                                              type(obs_msg)))

    return np.asarray(data_field)


def obs_vector_joint_state(obs_fields, joint_names, ros_joint_state_msg):
    observation = np.empty(len(joint_names)*len(obs_fields))

    for ii, obs_field in enumerate(obs_fields):
        observation[len(joint_names)*ii:len(joint_names)*(ii+1)] = \
            get_sensor_data(ros_joint_state_msg, obs_field)[get_indexes_from_list(ros_joint_state_msg.name,
                                                                                  joint_names)]
    return observation


def obs_vector_ft_sensor(obs_fields, ros_ft_sensor_msg):
    observation = np.empty(sum([ft_sensor_dof[x] for x in obs_fields]))
    prev_idx = 0
    for ii, obs_field in enumerate(obs_fields):
        wrench_data = get_sensor_data(ros_ft_sensor_msg.wrench, obs_field).item()
        observation[prev_idx] = wrench_data.x
        observation[prev_idx+1] = wrench_data.y
        observation[prev_idx+2] = wrench_data.z
        prev_idx += ft_sensor_dof[obs_field]

    return observation


def obs_vector_imu(obs_fields, ros_imu_msg):
    observation = np.empty(sum([imu_sensor_dof[x] for x in obs_fields]))
    prev_idx = 0
    for ii, obs_field in enumerate(obs_fields):
        imu_data = get_sensor_data(ros_imu_msg, obs_field).item()
        observation[prev_idx] = imu_data.x
        observation[prev_idx+1] = imu_data.y
        observation[prev_idx+2] = imu_data.z
        if obs_field == 'orientation':
            observation[prev_idx+3] = imu_data.w

        prev_idx += imu_sensor_dof[obs_field]

    return observation


def obs_vector_optitrack(obs_fields, body_names, ros_optitrack_msg):
    observation = np.empty(len(body_names)*sum([optitrack_dof[x] for x in obs_fields]))

    prev_idx = 0
    bodies_idx = get_indexes_from_list(ros_optitrack_msg.name, body_names)
    for hh, body_names in enumerate(body_names):
        for ii, obs_field in enumerate(obs_fields):

            pose_data = get_sensor_data(ros_optitrack_msg.pose[bodies_idx[hh]], obs_field).item()
            if obs_field == 'position':
                observation[prev_idx] = pose_data.x
                observation[prev_idx+1] = pose_data.y
                observation[prev_idx+2] = pose_data.z

            elif obs_field == 'orientation':
                observation[prev_idx] = pose_data.x
                observation[prev_idx+1] = pose_data.y
                observation[prev_idx+2] = pose_data.z
                observation[prev_idx+3] = pose_data.w
            else:
                raise ValueError("Wrong optitrack field")

            prev_idx += optitrack_dof[obs_field]

    return observation


def copy_class_attr(objfrom, objto, attribute_names):
    """
    Copy the attribute from one ROS message to another one.
    :param objfrom:
    :param objto:
    :param attribute_names:
    :return:
    """

    for n in attribute_names:
        if isinstance(objfrom, RelativePose):
            if hasattr(objfrom, 'pose'):
                new_pose = getattr(objfrom, 'pose')
                for body_idx, body_pose in enumerate(new_pose):
                    if hasattr(body_pose, n):
                        v = getattr(body_pose, n)
                        setattr(objto.pose[body_idx], n, v)
                        # setattr(objto, 'wrench', wrench)
        elif isinstance(objfrom, WrenchStamped):
            if hasattr(objfrom, 'wrench'):
                new_wrench = getattr(objfrom, 'wrench')
                if hasattr(new_wrench, n):
                    v = getattr(new_wrench, n)
                    setattr(objto.wrench, n, v)
                    # setattr(objto, 'wrench', wrench)
        elif isinstance(objfrom, JointStateAdvr) or isinstance(objfrom, Imu) or isinstance(objfrom, JointState):
            if hasattr(objfrom, n):
                v = getattr(objfrom, n)
                setattr(objto, n, v)
        else:
            raise TypeError("ROS sensor not supported %s" % type(objfrom))
