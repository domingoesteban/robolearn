import numpy as np

# ROS packages
import rospy
from XCM.msg import JointStateAdvr
from XCM.msg import CommandAdvr

# Robolearn package
from robolearn.utils.gazebo_ros.ros_utils import obs_vector_ft_sensor, ft_sensor_dof
from robolearn.utils.gazebo_ros.ros_utils import obs_vector_imu, imu_sensor_dof
from robolearn.utils.gazebo_ros.ros_utils import obs_vector_optitrack, optitrack_dof
from robolearn.utils.gazebo_ros.ros_utils import get_indexes_from_list
from robolearn.utils.gazebo_ros.ros_utils import get_sensor_data


def config_xbot_command(joint_names, cmd_type, init_cmd_vals):
    """
    Fill a CommandAdvr message with the specified joint_names and with some initial values in the cmd_type field.
    :param joint_names: List of joint names 
    :param cmd_type: Desired command. E.g. position, velocity, effort, stiffness, damping
    :param init_cmd_vals: Initial command values 
    :return: CommandAdvr message filled
    :rtype: XCM.msg._CommandAdvr.CommandAdvr
    """
    xbot_cmd_msg = CommandAdvr()
    xbot_cmd_msg.name = joint_names
    update_xbot_command(xbot_cmd_msg, cmd_type, init_cmd_vals)

    return xbot_cmd_msg


def update_xbot_command(cmd_msg, cmd_field, cmd_vals):
    """
    Update the values of a CommandAdvr ROS message for a specified field. 
    :param cmd_msg: CommandAdvr ROS message to be updated
    :param cmd_field: Field that will be updated. E.g. position, velocity, effort, stiffness, damping
    :param cmd_vals: Desired values
    :return None
    """
    if cmd_field == 'effort':
        setattr(cmd_msg, 'damping', cmd_vals*0)
        setattr(cmd_msg, 'stiffness', cmd_vals*0)

    if hasattr(cmd_msg, cmd_field):
        setattr(cmd_msg, cmd_field, cmd_vals)
    else:
        raise ValueError("Wrong field option for ADVR command. | type:%s" % cmd_field)




def state_vector_xbot_joint_state(state_fields, joint_names, ros_joint_state_msg):
    """
    Return a vector filled with data from a XCM/JointStateAdvr message for some specific state filds and joint names
    :param state_fields: List of joint state fields. E.g. [link_position, link_velocity]
    :param joint_names: List of joint names. E.g ['jointName1', 'jointName2'] 
    :param ros_joint_state_msg: ROS XCM/JointStateAdvr message from which the date will be obtained 
    :return: Array for the requested data
    :rtype: numpy.ndarray
    """
    state = np.empty(len(joint_names)*len(state_fields))
    for ii, obs_field in enumerate(state_fields):
        state[len(joint_names)*ii:len(joint_names)*(ii+1)] = \
            get_sensor_data(ros_joint_state_msg, obs_field)[get_indexes_from_list(ros_joint_state_msg.name,
                                                                                       joint_names)]
    return state


def get_last_xbot_state_field(robot_name, state_field, joint_names):
    joint_state_msg = rospy.wait_for_message("/xbotcore/"+robot_name+"/joint_states", JointStateAdvr)
    return get_sensor_data(joint_state_msg, state_field)[get_indexes_from_list(joint_state_msg.name,
                                                                                    joint_names)]
