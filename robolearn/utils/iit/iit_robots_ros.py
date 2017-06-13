import numpy as np
import rospy

# ROS package
from XCM.msg import JointStateAdvr
from XCM.msg import CommandAdvr
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Imu
from robolearn_gazebo_env.msg import RelativePose

# Robolearn package
from robolearn.utils.iit.iit_robots_params import *


def config_advr_command(joint_names, cmd_type, init_cmd_vals):
    """
    Fill a CommandAdvr message with the specified joint_names and with some initial values in the cmd_type field.
    :param joint_names: List of joint names 
    :param cmd_type: Desired command. E.g. position, velocity, effort, stiffness, damping
    :param init_cmd_vals: Initial command values 
    :return: CommandAdvr message filled
    :rtype: XCM.msg._CommandAdvr.CommandAdvr
    """
    advr_cmd_msg = CommandAdvr()
    advr_cmd_msg.name = joint_names
    #advr_cmd_msg = update_advr_command(advr_cmd_msg, cmd_type, init_cmd_vals)
    update_advr_command(advr_cmd_msg, cmd_type, init_cmd_vals)

    #if hasattr(advr_cmd_msg, cmd_type):
    #    setattr(advr_cmd_msg, cmd_type, init_cmd_vals)
    #else:
    #    raise ValueError("Wrong ADVR command type option")
    return advr_cmd_msg


def update_advr_command(cmd_msg, cmd_field, cmd_vals):
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
    #return cmd_msg


def get_advr_sensor_data(obs_msg, sensor_field):
    """
    Get the values from a ROS message for a specified field.
    :param obs_msg: 
    :param sensor_field: 
    :return: 
    """
    if hasattr(obs_msg, sensor_field):
        data_field = getattr(obs_msg, sensor_field)
    else:
        raise ValueError("Wrong field option for ADVR sensor. | type:%s | obs_msg_type:%s" % (sensor_field, type(obs_msg)))

    return np.asarray(data_field)


def copy_class_attr(objfrom, objto, attribute_names):
    """
    Copy the attribute from one ROS message to another one.
    :param objfrom: 
    :param objto: 
    :param attribute_names: 
    :return: 
    """

    for n in attribute_names:
        #if hasattr(objfrom, n):
        #    v = getattr(objfrom, n)
        #    setattr(objto, n, v)
        #else:
        #    raise ValueError("Wrong ADVR attribute")
        if isinstance(objfrom, RelativePose):
            if hasattr(objfrom, 'pose'):
                new_pose = getattr(objfrom, 'pose')
                if hasattr(new_pose, n):
                    v = getattr(new_pose, n)
                    setattr(objto.pose, n, v)
                    #setattr(objto, 'wrench', wrench)
        elif isinstance(objfrom, WrenchStamped):
            if hasattr(objfrom, 'wrench'):
                new_wrench = getattr(objfrom, 'wrench')
                if hasattr(new_wrench, n):
                    v = getattr(new_wrench, n)
                    setattr(objto.wrench, n, v)
                #setattr(objto, 'wrench', wrench)
        elif isinstance(objfrom, JointStateAdvr) or isinstance(objfrom, Imu):
            if hasattr(objfrom, n):
                v = getattr(objfrom, n)
                setattr(objto, n, v)
        else:
            raise TypeError("ROS sensor not supported %s" % type(objfrom))


def get_indexes_from_list(list_to_check, values):
    """
    Get the indexes of matching values 
    :param list: List whose values we want to look at.
    :param values: Values we want to find.
    :return: 
    """
    return [list_to_check.index(a) for a in values]


def obs_vector_joint_state(obs_fields, joint_names, ros_joint_state_msg):
    observation = np.empty((len(joint_names)*len(obs_fields), 1))
    #print (observation.shape)
    #print(obs_fields)

    for ii, obs_field in enumerate(obs_fields):
        observation[len(joint_names)*ii:len(joint_names)*(ii+1), -1] = \
                get_advr_sensor_data(ros_joint_state_msg, obs_field)[get_indexes_from_list(ros_joint_state_msg.name,
                                                                                           joint_names)]
    return observation


def obs_vector_ft_sensor(obs_fields, ros_ft_sensor_msg):
    observation = np.empty((sum([ft_sensor_dof[x] for x in obs_fields]), 1))
    prev_idx = 0
    for ii, obs_field in enumerate(obs_fields):
        #print(get_advr_sensor_data(ros_ft_sensor_msg.wrench, obs_field)[1])
        #print(get_advr_sensor_data(ros_ft_sensor_msg.wrench, obs_field)[2])
        wrench_data = get_advr_sensor_data(ros_ft_sensor_msg.wrench, obs_field).item()
        observation[prev_idx] = wrench_data.x
        observation[prev_idx+1] = wrench_data.y
        observation[prev_idx+2] = wrench_data.z
        prev_idx += ft_sensor_dof[obs_field]

    return observation


def obs_vector_imu(obs_fields, ros_imu_msg):
    observation = np.empty((sum([imu_sensor_dof[x] for x in obs_fields]), 1))
    prev_idx = 0
    for ii, obs_field in enumerate(obs_fields):
        #print(get_advr_sensor_data(ros_imu_sensor_msg.wrench, obs_field)[1])
        #print(get_advr_sensor_data(ros_fimusensor_msg.wrench, obs_field)[2])
        imu_data = get_advr_sensor_data(ros_imu_msg, obs_field).item()
        observation[prev_idx] = imu_data.x
        observation[prev_idx+1] = imu_data.y
        observation[prev_idx+2] = imu_data.z
        if obs_field == 'orientation':
            observation[prev_idx+3] = imu_data.w

        prev_idx += imu_sensor_dof[obs_field]

    return observation


def obs_vector_optitrack(obs_fields, body_names, ros_optitrack_msg):
    observation = np.empty((len(body_names)*sum([optitrack_dof[x] for x in obs_fields]), 1))
    #print (observation.shape)
    #print(obs_fields)
    #print(body_names)
    #print(ros_optitrack_msg)

    prev_idx = 0
    bodies_idx = get_indexes_from_list(ros_optitrack_msg.name, body_names)
    for hh, body_names in enumerate(body_names):
        for ii, obs_field in enumerate(obs_fields):

            pose_data = get_advr_sensor_data(ros_optitrack_msg.pose[bodies_idx[hh]], obs_field).item()
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

    #print(observation)
    #raw_input("AA")
    return observation


def state_vector_joint_state(state_fields, joint_names, ros_joint_state_msg):
    """
    Return a vector filled with data from a XCM/JointStateAdvr message for some specific state filds and joint names
    :param state_fields: List of joint state fields. E.g. [link_position, link_velocity]
    :param joint_names: List of joint names. E.g ['jointName1', 'jointName2'] 
    :param ros_joint_state_msg: ROS XCM/JointStateAdvr message from which the date will be obtained 
    :return: Array for the requested data
    :rtype: numpy.ndarray
    """
    state = np.empty((len(joint_names)*len(state_fields), 1))
    for ii, obs_field in enumerate(state_fields):
        state[len(joint_names)*ii:len(joint_names)*(ii+1), -1] = \
            get_advr_sensor_data(ros_joint_state_msg, obs_field)[get_indexes_from_list(ros_joint_state_msg.name,
                                                                                       joint_names)]
    return state


def get_last_advr_state_field(robot_name, state_field, joint_names):
    joint_state_msg = rospy.wait_for_message("/xbotcore/"+robot_name+"/joint_states", JointStateAdvr)
    return get_advr_sensor_data(joint_state_msg, state_field)[get_indexes_from_list(joint_state_msg.name,
                                                                                    joint_names)]
