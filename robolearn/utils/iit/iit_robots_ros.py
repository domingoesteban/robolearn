import numpy as np

from XCM.msg import JointStateAdvr
from XCM.msg import CommandAdvr
from std_srvs.srv import SetBool
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Imu
from robolearn_gazebo_env.msg import RelativePose

from robolearn.utils.iit.iit_robots_params import *

def config_advr_command(joint_names, cmd_type, init_cmd_vals):
    advr_cmd_msg = CommandAdvr()
    advr_cmd_msg.name = joint_names
    advr_cmd_msg = update_advr_command(advr_cmd_msg, cmd_type, init_cmd_vals)
    #if hasattr(advr_cmd_msg, cmd_type):
    #    setattr(advr_cmd_msg, cmd_type, init_cmd_vals)
    #else:
    #    raise ValueError("Wrong ADVR command type option")
    return advr_cmd_msg


def update_advr_command(cmd_msg, cmd_type, cmd_vals):
    if hasattr(cmd_msg, cmd_type):
        setattr(cmd_msg, cmd_type, cmd_vals)
    else:
        raise ValueError("Wrong ADVR command type option")
    return cmd_msg


def get_advr_sensor_data(obs_msg, state_type):
    if hasattr(obs_msg, state_type):
        data_field = getattr(obs_msg, state_type)
    else:
        raise ValueError("Wrong ADVR field type option")

    return np.asarray(data_field)


def copy_class_attr(objfrom, objto, names):

    for n in names:
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


def get_indeces_from_list(list, names):
    return [list.index(a) for a in names]


def obs_vector_joint_state(joint_state_fields, joint_names, ros_joint_state_msg):
    observation = np.empty((len(joint_names)*len(joint_state_fields), 1))
    #print (observation.shape)
    #print(joint_state_fields)

    for ii, obs_field in enumerate(joint_state_fields):
        observation[len(joint_names)*ii:len(joint_names)*(ii+1), -1] = \
                get_advr_sensor_data(ros_joint_state_msg, obs_field)[get_indeces_from_list(ros_joint_state_msg.name,
                                                                                           joint_names)]
    return observation

def state_vector_joint_state(joint_state_fields, joint_names, ros_joint_state_msg):
    state = np.empty((len(joint_names)*len(joint_state_fields), 1))
    for ii, obs_field in enumerate(joint_state_fields):
        state[len(joint_names)*ii:len(joint_names)*(ii+1), -1] = \
            get_advr_sensor_data(ros_joint_state_msg, obs_field)[get_indeces_from_list(ros_joint_state_msg.name,
                                                                                       joint_names)]
    return state


def obs_vector_ft_sensor(ft_sensor_fields, ros_ft_sensor_msg):
    observation = np.empty((sum([ft_sensor_dof[x] for x in ft_sensor_fields]), 1))
    prev_idx = 0
    for ii, obs_field in enumerate(ft_sensor_fields):
        #print(get_advr_sensor_data(ros_ft_sensor_msg.wrench, obs_field)[1])
        #print(get_advr_sensor_data(ros_ft_sensor_msg.wrench, obs_field)[2])
        wrench_data = get_advr_sensor_data(ros_ft_sensor_msg.wrench, obs_field).item()
        observation[prev_idx] = wrench_data.x
        observation[prev_idx+1] = wrench_data.y
        observation[prev_idx+2] = wrench_data.z
        prev_idx += ft_sensor_dof[obs_field]

    return observation

def obs_vector_imu(imu_sensor_fields, ros_imu_msg):
    observation = np.empty((sum([imu_sensor_dof[x] for x in imu_sensor_fields]), 1))
    prev_idx = 0
    for ii, obs_field in enumerate(imu_sensor_fields):
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

def obs_vector_optitrack(optitrack_fields, body_names, ros_optitrack_msg):
    observation = np.empty((len(body_names)*sum([optitrack_dof[x] for x in optitrack_fields]), 1))
    #print (observation.shape)
    #print(optitrack_fields)
    #print(body_names)
    #print(ros_optitrack_msg)

    prev_idx = 0
    bodies_idx = get_indeces_from_list(ros_optitrack_msg.name, body_names)
    for hh, body_names in enumerate(body_names):
        for ii, obs_field in enumerate(optitrack_fields):

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

    print(observation)
    #raw_input("AA")
    return observation

