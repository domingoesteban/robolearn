import numpy as np
import os
import rospy
import tf
from XCM.msg import CommandAdvr
from XCM.msg import JointStateAdvr
from robolearn.utils.trajectory_reproducer import TrajectoryReproducer
from robolearn.utils.iit.iit_robots_params import bigman_params
from robolearn.utils.transformations import *
from robolearn.utils.trajectory_interpolators import polynomial5_interpolation
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import DeleteModel
from geometry_msgs.msg import Pose

#current_path = os.path.abspath(__file__)
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)


T_init = 1
remove_spawn_new_box = True
freq = 100
box_position = np.array([0.75,
                         0.00,
                         0.0184])
box_size = [0.4, 0.5, 0.3]
box_yaw = 15  # Degrees
box_orient = tf.transformations.rotation_matrix(np.deg2rad(box_yaw), [0, 0, 1])
box_matrix = homogeneous_matrix(rot=box_orient, pos=box_position)

#traj_files = ['trajectories/traj1_x0.75_y0.0_Y0_m0_reach.npy',
#              'trajectories/traj1_x0.75_y0.0_Y0_m1_lift.npy']
#traj_files = ['trajectories/traj1_x0.8_y0.0_Y0_m0_reach.npy',
#              'trajectories/traj1_x0.8_y0.0_Y0_m1_lift.npy']
#traj_files = ['trajectories/traj1_x0.75_y0.0_Y-15_m0_reach.npy',
#              'trajectories/traj1_x0.75_y0.0_Y-15_m1_lift.npy']
traj_files = ['trajectories/traj1'+'_x'+str(box_position[0])+'_y'+str(box_position[1])+'_Y'+str(box_yaw)+'_m0_reach.npy',
              'trajectories/traj1'+'_x'+str(box_position[0])+'_y'+str(box_position[1])+'_Y'+str(box_yaw)+'_m1_lift.npy']
traj_rep = TrajectoryReproducer(traj_files)

joint_state = np.zeros(31)
joint_state_id = []
def callback(data, params):
    joint_state = params[0]
    joint_state_id = params[1]
    if not joint_state_id:
        joint_state_id[:] = [bigman_params['joints_names'].index(name) for name in data.name]
    joint_state[joint_state_id] = data.link_position
publisher = rospy.Publisher("/xbotcore/bigman/command", CommandAdvr, queue_size=10)
subscriber = rospy.Subscriber("/xbotcore/bigman/joint_states", JointStateAdvr, callback, (joint_state, joint_state_id))
rospy.init_node('traj_example')
pub_rate = rospy.Rate(freq)
des_cmd = CommandAdvr()
des_cmd.name = bigman_params['joints_names']

q_init = traj_rep.get_data(0)
N = int(np.ceil(T_init*freq))
joint_init_traj = polynomial5_interpolation(N, q_init, joint_state)[0]
#raw_input("Press key for moving to the initial configuration of trajectory")
print("Moving to the initial configuration of trajectory")
for ii in range(N):
    des_cmd.position = joint_init_traj[ii, :]
    publisher.publish(des_cmd)
    pub_rate.sleep()

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



#raw_input("Press key for reproducing trajectory")
print("Reproducing trajectory...")
for ii in range(traj_rep.data_points):
    #print("Sending LIFTING cmd...")
    #error = joint_lift_trajectory[ii, :] - joint_state
    #print(error[bigman_params['joint_ids']['BA']])
    #des_cmd.position += K*error
    des_cmd.position = traj_rep.get_data(ii)
    publisher.publish(des_cmd)
    pub_rate.sleep()
