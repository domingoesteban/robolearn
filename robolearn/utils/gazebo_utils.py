import rospy
import tf
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import Pose


class ResetGazeboEnv(object):
    def __init__(self, model_names, model_sdfs, model_poses):
        self.model_names = model_names
        self.model_sdfs = model_sdfs
        self.model_poses = [pose_to_geometry_msg(pose_array) for pose_array in model_poses]

        # ROS stuff
        rospy.wait_for_service('gazebo/spawn_sdf_model')
        self.spawn_model_proxy = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        rospy.wait_for_service('gazebo/delete_model')
        self.delete_model_proxy = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)

    def reset(self, delete_prev_models=True):
        if delete_prev_models:
            # TODO: Check if model exist before delete it
            for model_name in self.model_names:
                print("Deleting gazebo model %s..." % model_name)
                self.delete_gazebo_model(model_name)

        for model_name, model_sdf, model_pose in zip(self.model_names, self.model_sdfs, self.model_poses):
            print("Spawning gazebo model %s..." % model_name)
            self.spawn_gazebo_model(model_name, model_sdf, model_pose)

    def delete_gazebo_model(self, model_name):
        try:
            self.delete_model_proxy(model_name)
        except rospy.ServiceException as exc:
            print("/gazebo/delete_model service call failed: %s" % str(exc))

    def spawn_gazebo_model(self, model_name, model_sdf, model_pose):
        try:
            self.spawn_model_proxy(model_name, model_sdf, model_name, model_pose, "world")
        except rospy.ServiceException as exc:
            print("/gazebo/spawn_sdf_model service call failed: %s" % str(exc))


def pose_to_geometry_msg(pose_array):
    """
    pose_array --> [pos_x, pos_y, pos_z, orient_x, orient_y, orient_z, orient_w]
    """
    pose_msg = Pose()
    pose_msg.position.x = pose_array[0]
    pose_msg.position.y = pose_array[1]
    pose_msg.position.z = pose_array[2]
    pose_msg.orientation.x = pose_array[3]
    pose_msg.orientation.y = pose_array[4]
    pose_msg.orientation.z = pose_array[5]
    pose_msg.orientation.w = pose_array[6]

    return pose_msg


def get_gazebo_model_pose(model_name, relative_name=None):
    if relative_name is None:
        relative_name = 'world'
    rospy.wait_for_service('gazebo/get_model_state')
    get_model_state_proxy = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
    try:
        ans = get_model_state_proxy(model_name, relative_name)
        if ans.success is True:
            return ans.pose
        else:
            return None
    except rospy.ServiceException as exc:
        print("/gazebo/spawn_sdf_model service call failed: %s" % str(exc))


def spawn_gazebo_model(model_name, model_sdf, model_pose, relative_name=None):
    if relative_name is None:
        relative_name = 'world'
    if not isinstance(type(model_pose), Pose):
        model_pose = pose_to_geometry_msg(model_pose)

    print("Spawning '%s' gazebo model..." % model_name)
    rospy.wait_for_service('gazebo/spawn_sdf_model')
    spawn_model_proxy = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
    try:
        spawn_model_proxy(model_name, model_sdf, model_name, model_pose, relative_name)
    except rospy.ServiceException as exc:
        print("/gazebo/spawn_sdf_model service call failed: %s" % str(exc))


def delete_gazebo_model(model_name):
    print("Deleting '%s' gazebo model..." % model_name)
    # Check if model exists
    if get_gazebo_model_pose(model_name, relative_name=None) is None:
        print("Model %s does not exist. Then not deleted." % model_name)

    rospy.wait_for_service('gazebo/delete_model')
    delete_model_proxy = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)

    try:
        delete_model_proxy(model_name)
    except rospy.ServiceException as exc:
        print("/gazebo/delete_model service call failed: %s" % str(exc))
