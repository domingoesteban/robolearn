from robolearn.utils.gazebo_utils import *
from robolearn.utils.lift_box_utils import *
import rospkg

rospack = rospkg.RosPack()
box_sdf = open(rospack.get_path('robolearn_gazebo_env')+'/models/cardboard_cube_box/model.sdf', 'r').read()

model_name = 'box'
model_sdf = box_sdf
model_pose = [0.75, 0, 0.019, 0, 0, 0, 1]

#reset_gazebo_env = ResetGazeboEnv(model_names, model_sdfs, model_poses)
#reset_gazebo_env.reset()

#delete_gazebo_model(model_name)
#spawn_gazebo_model(model_name, model_sdf, model_pose)
#bigman_pose = get_gazebo_model_pose('bigman', 'world')

reset_bigman_box_gazebo(model_pose)

