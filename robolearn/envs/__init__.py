# from ros_env_interface import ROSEnvInterface
# from robot_ros_env_interface import RobotROSEnvInterface
# from acrobot.acrobot_ros_env_interface import AcrobotROSEnvInterface

# from .bigman.bigman_env import BigmanEnv
# from .bigman.bigman_task_envs import BigmanBoxEnv
# from .centauro.centauro_env import CentauroEnv
# from .gym_environment import GymEnv

from gym.envs.registration import register

register(
    id='ReacherBullet-v0',
    entry_point='robolearn.envs.pybullet:ReacherBulletEnv',
    # timestep_limit=1000,
    # max_episode_steps=50,
    # reward_threshold=-3.75,
)

# register(
# 	id='ReacherBulletEnv-v0',
# 	entry_point='pybullet_envs.gym_manipulator_envs:ReacherBulletEnv',
# 	max_episode_steps=150,
# 	reward_threshold=18.0,
# 	)