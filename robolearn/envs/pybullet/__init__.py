# from gym.envs.registration import registry, register, make, spec
#
# register(
#     id='Manipulator2dBulletEnv-v0',
#     entry_point='robolearn.envs.pybullet.manipulator2d_env:Manipulator2dBulletEnv',
#     max_episode_steps=1000,
#     #timestep_limit=1000,
#     reward_threshold=2500.0
# )

from robolearn.envs.pybullet.reacher import ReacherBulletEnv
