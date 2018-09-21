from builtins import input
import numpy as np
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.envs.simple_envs import GoalCompositionEnv

GOAL = (0.65, 0.65)
TGT_POSE = (0.5, 0.25, 1.4660)


PATH_LENGTH = 500
SIM_TIMESTEP = 0.001
FRAME_SKIP = 10
DT = SIM_TIMESTEP * FRAME_SKIP

env_params = dict(
    goal_reward=5,
    actuation_cost_coeff=0.5,
    distance_cost_coeff=1.5,
    log_distance_cost_coeff=0,#1.5,
    alpha=1e-6,
    # Initial Condition
    init_position=(-4., -4.),
    init_sigma=1.50,
    # Goal
    goal_position=(5., 5.),
    goal_threshold=0.25,
    # Others
    dynamics_sigma=0.1,
    # horizon=PATH_LENGTH,
    horizon=None,
)
env = NormalizedBoxEnv(
    GoalCompositionEnv(**env_params)
)
for ii in range(5):
    env.reset()
    env.render()

env.reset()
env.render()

# input('Press a key to start interacting...')
for ii in range(50):
    action = env.action_space.sample()
    obs, reward, done, env_info = env.step(action)
    print('')
    print('---'*3, ii, '---'*3)
    print('action -->', action)
    print('obs -->', obs)
    print('reward -->', reward)
    print('done -->', done)
    print('info -->', env_info)
    env.render()

input('Press a key to reset...')

env.reset()
env.render()

input('Press a key to close the script')
