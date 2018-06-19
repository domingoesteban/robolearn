from builtins import input
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.envs.simple_envs import GoalCompositionEnv

GOAL = (0.65, 0.65)
TGT_POSE = (0.5, 0.25, 1.4660)


PATH_LENGTH = 500
SIM_TIMESTEP = 0.001
FRAME_SKIP = 10
DT = SIM_TIMESTEP * FRAME_SKIP

env_params = dict(
    goal_reward=10,
    actuation_cost_coeff=30,
    distance_cost_coeff=1,
    init_position=None,
    init_sigma=0.1,
    goal_position=None,
    dynamics_sigma=0,
    goal_threshold=0.1,
    horizon=None,
    log_distance_cost_coeff=1,
    alpha=1e-6,
)
env = NormalizedBoxEnv(
    GoalCompositionEnv(**env_params)
)

# for ii in range(400):
#     env.reset()
#     env.render()

env.reset()
env.render()
# input('Press a key to start...')

for ii in range(50):
    action = env.action_space.sample()
    obs, reward, done, env_info = env.step(action)
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
