from builtins import input
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn_gym_envs.pybullet import Pusher2D3DofGoalCompoEnv

GOAL = (0.65, 0.65)
TGT_POSE = (0.5, 0.25, 1.4660)


PATH_LENGTH = 500
SIM_TIMESTEP = 0.001
FRAME_SKIP = 10
DT = SIM_TIMESTEP * FRAME_SKIP

env_params = dict(
    is_render=True,
    # is_render=False,
    obs_with_img=False,
    goal_poses=[GOAL, (GOAL[0], 'any'), ('any', GOAL[1])],
    rdn_goal_pose=True,
    tgt_pose=TGT_POSE,
    rdn_tgt_object_pose=True,
    sim_timestep=0.001,
    frame_skip=10,
    obs_distances=False,  # If True obs contain 'distance' vectors instead poses
    tgt_cost_weight=1.,#1.5,
    goal_cost_weight=1.,#3.0,
    # goal_cost_weight=0.0,
    ctrl_cost_weight=1.0e-4,
    use_log_distances=False,
    log_alpha=1e-1,  # In case use_log_distances=True
    # max_time=PATH_LENGTH*DT,
    max_time=None,
)
env = NormalizedBoxEnv(
    Pusher2D3DofGoalCompoEnv(**env_params)
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
