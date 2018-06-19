from builtins import input
import numpy as np
from robolearn.envs.simple_envs import CrawlingRobotEnv

env_params = dict(
    horizon=np.inf,
    render=True,
)
env = CrawlingRobotEnv(**env_params)

# for ii in range(400):
#     env.reset()
#     env.render()

env.reset()
env.render()
# input('Press a key to start...')

for ii in range(1000):
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
