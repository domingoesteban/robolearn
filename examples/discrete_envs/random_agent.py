import pybullet as pb
import numpy as np
import matplotlib.pyplot as plt
import time
import gym
from gym import spaces
from gym.utils import seeding

from pybullet_envs.robot_bases import BodyPart, Joint, Pose_Helper

# from gym.envs.registration import register
# register(
#     id='ReacherBullet-v5',
#     entry_point='robolearn.envs.reacher.reacher_env:ReacherBulletEnv')


from robolearn.envs.reacher import ReacherBulletEnv
from robolearn.envs.r2d2 import R2D2BulletEnv
from robolearn.envs.bigman_pb import BigmanBulletEnv
from robolearn.envs.frozen_lake import FrozenLakeEnv
from robolearn.envs.bigman_pb.bigman_robot import BIGMAN_INIT_CONFIG


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    # env = gym.make('ReacherBullet-v5')

    # env = ReacherBulletEnv(render=True)
    # env = R2D2BulletEnv(render=True)
    # env = BigmanBulletEnv(render=True)
    env = FrozenLakeEnv(map_name="4x4", is_slippery=True)

    env.seed(0)

    agent = RandomAgent(env.action_space)

    episode_count = 5
    reward = 0
    done = False

    # fig, ax = plt.subplots(1, 1)
    # my_image = ax.imshow(np.zeros((320, 320, 3)), interpolation='nearest', animated=True)
    # fig.canvas.draw()
    # background = fig.canvas.copy_from_bbox(ax.bbox)  # cache the background
    # plt.ion()
    # plt.show()

    # env.render(mode='human')  # Only if we want at the beginning
    for i in range(episode_count):
        input('Press key to reset episode %d/%d' % (i+1, episode_count))
        ob = env.reset()
        input('Press key to start episode %d/%d' % (i+1, episode_count))
        action = np.array(BIGMAN_INIT_CONFIG)

        EndTime = 5.0
        init_pos = action
        final_pos = np.zeros_like(init_pos)#action[18] + np.deg2rad(90)
        ts = env.dt
        total_steps = EndTime/ts
        steps_counter = 0

        # while True:
        while steps_counter < total_steps:
            steps_counter += 1
            # action = agent.act(ob, reward, done) * 0.001
            # action[:] += (final_pos - init_pos)/total_steps
            action = env.action_space.sample()
            # print('Agent obs:', ob, '| reward:', reward, '| action', action)
            ob, reward, done, _ = env.step(action)
            # print(env._robot.robot_body.get_pose())
            # print(pb.getBasePositionAndOrientation(env.drill_uid[-1]))
            # print('---')
            # input('---')
            if done:
                print('ENVIRONMENT DONE!!!')
                break
            env.render()

            # rgb_image = env.render(mode='rgb_array')
            # my_image.set_data(rgb_image)
            # fig.canvas.restore_region(background)  # restore background
            # ax.draw_artist(my_image)
            # fig.canvas.blit(ax.bbox)  # redraw the axes rectangle
            # # fig.canvas.draw()

            # plt.pause(1./100.)
            # time.sleep(1./100.)

    env.close()
    input('Press a key to finish the script...')
