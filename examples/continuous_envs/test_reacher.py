import pybullet as pb
import numpy as np
import matplotlib.pyplot as plt
import time
import gym
from gym import spaces
from gym.utils import seeding

# from gym.envs.registration import register
# register(
#     id='ReacherBullet-v5',
#     entry_point='robolearn.envs.reacher.reacher_env:ReacherBulletEnv')


from robolearn.envs.reacher import ReacherBulletEnv


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    # env = gym.make('ReacherBullet-v5')
    env = ReacherBulletEnv(render=False, obs_with_img=True)

    env.seed(0)

    agent = RandomAgent(env.action_space)

    episode_count = 5
    reward = 0
    done = False
    img_width = 320
    img_height = 320

    # fig, ax = plt.subplots(1, 1)
    # my_image = ax.imshow(np.zeros((img_width, img_height, 3)), interpolation='nearest', animated=True)
    # fig.canvas.draw()
    # background = fig.canvas.copy_from_bbox(ax.bbox)  # cache the background
    # plt.ion()
    # plt.show()

    # env.render(mode='human')  # Only if we want at the beginning
    for i in range(episode_count):
        # input('Press key to reset episode %d/%d' % (i+1, episode_count))
        ob = env.reset()
        input('Press key to start episode %d/%d' % (i+1, episode_count))

        EndTime = 1.0
        ts = env.dt
        total_steps = EndTime/ts
        steps_counter = 0

        # while True:
        while steps_counter < total_steps:
            action = agent.act(ob, reward, done) * 0.001
            print(action)
            ob, reward, done, _ = env.step(action)
            if done:
                print('ENVIRONMENT DONE!!!')
                break
            # env.render()

            # dim_img_data = img_width*img_height*3
            # rgb_image = ob[-dim_img_data:].astype(np.uint8).reshape(img_width, img_height, 3)
            # # rgb_image = env.render(mode='rgb_array')
            # my_image.set_data(rgb_image)
            # fig.canvas.restore_region(background)  # restore background
            # ax.draw_artist(my_image)
            # fig.canvas.blit(ax.bbox)  # redraw the axes rectangle
            # # fig.canvas.draw()

            # plt.pause(1./100.)
            # time.sleep(1./100.)

    env.close()
    input('Press a key to finish the script...')
