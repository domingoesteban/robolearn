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


from robolearn.old_envs.reacher import ReacherBulletEnv
from robolearn.old_agents.random_gym_agent import RandomGymAgent


if __name__ == '__main__':
    env_with_img = False
    obs_like_mjc = True
    ntargets = 2
    tgt_weights = [1.0, -1.0]
    rdn_tgt_pos = True
    tgt_positions = [(0.1, 0.2), (-0.1, -0.2)]  # Values between [-0.2, 0.2]
    # env = gym.make('ReacherBullet-v5')
    env = ReacherBulletEnv(render=False, obs_with_img=env_with_img, obs_mjc_gym=obs_like_mjc, ntargets=ntargets,
                           rdn_tgt_pos=rdn_tgt_pos)

    env.seed(0)

    agent = RandomGymAgent(env.action_space)

    episode_count = 5
    reward = 0
    done = False
    img_width = 320
    img_height = 320
    env.change_img_size(height=img_height, width=img_width)
    env.set_tgt_cost_weights(tgt_weights)
    env.set_tgt_pos(tgt_positions)

    fig, ax = plt.subplots(1, 1)
    my_image = ax.imshow(np.zeros((img_width, img_height, 3)), interpolation='nearest', animated=True)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)  # cache the background
    plt.ion()
    plt.show()

    # env.render(mode='human')  # Only if we want at the beginning
    for i in range(episode_count):
        # input('Press key to reset episode %d/%d' % (i+1, episode_count))
        ob = env.reset()
        input('Press key to start episode %d/%d' % (i+1, episode_count))

        EndTime = 1.0
        ts = env.dt
        total_steps = EndTime/ts
        steps_counter = 0

        temp_obs = []
        temp_obs2 = []
        temp_obs3 = []
        # while True:
        while steps_counter < total_steps:
            action = agent.act(ob, reward, done) * 0.001
            action = np.zeros_like(action)
            action[0] = 0.005

            obs, reward, done, _ = env.step(action)
            if done:
                print('ENVIRONMENT DONE!!!')
                break
            # env.render()

            if env_with_img:
                dim_img_data = img_width*img_height*3
                rgb_image = obs[-dim_img_data:].astype(np.uint8).reshape(img_width, img_height, 3)
            else:
                rgb_image = env.render(mode='rgb_array')
            my_image.set_data(rgb_image)
            fig.canvas.restore_region(background)  # restore background
            ax.draw_artist(my_image)
            fig.canvas.blit(ax.bbox)  # redraw the axes rectangle
            # fig.canvas.draw()

            # plt.pause(1./100.)
            # time.sleep(1./100.)

        fig = plt.figure()
        plt.plot(temp_obs)
        fig = plt.figure()
        plt.plot(temp_obs2)
        fig = plt.figure()
        plt.plot(temp_obs3)

    env.close()
    input('Press a key to finish the script...')
