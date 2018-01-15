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
from robolearn.agents.ilqr_agent import ILQRAgent


if __name__ == '__main__':
    EndTime = 1.0
    env_with_img = False
    obs_like_mjc = True
    ntargets = 2
    tgt_weights = [1.0, -1.0]
    rdn_tgt_pos = False
    tgt_positions = [(0.1, 0.2), (-0.1, -0.2)]  # Values between [-0.2, 0.2]
    rdn_init_cfg = False
    # env = gym.make('ReacherBullet-v5')
    env = ReacherBulletEnv(render=False, obs_with_img=env_with_img, obs_mjc_gym=obs_like_mjc, ntargets=ntargets,
                           rdn_tgt_pos=rdn_tgt_pos)
    env.seed(0)

    episode_count = 5
    reward = 0
    done = False
    img_width = 256
    img_height = 256
    env.change_img_size(height=img_height, width=img_width)
    env.set_tgt_cost_weights(tgt_weights)
    env.set_tgt_pos(tgt_positions)

    ts = env.dt
    total_steps = int(EndTime/ts)

    # Agent
    agent = ILQRAgent(env.action_space.shape[0], env.observation_space.shape[0], total_steps)
    agent.seed(5)
    print(agent.obs_dim)
    print(agent.act_dim)
    input('saadsfsdhfkj')

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

        steps_counter = 0

        # while True:
        while steps_counter < total_steps:
            print('external_counter', steps_counter)
            action = agent.act(ob, reward, done) * 0.001

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
            steps_counter += 1

    env.close()
    input('Press a key to finish the script...')
