import numpy as np
import argparse

from robolearn.old_envs.centauro_pb import CentauroBulletEnv
from robolearn.old_agents.random_gym_agent import RandomGymAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg1', type=int, default=60)
    parser.add_argument('--arg2', type=str, default='string1')
    parser.add_argument('--env_with_img', action="store_true", dest='env_with_img',
                        default=False)

    training_rollouts = 30
    validation_rollouts = 10

    # Initialize environment
    render = True
    # render = False
    EndTime = 10.0
    env_with_img = False
    active_joints = 'LA'
    control_type = 'torque'
    env = CentauroBulletEnv(render=render, active_joints=active_joints,
                            control_type=control_type)
    env.seed(0)

    ts = env.dt
    total_steps = int(EndTime/ts)

    # Agent
    agent = RandomGymAgent(env.action_space)
    # agent.seed(5)
    # print(agent.obs_dim)
    # print(agent.act_dim)
    # input('saadsfsdhfkj')

    # Collect initial training data
    # env.render(mode='human')  # Only if we want at the beginning
    print('Generating training data')
    max_rollouts = training_rollouts
    for rr in range(max_rollouts):
        # input('Press key to reset episode %d/%d' % (i+1, episode_count))
        ob = env.reset()
        reward = None
        done = False
        input('Press key to start episode %d/%d' % (rr+1, max_rollouts))

        steps_counter = 0

        # while True:
        while steps_counter < total_steps:
            print('external_counter', steps_counter,
                  ' | mx_steps:', total_steps)
            action = agent.act(ob, reward, done) * 0.001

            obs, reward, done, _ = env.step(action)
            if done:
                print('ENVIRONMENT DONE!!!')
                break
            # env.render()

            # if env_with_img:
            #     dim_img_data = img_width*img_height*3
            #     rgb_image = obs[-dim_img_data:].astype(np.uint8).reshape(img_width, img_height, 3)
            # else:
            #     rgb_image = env.render(mode='rgb_array')
            # my_image.set_data(rgb_image)
            # fig.canvas.restore_region(background)  # restore background
            # ax.draw_artist(my_image)
            # fig.canvas.blit(ax.bbox)  # redraw the axes rectangle
            # # fig.canvas.draw()

            # plt.pause(1./100.)
            # time.sleep(1./100.)
            steps_counter += 1

    env.close()
    input('Press a key to finish the script...')



if __name__ == '__main__':
    main()
