# Some basic imports and setup
import time
import numpy as np, numpy.random as nr, gym
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)
from robolearn.envs.simple_envs.cliff import CliffEnv

render = 'human'
# render = 'rgb_array'
env = CliffEnv(map_name='4x12', is_slippery=False,
               reward_dict={'G': -1, 'S': -1, 'C': -1, 'F': 0})

env.seed(0); from gym.spaces import prng; prng.seed(10)
if render != 'human':
    img_width = 240
    img_height = 240
    fig, ax = plt.subplots(1, 1)
    my_image = ax.imshow(0.5*np.ones((img_width, img_height, 3)),
                         interpolation='nearest', animated=True)
    fig.canvas.draw()
    plt.ion()
    fig.show()

# Generate the episode
env.reset()
for t in range(100):
    print('Iter %d' % t)
    time.sleep(0.5)
    rgb_image = env.render(mode=render)
    if render != 'human':
        my_image.set_data(rgb_image)
        ax.draw_artist(my_image)
        fig.canvas.blit(ax.bbox)  # redraw the axes rectangle
        plt.pause(0.0001)

    a = env.action_space.sample()
    ob, rew, done, _ = env.step(a)
    if done:
        break
assert done

rgb_image = env.render(mode=render)

if render != 'human':
    print(rgb_image.shape)
    my_image.set_data(rgb_image)
    ax.draw_artist(my_image)
    fig.canvas.blit(ax.bbox)  # redraw the axes rectangle
    plt.pause(0.0001)

input('Press a key to close')
env.render(close=True)
