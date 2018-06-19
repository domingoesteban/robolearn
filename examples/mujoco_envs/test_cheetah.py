import gym
import time

env = gym.make('HalfCheetah-v2')

env.reset()
env.render()
ii = 0
while True:
    time.sleep(0.1)
    ii += 1
    print(ii)
    env.step(env.action_space.sample())
    env.render()
print("ME CERRRROOO")
