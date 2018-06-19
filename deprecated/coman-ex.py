from robolearn.old_agents import LinearTFAgent

import numpy as np

print "*"*20
print " Running coman-ex"
print "*"*20

action_space_size = 2
state_space_size = 20

obs = np.random.rand(1, state_space_size)

agent = LinearTFAgent(action_space_size, state_space_size)

print agent.act_dim

print agent.act(obs)

