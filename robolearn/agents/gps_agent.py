from robolearn.agents.agent import Agent
from robolearn.policies.policy import Policy
from robolearn.policies.random_policy import RandomPolicy

import numpy as np


class GPSAgent(Agent):
    def __init__(self, act_dim, obs_dim, state_dim, policy=None):
        super(GPSAgent, self).__init__(act_dim=act_dim, obs_dim=obs_dim, state_dim=state_dim)

        if policy is None:
            self.policy = RandomPolicy(self.act_dim)
            #raise ValueError("Policy has not been defined!")

        #if not issubclass(type(policy), Policy):
        #    raise TypeError("Policy argument is not a Policy class")



    def sample(self, **kwargs):
        pass
