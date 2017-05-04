from robolearn.agents.agent import Agent
from robolearn.policies.policy import Policy

import numpy as np


class GPSAgent(Agent):
    def __init__(self, act_dim, obs_dim, state_dim, policy=None):
        super(GPSAgent, self).__init__(act_dim=act_dim, obs_dim=obs_dim, state_dim=state_dim)

        if issubclass(type(policy), Policy):
            raise TypeError("Policy argument is not a Policy class")
        self.policy = policy

    def sample(self, **kwargs):
        pass
