from robolearn.agents.agent import Agent
from robolearn.policies.policy import Policy
from robolearn.policies.policy_opt.policy_opt import PolicyOpt
from robolearn.policies.policy_opt.policy_opt_random import PolicyOptRandom

from robolearn.utils.sample_list import SampleList

import numpy as np


class GPSAgent(Agent):
    def __init__(self, act_dim, obs_dim, state_dim, policy=None, **kwargs):
        super(GPSAgent, self).__init__(act_dim=act_dim, obs_dim=obs_dim, state_dim=state_dim)

        # TODO: We assume that an agent should remember his samples (experience??). Check if we include it in all agents
        if 'conditions' in kwargs:
            setattr(self, 'conditions', kwargs['conditions'])
        else:
            setattr(self, 'conditions', 1)

        self._samples = [[] for _ in range(self.conditions)]

        if policy is None:
            policy = PolicyOptRandom({}, self.obs_dim, self.act_dim)
            #raise ValueError("Policy has not been defined!")

        if not issubclass(type(policy), Policy) and not issubclass(type(policy), PolicyOpt):
            raise TypeError("Policy argument is neither a Policy or PolicyOpt class")
        self.policy = policy

    def get_samples(self, condition, start=0, end=None):
        """
        Return the requested samples based on the start and end indices.
        Args:
            start: Starting index of samples to return.
            end: End index of samples to return.
        """
        return (SampleList(self._samples[condition][start:]) if end is None
                else SampleList(self._samples[condition][start:end]))

    def clear_samples(self, condition=None):
        """
        Reset the samples for a given condition, defaulting to all conditions.
        Args:
            condition: Condition for which to reset samples.
        """
        if condition is None:
            self._samples = [[] for _ in range(self.conditions)]
        else:
            self._samples[condition] = []

    def act(self, **kwargs):
        """
        Return the action given the current policy
        :param obs: Environment observations
        :return:
        """
        #return self.policy(kwargs)
        return self.policy.policy.act(**kwargs)
