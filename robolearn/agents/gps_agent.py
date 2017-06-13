from robolearn.agents.agent import Agent
from robolearn.policies.policy_opt.policy_opt import PolicyOpt
from robolearn.policies.policy_opt.policy_opt_random import PolicyOptRandom
from robolearn.utils.sample_list import SampleList


class GPSAgent(Agent):
    """
    GPSAgent class: An agent with samples and policy_opt.
    """
    def __init__(self, act_dim, obs_dim, state_dim, policy_opt=None, **kwargs):
        super(GPSAgent, self).__init__(act_dim=act_dim, obs_dim=obs_dim, state_dim=state_dim)

        # TODO: We assume that an agent should remember his samples (experience??). Check if we include it in all agents
        self._samples = []  # List of lists, one list for each condition (sampled from a local policy)

        if policy_opt is None:
            print("Policy optimization not defined. Using default PolicyOptRandom class!")
            policy_opt = PolicyOptRandom({}, self.obs_dim, self.act_dim)

        if not issubclass(type(policy_opt), PolicyOpt):
            raise TypeError("'policy_opt' argument is not a PolicyOpt class")

        self.policy_opt = policy_opt

        # Assign the internal policy in policy_opt class as agent's policy.
        self.policy = self.policy_opt.policy

    def get_samples(self, condition, start=0, end=None):
        """
        Return a SampleList object with the requested samples based on the start and end indices.
        :param condition: 
        :param start: Starting index of samples to return.
        :param end: End index of samples to return.
        :return: 
        """
        return SampleList(self._samples[condition][start:]) if end is None \
            else SampleList(self._samples[condition][start:end])

    def clear_samples(self, condition=None):
        """
        Reset the samples for a given condition, defaulting to all conditions.
        :param condition: Condition for which to reset samples. If not specified clean all the samples!
        :return: 
        """
        if condition is None:
            #self._samples = [[] for _ in range(self.conditions)]
            self._samples = []
        else:
            self._samples[condition] = []

    def add_sample(self, sample, condition):
        """
        Add a sample to the agent samples list.
        :param sample: Sample to be added
        :param condition: Condition ID
        :return: Sample id
        """
        # If it does not exist exist samples from that condition, create one
        if condition > len(self._samples)-1:
            self._samples.append(list())

        self._samples[condition].append(sample)

        return len(self._samples[condition]) - 1
