from robolearn.agents.agent import Agent
from robolearn.policies.policy_opt.policy_opt import PolicyOpt
from robolearn.policies.policy_opt.policy_opt_random import PolicyOptRandom
from robolearn.utils.sample.sample_list import SampleList
from robolearn.utils.data_logger import DataLogger
import copy


class GPSAgent(Agent):
    """
    GPSAgent class: An agent with samples attribute and policy_opt method.
    """
    def __init__(self, act_dim, obs_dim, state_dim, policy_opt=None, agent_name=""):
        super(GPSAgent, self).__init__(act_dim=act_dim, obs_dim=obs_dim, state_dim=state_dim)

        # TODO: We assume that an agent should remember his samples (experience??). Check if we include it in all agents
        self._samples = []  # List of lists, one list for each condition (sampled from a local policy)

        # Good and Bad experiences
        self._good_experience = []
        self._bad_experience = []

        if policy_opt is None:
            print("Policy optimization not defined. Using default PolicyOptRandom class!")
            self.policy_opt = PolicyOptRandom({}, self.obs_dim, self.act_dim)
        else:
            if not issubclass(policy_opt['type'], PolicyOpt):
                raise TypeError("'policy_opt' type %s is not a PolicyOpt class" % str(policy_opt['type']))
            policy_opt['hyperparams']['name'] = agent_name
            self.policy_opt = policy_opt['type'](policy_opt['hyperparams'], obs_dim, act_dim)

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

    # For pickling.
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'policy_opt' in state:
            state.pop('policy_opt')
        if 'policy' in state:
            state.pop('policy')
        return state

    # For unpickling.
    def __setstate__(self, state):
        self.__dict__.update(state)
        #self.__dict__ = state
        #self.__dict__['policy'] = None
