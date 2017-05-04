class Agent(object):
    """
    Agent base class
    """

    def __init__(self, act_dim, obs_dim):
        self.act_dim = act_dim
        self.obs_dim = obs_dim

    def act(self, obs):
        """
        Return the action given the current policy
        :param obs: Environment observations
        :return:
        """
        return self.policy(obs)

    def policy(self, state):
        """
        Function that maps state to action
        :param state:
        :return:
        """
        NotImplementedError

    def train(self, history):
        raise NotImplementedError


