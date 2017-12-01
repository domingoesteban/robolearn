class Agent(object):
    """
    Agent base class
    """

    def __init__(self, act_dim, obs_dim, state_dim=None):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        if state_dim is None:
            state_dim = obs_dim
        self.state_dim = state_dim

    def act(self, **kwargs):
        """
        Return the action given the current policy.
        :param obs: Environment observations
        :return:
        """
        return self.policy.eval(**kwargs)

    def policy(self, **kwargs):
        """
        Function that maps state to action
        :param state:
        :return:
        """
        raise NotImplementedError

