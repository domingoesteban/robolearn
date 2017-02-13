class Agent(object):
    """
    Agent base class
    """

    def __init__(self, act_dim, obs_dim):
        self.act_dim = act_dim
        self.obs_dim = obs_dim

    def act(self, obs):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


