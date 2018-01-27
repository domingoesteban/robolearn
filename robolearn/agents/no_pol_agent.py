from robolearn.agents.agent import Agent


class NoPolAgent(Agent):
    """
    Agent without any policy. Used mainly for sampling for env.
    """
    def __init__(self, act_dim, obs_dim, state_dim, agent_name=""):
        super(NoPolAgent, self).__init__(act_dim=act_dim, obs_dim=obs_dim,
                                         state_dim=state_dim)

        self.policy = None
