from robolearn.agents.agent import Agent


class RandomGymAgent(Agent):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        super(RandomGymAgent, self).__init__(action_space.shape[0], None, state_dim=None)
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()
