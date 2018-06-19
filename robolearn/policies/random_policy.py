from robolearn.policies.base import SerializablePolicy


class RandomPolicy(SerializablePolicy):
    """
    Policy that samples an action from action space.
    """

    def __init__(self, action_space):

        self.action_space = action_space

        super(RandomPolicy, self).__init__(action_dim=action_space.n)

    def get_action(self, obs):
        return self.action_space.sample(), {}
