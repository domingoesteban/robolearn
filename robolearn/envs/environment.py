class Environment(object):
    def send_action(self, action):
        NotImplementedError

    def read_observation(self):
        NotImplementedError

    def get_reward(self):
        NotImplementedError

    def reset(self):
        NotImplementedError


class ActionManager(object):
    pass


class ObservationManager(object):
    pass


class RewardManager(object):
    pass


class EnvInterface(object):
    def __init__(self):
        # General Environment properties
        self.obs_dim = 0
        self.act_dim = 0

    def send_action(self, *args):
        NotImplementedError

    def get_observation(self, *args):
        NotImplementedError