class Environment(object):
    def send_action(self):
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
    def send_action(self):
        NotImplementedError

    def read_observation(self):
        NotImplementedError