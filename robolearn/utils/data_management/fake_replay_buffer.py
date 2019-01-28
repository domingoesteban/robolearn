import numpy as np

from robolearn.utils.data_management.replay_buffer import ReplayBuffer


class FakeReplayBuffer(ReplayBuffer):
    def __init__(self):
        pass

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        pass

    def terminate_episode(self):
        pass

    def random_batch(self, batch_size):
        pass

    def available_samples(self):
        return -1
