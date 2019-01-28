import numpy as np

from robolearn.utils.data_management.replay_buffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(
            self, max_replay_buffer_size, obs_dim, action_dim,
    ):
        if not max_replay_buffer_size > 1:
            raise ValueError("Invalid Maximum Replay Buffer Size: {}".format(
                max_replay_buffer_size)
            )

        max_size = int(max_replay_buffer_size)

        self._obs_buffer = np.zeros((max_size, obs_dim), dtype=np.float32)
        self._next_obs_buffer = np.zeros((max_size, obs_dim), dtype=np.float32)
        self._acts_buffer = np.zeros((max_size, action_dim), dtype=np.float32)
        self._rewards_buffer = np.zeros((max_size, 1), dtype=np.float32)
        self._terminals_buffer = np.zeros((max_size, 1), dtype='uint8')

        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._max_size = max_size
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self._obs_buffer[self._top] = observation
        self._acts_buffer[self._top] = action
        self._rewards_buffer[self._top] = reward
        self._terminals_buffer[self._top] = terminal
        self._next_obs_buffer[self._top] = next_observation
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_size
        if self._size < self._max_size:
            self._size += 1

    def random_batch(self, batch_size):
        if batch_size > self._size:
            raise AttributeError('Not enough samples to get. %d bigger than '
                                 'current %d!' % (batch_size, self._size))

        indices = self.random_indices(0, self._size, batch_size)
        return dict(
            observations=self._obs_buffer[indices],
            actions=self._acts_buffer[indices],
            rewards=self._rewards_buffer[indices],
            terminals=self._terminals_buffer[indices],
            next_observations=self._next_obs_buffer[indices],
        )

    def available_samples(self):
        return self._size

    @staticmethod
    def random_indices(low, high, size):
        return np.random.randint(low, high, size)
