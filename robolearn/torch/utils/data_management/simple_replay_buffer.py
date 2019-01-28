import torch
from robolearn.torch.utils import pytorch_util as ptu

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

        self._obs_buffer = torch.zeros((max_size, obs_dim),
                                       dtype=torch.float32,
                                       device=ptu.device)
        self._next_obs_buffer = torch.zeros((max_size, obs_dim),
                                            dtype=torch.float32,
                                            device=ptu.device)
        self._acts_buffer = torch.zeros((max_size, action_dim),
                                        dtype=torch.float32,
                                        device=ptu.device)
        self._rewards_buffer = torch.zeros((max_size, 1),
                                           dtype=torch.float32,
                                           device=ptu.device)
        self._terms_buffer = torch.zeros((max_size, 1),
                                         dtype=torch.float32,
                                         device=ptu.device)

        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._max_size = max_size
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self._obs_buffer[self._top] = torch.as_tensor(observation)
        self._acts_buffer[self._top] = torch.as_tensor(action)
        self._rewards_buffer[self._top] = torch.as_tensor(reward)
        self._terms_buffer[self._top] = torch.as_tensor(terminal.astype(float))
        self._next_obs_buffer[self._top] = torch.as_tensor(next_observation)
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

        indices = torch.randint(0, self._size, (batch_size,))
        return dict(
            observations=self._obs_buffer[indices],
            actions=self._acts_buffer[indices],
            rewards=self._rewards_buffer[indices],
            terminals=self._terms_buffer[indices],
            next_observations=self._next_obs_buffer[indices],
        )

    def available_samples(self):
        return self._size
