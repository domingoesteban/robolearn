import torch
from robolearn.torch.utils import pytorch_util as ptu

from robolearn.utils.data_management.replay_buffer import ReplayBuffer


class MultiGoalReplayBuffer(ReplayBuffer):
    def __init__(self, max_replay_buffer_size, obs_dim, action_dim,
                 reward_vector_size):
        if not max_replay_buffer_size > 1:
            raise ValueError("Invalid Maximum Replay Buffer Size: {}".format(
                max_replay_buffer_size)
            )
        if not reward_vector_size > 0:
            raise ValueError("Invalid Reward Vector Size: {}".format(
                reward_vector_size)
            )

        max_size = int(max_replay_buffer_size)
        multi_size = int(reward_vector_size)

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
        self._terminals_buffer = torch.zeros((max_size, 1),
                                             dtype=torch.float32,
                                             device=ptu.device)
        self._rew_vects_buffer = torch.zeros((max_size, multi_size),
                                             dtype=torch.float32,
                                             device=ptu.device)
        self._term_vects_buffer = torch.zeros((max_size, multi_size),
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
        self._rew_vects_buffer[self._top] = \
            torch.as_tensor(kwargs['env_info']['reward_multigoal'])
        self._term_vects_buffer[self._top] = \
            torch.as_tensor(kwargs['env_info']['terminal_multigoal'].astype(float))
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
            terminals=self._terminals_buffer[indices],
            next_observations=self._next_obs_buffer[indices],
            reward_vectors=self._rew_vects_buffer[indices],
            terminal_vectors=self._term_vects_buffer[indices],
        )

    def available_samples(self):
        return self._size
