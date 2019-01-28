from robolearn.torch.utils import pytorch_util as ptu

from robolearn.utils.data_management.multigoal_replay_buffer \
    import MultiGoalReplayBuffer as NumpyMultiGoalReplayBuffer


class MultiGoalReplayBuffer(NumpyMultiGoalReplayBuffer):
    def __init__(self, max_replay_buffer_size, obs_dim, action_dim,
                 reward_vector_size):
        NumpyMultiGoalReplayBuffer.__init__(
            self, max_replay_buffer_size, obs_dim, action_dim,
            reward_vector_size
        )
        self._torch_obs_buffer = ptu.from_numpy(self._obs_buffer)
        self._torch_next_obs_buffer = ptu.from_numpy(self._next_obs_buffer)
        self._torch_acts_buffer = ptu.from_numpy(self._acts_buffer)
        self._torch_rewards_buffer = ptu.from_numpy(self._rewards_buffer)
        self._torch_terminals_buffer = ptu.from_numpy(self._terminals_buffer)
        self._torch_rew_vect_buffer = ptu.from_numpy(self._rewards_buffer)
        self._torch_term_vect_buffer = ptu.from_numpy(self._terminals_buffer)

    def random_batch(self, batch_size):
        if batch_size > self._size:
            raise AttributeError('Not enough samples to get. %d bigger than '
                                 'current %d!' % (batch_size, self._size))

        indices = self.random_indices(0, self._size, batch_size)
        return dict(
            observations=self._torch_obs_buffer[indices],
            actions=self._torch_acts_buffer[indices],
            rewards=self._torch_rewards_buffer[indices],
            terminals=self._torch_terminals_buffer[indices],
            next_observations=self._torch_next_obs_buffer[indices],
            reward_vectors=self._torch_rew_vect_buffer[indices],
            terminal_vectors=self._torch_term_vect_buffer[indices],
        )

