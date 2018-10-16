import numpy as np

from robolearn.utils.data_management.replay_buffer import ReplayBuffer


class MultiGoalReplayBuffer(ReplayBuffer):
    def __init__(self, max_replay_buffer_size, obs_dim, action_dim,
                 reward_vector_size):
        if not max_replay_buffer_size > 1:
            raise ValueError("Invalid Maximum Replay Buffer Size: {}".format(
                max_replay_buffer_size)
            )
        if not obs_dim > 1:
            raise ValueError("Invalid Observation Dimension: {}".format(
                obs_dim)
            )
        if not action_dim > 1:
            raise ValueError("Invalid Action Dimension: {}".format(
                action_dim)
            )
        if not reward_vector_size > 1:
            raise ValueError("Invalid Reward Vector Size: {}".format(
                reward_vector_size)
            )

        max_replay_buffer_size = int(max_replay_buffer_size)
        reward_vector_size = int(reward_vector_size)
        self._obs_dim = obs_dim
        self._action_dim = action_dim

        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, obs_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, obs_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        self._reward_vectors = np.zeros((max_replay_buffer_size,
                                         reward_vector_size))
        # self._terminals[t] = a terminal was received at time t
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self._terminal_vectors = np.zeros((max_replay_buffer_size,
                                           reward_vector_size), dtype='uint8')
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation
        self._reward_vectors[self._top] = kwargs['env_info']['reward_multigoal']
        self._terminal_vectors[self._top] = \
            kwargs['env_info']['terminal_multigoal']
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        if batch_size > self._size:
            raise AttributeError('Not enough samples to get. %d bigger than '
                                 'current %d!' % (batch_size, self._size))

        indices = np.random.randint(0, self._size, batch_size)
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            reward_vectors=self._reward_vectors[indices],
            terminal_vectors=self._terminal_vectors[indices],
        )

    def num_steps_can_sample(self):
        return self._size
