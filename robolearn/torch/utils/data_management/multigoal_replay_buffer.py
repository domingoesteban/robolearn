from robolearn.torch.utils import pytorch_util as ptu

from robolearn.utils.data_management.multigoal_replay_buffer \
    import MultiGoalReplayBuffer as NpMultiGoalReplayBuffer


class MultiGoalReplayBuffer(NpMultiGoalReplayBuffer):
    def __init__(self, max_replay_buffer_size, obs_dim, action_dim,
                 reward_vector_size):
        NpMultiGoalReplayBuffer.__init__(
            self, max_replay_buffer_size, obs_dim, action_dim,
            reward_vector_size
        )

    def random_batch(self, batch_size):
        batch = NpMultiGoalReplayBuffer.random_batch(
            self, batch_size=batch_size
        )

        return ptu.np_to_pytorch_batch(batch)
