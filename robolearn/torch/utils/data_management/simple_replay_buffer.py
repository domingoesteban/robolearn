import numpy as np
from robolearn.torch.utils import pytorch_util as ptu

from robolearn.utils.data_management.simple_replay_buffer \
    import SimpleReplayBuffer as NpSimpleReplayBuffer


class SimpleReplayBuffer(NpSimpleReplayBuffer):
    def __init__(self, max_replay_buffer_size, obs_dim, action_dim):
        NpSimpleReplayBuffer.__init__(
            self, max_replay_buffer_size, obs_dim, action_dim
        )

    def random_batch(self, batch_size):
        batch = NpSimpleReplayBuffer.random_batch(
            self, batch_size=batch_size
        )

        return ptu.np_to_pytorch_batch(batch)
