import torch
from robolearn.torch.TOREMOVE_networks import FlattenMlp
from robolearn.core.serializable import Serializable
from robolearn.torch.core import PyTorchModule


class NNQFunction(FlattenMlp, Serializable):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_sizes=(100, 100)):

        self._obs_dim = obs_dim
        self._action_dim = action_dim

        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        self.save_init_params(locals())
        FlattenMlp.__init__(self,
                            hidden_sizes=hidden_sizes,
                            input_size=obs_dim+action_dim,
                            output_size=1,
                            )

    def get_value(self, obs_np, deterministic=False):
        # TODO: CHECK IF INDEX 0
        actions = self.get_values(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_values(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)


class AvgNNQFunction(PyTorchModule):
    def __init__(self, obs_dim, action_dim, q_functions):
        self._obs_dim = obs_dim
        self._action_dim = action_dim

        self._q_fcns = q_functions

        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        super(AvgNNQFunction, self).__init__()

    def forward(self, *inputs, **kwargs):
        all_outputs = torch.stack(
            [q_val(*inputs, **kwargs) for q_val in self._q_fcns],
            dim=-1).squeeze()
        avg_output = torch.mean(all_outputs, dim=-1, keepdim=True)

        return avg_output


class SumNNQFunction(PyTorchModule):
    def __init__(self, obs_dim, action_dim, q_functions):
        self._obs_dim = obs_dim
        self._action_dim = action_dim

        self._q_fcns = q_functions

        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        super(SumNNQFunction, self).__init__()

    def forward(self, *inputs, **kwargs):
        all_outputs = torch.stack(
            [q_val(*inputs, **kwargs) for q_val in self._q_fcns],
            dim=-1).squeeze()
        sum_output = torch.sum(all_outputs, dim=-1, keepdim=True)

        return sum_output
