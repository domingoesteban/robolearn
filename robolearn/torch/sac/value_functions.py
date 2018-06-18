import torch
from robolearn.torch.pytorch_util import FloatTensor
from robolearn.torch.TOREMOVE_networks import FlattenMlp
from robolearn.core.serializable import Serializable
from robolearn.torch.core import PyTorchModule
import robolearn.torch.pytorch_util as ptu
from robolearn.torch.nn import LayerNorm
from torch.nn import functional as F
from robolearn.torch.nn import identity
from torch import nn as nn

# ########### #
# ########### #
# Q-Functions #
# ########### #
# ########### #


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

    def get_value(self, obs_np, act_np, deterministic=False):
        # TODO: CHECK IF INDEX 0
        values = self.get_values(obs_np[None], act_np[None],
                                 deterministic=deterministic)
        return values[0, :], {}

    def get_values(self, obs_np, act_np, deterministic=False):
        return self.eval_np(obs_np, act_np, deterministic=deterministic)


class NNMultiQFunction(PyTorchModule):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 n_qs,
                 shared_hidden_sizes,
                 unshared_hidden_sizes=None,
                 stds=None,
                 hidden_activation=F.relu,
                 output_activation=identity,
                 hidden_w_init=ptu.xavier_init,
                 hidden_b_init_val=0,
                 output_w_init=ptu.xavier_init,
                 output_b_init_val=0,
                 shared_layer_norm=False,
                 unshared_layer_norm=False,
                 layer_norm_kwargs=None,
                 **kwargs
                 ):

        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._n_qs = n_qs

        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        super(NNMultiQFunction, self).__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.shared_layer_norm = shared_layer_norm
        self.unshared_layer_norm = unshared_layer_norm
        self.fcs = []
        self.shared_layer_norms = []
        self.ufcs = [list() for _ in range(self._n_qs)]
        self.unshared_layer_norms = [list() for _ in range(self._n_qs)]
        self.last_fcs = []

        in_size = obs_dim + action_dim
        # Shared Layers
        for i, next_size in enumerate(shared_hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_w_init(fc.weight)
            ptu.fill(fc.bias, hidden_b_init_val)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.shared_layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("shared_layer_norm{}".format(i), ln)
                self.shared_layer_norms.append(ln)

        # Unshared Layers
        if unshared_hidden_sizes is not None:
            for i, next_size in enumerate(unshared_hidden_sizes):
                for q_idx in range(self._n_qs):
                    ufc = nn.Linear(in_size, next_size)
                    hidden_w_init(ufc.weight)
                    ptu.fill(ufc.bias, hidden_b_init_val)
                    self.__setattr__("ufc{}_{}".format(q_idx, i), ufc)
                    self.ufcs[q_idx].append(ufc)

                    if self.unshared_layer_norm:
                        ln = LayerNorm(next_size)
                        tmp_txt = "unshared_layer_norm{}_{}".format(q_idx, i)
                        self.__setattr__(tmp_txt, ln)
                        self.unshared_layer_norms[q_idx].append(ln)
                in_size = next_size

        for q_idx in range(self._n_qs):
            last_fc = nn.Linear(in_size, 1)
            output_w_init(last_fc.weight)
            ptu.fill(last_fc.bias, output_b_init_val)
            self.__setattr__("last_fc{}".format(q_idx), last_fc)
            self.last_fcs.append(last_fc)

    def forward(self, obs, act, val_idxs=None):
        if val_idxs is None:
            val_idxs = list(range(self._n_qs))

        h = torch.cat((obs, act), dim=-1)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))

        hs = [h for _ in val_idxs]
        # Unshared Layers
        if len(self.ufcs) > 0:
            for ii, idx in enumerate(val_idxs):
                for i, fc in enumerate(self.ufcs[idx]):
                    hs[ii] = self.hidden_activation(fc(hs[ii]))

        values = [self.last_fcs[idx](hs[ii])
                  for ii, idx in enumerate(val_idxs)]

        return values

    def get_value(self, obs_np, val_idxs=None):
        if val_idxs is None:
            val_idxs = list(range(self._n_qs))

        values = self.get_values(obs_np[None], val_idxs=val_idxs)
        # TODO: CHECK IF INDEX 0
        return values[0, :], {}

    def get_values(self, obs_np, val_idxs=None):
        if val_idxs is None:
            val_idxs = list(range(self._n_qs))

        return self.eval_np(obs_np, val_idxs=val_idxs)

    def get_n_heads(self):
        return len(self.last_fcs)


class WeightedNNMultiQFunction(PyTorchModule):
    def __init__(self, multiqvalue, val_idxs=None, weights=None):
        self.save_init_params(locals())
        super(WeightedNNMultiQFunction, self).__init__()

        self._multiqvalue = multiqvalue

        if val_idxs is None:
            n_heads = self._multiqvalue.get_n_heads()
            val_idxs = list(range(n_heads))
        self._val_idxs = val_idxs

        if weights is None:
            weights = [1. / len(self._val_idxs) for _ in val_idxs]
        self._weights = FloatTensor(weights)

    def forward(self, *nn_input):
        values = self._multiqvalue(*nn_input, val_idxs=self._val_idxs)

        weighted_action = torch.zeros_like(values[0])
        for ii in range(len(values)):
            weighted_action += self._weights[ii] * values[ii]

        return weighted_action

    def get_value(self, obs_np):
        # TODO: CHECK IF INDEX 0
        values = self.get_values(obs_np[None])

        return values[0, :], {}

    def get_values(self, obs_np):
        return self.eval_np(obs_np)


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


# ########### #
# ########### #
# V-Functions #
# ########### #
# ########### #


class NNVFunction(FlattenMlp, Serializable):
    def __init__(self,
                 obs_dim,
                 hidden_sizes=(100, 100)):

        self._obs_dim = obs_dim

        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        self.save_init_params(locals())
        FlattenMlp.__init__(self,
                            hidden_sizes=hidden_sizes,
                            input_size=obs_dim,
                            output_size=1,
                            )

    def get_value(self, obs_np, deterministic=False):
        # TODO: CHECK IF INDEX 0
        values = self.get_values(obs_np[None], deterministic=deterministic)
        return values[0, :], {}

    def get_values(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)


class AvgNNVFunction(PyTorchModule):
    def __init__(self, obs_dim, v_functions):
        self._obs_dim = obs_dim

        self._v_fcns = v_functions

        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        super(AvgNNVFunction, self).__init__()

    def forward(self, *inputs, **kwargs):
        all_outputs = torch.stack(
            [v_val(*inputs, **kwargs) for v_val in self._v_fcns],
            dim=-1).squeeze()
        avg_output = torch.mean(all_outputs, dim=-1, keepdim=True)

        return avg_output


class SumNNVFunction(PyTorchModule):
    def __init__(self, obs_dim, v_functions):
        self._obs_dim = obs_dim

        self._v_fcns = v_functions

        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        super(SumNNVFunction, self).__init__()

    def forward(self, *inputs, **kwargs):
        all_outputs = torch.stack(
            [v_val(*inputs, **kwargs) for v_val in self._v_fcns],
            dim=-1).squeeze()
        sum_output = torch.sum(all_outputs, dim=-1, keepdim=True)

        return sum_output


class NNMultiVFunction(PyTorchModule):
    def __init__(self,
                 obs_dim,
                 n_vs,
                 shared_hidden_sizes,
                 unshared_hidden_sizes=None,
                 hidden_activation=F.relu,
                 output_activation=identity,
                 hidden_w_init=ptu.xavier_init,
                 hidden_b_init_val=0,
                 output_w_init=ptu.xavier_init,
                 output_b_init_val=0,
                 shared_layer_norm=False,
                 unshared_layer_norm=False,
                 layer_norm_kwargs=None,
                 ):

        self._obs_dim = obs_dim
        self._n_vs = n_vs

        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        super(NNMultiVFunction, self).__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.shared_layer_norm = shared_layer_norm
        self.unshared_layer_norm = unshared_layer_norm
        self.fcs = []
        self.shared_layer_norms = []
        self.ufcs = [list() for _ in range(self._n_vs)]
        self.unshared_layer_norms = [list() for _ in range(self._n_vs)]
        self.last_fcs = []

        in_size = obs_dim
        # Shared Layers
        for i, next_size in enumerate(shared_hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_w_init(fc.weight)
            ptu.fill(fc.bias, hidden_b_init_val)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.shared_layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("shared_layer_norm{}".format(i), ln)
                self.shared_layer_norms.append(ln)

        # Unshared Layers
        if unshared_hidden_sizes is not None:
            for i, next_size in enumerate(unshared_hidden_sizes):
                for q_idx in range(self._n_vs):
                    ufc = nn.Linear(in_size, next_size)
                    hidden_w_init(ufc.weight)
                    ptu.fill(ufc.bias, hidden_b_init_val)
                    self.__setattr__("ufc{}_{}".format(q_idx, i), ufc)
                    self.ufcs[q_idx].append(ufc)

                    if self.unshared_layer_norm:
                        ln = LayerNorm(next_size)
                        tmp_txt = "unshared_layer_norm{}_{}".format(q_idx, i)
                        self.__setattr__(tmp_txt, ln)
                        self.unshared_layer_norms[q_idx].append(ln)
                in_size = next_size

        for q_idx in range(self._n_vs):
            last_fc = nn.Linear(in_size, 1)
            output_w_init(last_fc.weight)
            ptu.fill(last_fc.bias, output_b_init_val)
            self.__setattr__("last_fc{}".format(q_idx), last_fc)
            self.last_fcs.append(last_fc)

    def forward(self, obs, val_idxs=None):
        if val_idxs is None:
            val_idxs = list(range(self._n_vs))

        h = obs

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))

        hs = [h for _ in val_idxs]
        # Unshared Layers
        if len(self.ufcs) > 0:
            for ii, idx in enumerate(val_idxs):
                for i, fc in enumerate(self.ufcs[idx]):
                    hs[ii] = self.hidden_activation(fc(hs[ii]))

        values = [self.last_fcs[idx](hs[ii])
                  for ii, idx in enumerate(val_idxs)]

        return values

    def get_value(self, obs_np):
        # TODO: CHECK IF INDEX 0
        values = self.get_values(obs_np[None])
        return values[0, :], {}

    def get_values(self, obs_np):
        return self.eval_np(obs_np)

    def get_n_heads(self):
        return len(self.last_fcs)


class WeightedNNMultiVFunction(PyTorchModule):
    def __init__(self, multivalue, val_idxs=None, weights=None):
        self.save_init_params(locals())
        super(WeightedNNMultiVFunction, self).__init__()

        self._multivalue = multivalue

        if val_idxs is None:
            n_heads = self._multivalue.get_n_heads()
            val_idxs = list(range(n_heads))
        self._val_idxs = val_idxs

        if weights is None:
            weights = [1. / len(self._val_idxs) for _ in val_idxs]
        self._weights = FloatTensor(weights)

    def forward(self, *nn_input):
        values = self._multivalue(*nn_input, val_idxs=self._val_idxs)

        weighted_action = torch.zeros_like(values[0])
        for ii in range(len(values)):
            weighted_action += self._weights[ii] * values[ii]

        return weighted_action

    def get_value(self, obs_np):
        # TODO: CHECK IF INDEX 0
        values = self.get_values(obs_np[None])

        return values[0, :], {}

    def get_values(self, obs_np):
        return self.eval_np(obs_np)

