import numpy as np
from robolearn.torch.core import PyTorchModule
from robolearn.core.serializable import Serializable
from robolearn.torch.core import np_ify
import robolearn.torch.pytorch_util as ptu
import torch
import torch.nn as nn
import torch.nn.functional as F

from robolearn.torch.nn import identity
from robolearn.torch.nn import LayerNorm
from robolearn.models import QFunction


class NNMultiQFunction(PyTorchModule, QFunction):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 n_qs,
                 shared_hidden_sizes=None,
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

        QFunction.__init__(self, obs_dim=obs_dim, action_dim=action_dim)

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
        self.sfcs = []
        self.shared_layer_norms = []
        self.ufcs = [list() for _ in range(self._n_qs)]
        self.unshared_layer_norms = [list() for _ in range(self._n_qs)]
        self.last_fcs = []

        in_size = obs_dim + action_dim
        # Shared Layers
        if shared_hidden_sizes is not None:
            for i, next_size in enumerate(shared_hidden_sizes):
                fc = nn.Linear(in_size, next_size)
                # hidden_w_init(fc.weight)
                nn.init.xavier_normal_(fc.weight.data,
                                       gain=nn.init.calculate_gain('relu'))
                ptu.fill(fc.bias, hidden_b_init_val)
                self.__setattr__("sfc{}".format(i), fc)
                self.sfcs.append(fc)

                if self.shared_layer_norm:
                    ln = LayerNorm(next_size)
                    self.__setattr__("shared_layer_norm{}".format(i), ln)
                    self.shared_layer_norms.append(ln)
                in_size = next_size

        # Unshared Layers
        if unshared_hidden_sizes is not None:
            for i, next_size in enumerate(unshared_hidden_sizes):
                for q_idx in range(self._n_qs):
                    ufc = nn.Linear(in_size, next_size)
                    # hidden_w_init(ufc.weight)
                    nn.init.xavier_normal_(ufc.weight.data,
                                           gain=nn.init.calculate_gain('relu'))
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
            # output_w_init(last_fc.weight)
            nn.init.xavier_normal_(last_fc.weight.data,
                                   gain=nn.init.calculate_gain('linear'))
            ptu.fill(last_fc.bias, output_b_init_val)
            self.__setattr__("last_fc{}".format(q_idx), last_fc)
            self.last_fcs.append(last_fc)

        # print('TODOOO: SETTING MULTIQ-FCN INIT VALS')
        # init_w = 1e-4
        # for param in self.parameters():
        #     param.data.uniform_(-init_w, init_w)

    def forward(self, obs, act, val_idxs=None):
        """

        Args:
            obs (Tensor): Observation(s)
            act (Tensor): Action(s)
            val_idxs (iterable):

        Returns:
            values (list)
            info (dict): empty dictionary

        """
        if val_idxs is None:
            val_idxs = list(range(self._n_qs))

        h = torch.cat((obs, act), dim=-1)

        # Shared layers
        for i, fc in enumerate(self.sfcs):
            h = self.hidden_activation(fc(h))

        hs = [h.clone() for _ in val_idxs]
        # Unshared Layers
        if len(self.ufcs) > 0:
            for ii, idx in enumerate(val_idxs):
                for i, fc in enumerate(self.ufcs[idx]):
                    hs[ii] = self.hidden_activation(fc(hs[ii]))

        values = [self.output_activation(self.last_fcs[idx](hs[ii]))
                  for ii, idx in enumerate(val_idxs)]

        return values, dict()

    def get_value(self, obs_np, action_np, val_idxs=None):
        if val_idxs is None:
            val_idxs = list(range(self._n_qs))

        values, info_dict = self.get_values(obs_np[None], action_np[None],
                                            val_idxs=val_idxs)

        values = [value[0, :] for value in values]

        for key, vals in info_dict.items():
            info_dict[key] = [val[0, :] if isinstance(val, np.ndarray)
                              else None for val in vals]

        return values, info_dict

    def get_values(self, obs_np, action_np, val_idxs=None):
        if val_idxs is None:
            val_idxs = list(range(self._n_qs))

        values, info_dict = self.eval_np(obs_np, action_np, val_idxs=val_idxs)

        values = [np_ify(tensor) for tensor in values]

        for key, vals in info_dict.items():
            info_dict[key] = [np_ify(val) for val in vals]

        return values, info_dict

    @property
    def n_heads(self):
        return self._n_vs
