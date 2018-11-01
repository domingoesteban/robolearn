import numpy as np
from robolearn.torch.core import PyTorchModule
from robolearn.core.serializable import Serializable
from robolearn.torch.core import np_ify
import robolearn.torch.pytorch_util as ptu
import torch
import torch.nn as nn

from robolearn.torch.nn import LayerNorm
from robolearn.models import QFunction


class NNMultiQFunction(PyTorchModule, QFunction):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 n_qs,
                 shared_hidden_sizes=None,
                 unshared_hidden_sizes=None,
                 hidden_activation='relu',
                 hidden_w_init='xavier_normal',
                 hidden_b_init_val=0,
                 output_activation='linear',
                 output_w_init='xavier_normal',
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

        self._hidden_activation = ptu.get_activation(hidden_activation)
        self._output_activation = ptu.get_activation(output_activation)
        self._shared_layer_norm = shared_layer_norm
        self._unshared_layer_norm = unshared_layer_norm
        self._sfcs = []
        self._sfc_norms = []
        self._ufcs = [list() for _ in range(self._n_qs)]
        self._ufc_norms = [list() for _ in range(self._n_qs)]
        self._ufcs_lasts = []

        in_size = obs_dim + action_dim
        # Shared Layers
        if shared_hidden_sizes is not None:
            for ii, next_size in enumerate(shared_hidden_sizes):
                sfc = nn.Linear(in_size, next_size)
                ptu.layer_init_xavier_normal(layer=sfc,
                                             activation=hidden_activation,
                                             b=hidden_b_init_val)
                self.__setattr__("sfc{}".format(ii), sfc)
                self._sfcs.append(sfc)

                if self._shared_layer_norm:
                    ln = LayerNorm(next_size)
                    self.__setattr__("sfc{}_norm".format(ii), ln)
                    self._sfc_norms.append(ln)
                in_size = next_size

        # Unshared Layers
        if unshared_hidden_sizes is not None:
            for ii, next_size in enumerate(unshared_hidden_sizes):
                for q_idx in range(self._n_qs):
                    ufc = nn.Linear(in_size, next_size)
                    ptu.layer_init_xavier_normal(layer=ufc,
                                                 activation=hidden_activation,
                                                 b=hidden_b_init_val)
                    self.__setattr__("ufc{}_{}".format(q_idx, ii), ufc)
                    self._ufcs[q_idx].append(ufc)

                    if self._unshared_layer_norm:
                        ln = LayerNorm(next_size)
                        tmp_txt = "ufc{}_{}_norm".format(q_idx, ii)
                        self.__setattr__(tmp_txt, ln)
                        self._ufc_norms[q_idx].append(ln)
                in_size = next_size

        for q_idx in range(self._n_qs):
            last_ufc = nn.Linear(in_size, 1)
            ptu.layer_init_xavier_normal(layer=last_ufc,
                                         activation=output_activation,
                                         b=output_b_init_val)
            self.__setattr__("ufc_last{}".format(q_idx), last_ufc)
            self._ufcs_lasts.append(last_ufc)

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
        # Shared Layers
        for i, fc in enumerate(self._sfcs):
            h = self._hidden_activation(fc(h))

        hs = [h.clone() for _ in val_idxs]
        # Unshared Layers
        if len(self._ufcs) > 0:
            for ii, idx in enumerate(val_idxs):
                for i, fc in enumerate(self._ufcs[idx]):
                    hs[ii] = self._hidden_activation(fc(hs[ii]))

        values = [self._output_activation(self._ufcs_lasts[idx](hs[ii]))
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
        return self._n_qs
