from robolearn.torch.core import PyTorchModule
from robolearn.core.serializable import Serializable
import robolearn.torch.pytorch_util as ptu
import torch.nn as nn
import torch.nn.functional as F
from robolearn.torch.nn import identity
from robolearn.torch.nn import LayerNorm
from robolearn.models import VFunction


class NNMultiVFunction(PyTorchModule, VFunction):
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

        VFunction.__init__(self, obs_dim=obs_dim)

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
        """

        Args:
            obs (Tensor):
            val_idxs (iterable):

        Returns:
            values (list)
            info (dict): empty dictionary

        """
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

        return values, {}

    def get_value(self, obs_np):
        # TODO: CHECK IF INDEX 0
        values = self.get_values(obs_np[None])
        return values[0, :], {}

    def get_values(self, obs_np):
        return self.eval_np(obs_np)

    @property
    def n_heads(self):
        return self._n_vs
