import numpy as np
from robolearn.torch.core import PyTorchModule
from robolearn.policies import Policy
import torch
import torch.nn as nn
import torch.nn.functional as F
from robolearn.core.serializable import Serializable
import robolearn.torch.pytorch_util as ptu


class LinearGaussianPolicy(PyTorchModule, Serializable, Policy):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 T,
                 K=None,
                 k=None,
                 pol_covar_diag=None,
                 ):

        Policy.__init__(self,
                        action_dim=action_dim)

        # self._serializable_initialized = False
        # Serializable.quick_init(self, locals())
        self.save_init_params(locals())
        PyTorchModule.__init__(self)

        self._obs_dim = obs_dim

        if K is None:
            K = torch.rand((self.action_dim, self._obs_dim)).repeat(T, 1, 1)
        else:
            K = ptu.from_numpy(K)

        if k is None:
            k = torch.zeros(T, self.action_dim)
        else:
            k = ptu.from_numpy(k)

        if pol_covar_diag is None:
            pol_covar_diag = torch.ones(self.action_dim).repeat(T, 1, 1)
        else:
            pol_covar_diag = ptu.from_numpy(pol_covar_diag)

        self.K = nn.Parameter(data=K)
        # self.K = K

        self.k = nn.Parameter(data=k)
        # self.k = k

        self._covar_diag = nn.Parameter(pol_covar_diag)
        # self._covar_diag = pol_covar_diag

        self._T = T

    # def K(self, t):
    #     return torch.diag(self.K_params[t, :])

    @property
    def H(self):
        return self._T

    @property
    def T(self):
        return self._T

    def get_action(self, obs_np, **kwargs):
        values, info_dict = \
            self.get_actions(obs_np[None], **kwargs)

        for key, val in info_dict.items():
            if isinstance(val, np.ndarray):
                info_dict[key] = val[0, :]

        return values[0, :], info_dict

    def get_actions(self, obs_np, **kwargs):
        return self.eval_np(obs_np, **kwargs)

    def forward(
            self,
            obs,
            t=None,
            noise=None,
            return_preactivations=False,
    ):
        if t is None:
            raise NotImplementedError
        else:
            action = F.linear(obs, self.K[t, :, :], self.k[t, :])
            if noise is None:
                noise = 0
            else:
                covar = torch.diag(self._covar_diag[t, :].squeeze())
                noise = F.linear(noise, covar)
            action += noise
        info_dict = dict()

        return action, info_dict

    def set_K(self, K):
        self.K.data = K

    def set_k(self, k):
        self.k.data = k
