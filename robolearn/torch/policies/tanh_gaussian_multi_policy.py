import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from robolearn.torch.core import PyTorchModule
from robolearn.torch.core import np_ify
from robolearn.torch.nn import LayerNorm
import robolearn.torch.pytorch_util as ptu
from robolearn.policies.base import ExplorationPolicy
from robolearn.torch.nn import identity
from robolearn.torch.distributions import TanhNormal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhGaussianMultiPolicy(PyTorchModule, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianMultiPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            obs_dim,
            action_dim,
            n_policies,
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
        self.save_init_params(locals())
        super(TanhGaussianMultiPolicy, self).__init__()
        ExplorationPolicy.__init__(self, action_dim)

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = obs_dim
        self.output_sizes = action_dim
        self._n_policies = n_policies
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.shared_layer_norm = shared_layer_norm
        self.unshared_layer_norm = unshared_layer_norm
        self.fcs = []
        self.shared_layer_norms = []
        self.ufcs = [list() for _ in range(self._n_policies)]
        self.unshared_layer_norms = [list() for _ in range(self._n_policies)]
        self.last_fcs = []
        in_size = self.input_size

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
                for pol_idx in range(self._n_policies):
                    ufc = nn.Linear(in_size, next_size)
                    hidden_w_init(ufc.weight)
                    ptu.fill(ufc.bias, hidden_b_init_val)
                    self.__setattr__("ufc{}_{}".format(pol_idx, i), ufc)
                    self.ufcs[pol_idx].append(ufc)

                    if self.unshared_layer_norm:
                        ln = LayerNorm(next_size)
                        tmp_txt = "unshared_layer_norm{}_{}".format(pol_idx, i)
                        self.__setattr__(tmp_txt, ln)
                        self.unshared_layer_norms[pol_idx].append(ln)
                in_size = next_size

        for pol_idx in range(self._n_policies):
            last_fc = nn.Linear(in_size, self._action_dim)
            output_w_init(last_fc.weight)
            ptu.fill(last_fc.bias, output_b_init_val)
            self.__setattr__("last_fc{}".format(pol_idx), last_fc)
            self.last_fcs.append(last_fc)

        self.stds = stds
        self.log_std = list()
        if stds is None:
            self.last_fc_log_stds = list()
            for pol_idx in range(self._n_policies):
                last_hidden_size = obs_dim
                if unshared_hidden_sizes is None:
                    if len(shared_hidden_sizes) > 0:
                        last_hidden_size = shared_hidden_sizes[-1]
                else:
                    last_hidden_size = unshared_hidden_sizes[-1]
                last_fc_log_std = nn.Linear(last_hidden_size,
                                            action_dim)
                hidden_w_init(last_fc_log_std.weight)
                ptu.fill(last_fc_log_std.bias, hidden_b_init_val)
                self.__setattr__("last_fc_log_std{}".format(pol_idx),
                                 last_fc_log_std)
                self.last_fc_log_stds.append(last_fc_log_std)

        else:
            for std in stds:
                self.log_std.append(np.log(stds))
                assert LOG_SIG_MIN <= self.log_std[-1] <= LOG_SIG_MAX

    def get_action(self, obs_np, **kwargs):
<<<<<<< HEAD
        pol_idxs = kwargs['pol_idxs']

        actions, info_dict = self.get_actions(obs_np[None], **kwargs)

        if len(pol_idxs) > 1:
            actions = [action[0, :] for action in actions]
        else:
            actions = actions[0, :]

        for key, vals in info_dict.items():
            if len(pol_idxs) > 1:
                info_dict[key] = [val[0, :] if isinstance(val, np.ndarray)
                                  else None for val in vals]
            else:
                info_dict[key] = vals[0, :] if isinstance(vals, np.ndarray) \
                                  else None
=======
        actions, info_dict = self.get_actions(obs_np[None], **kwargs)

        actions = [action[0, :] for action in actions]

        for key, vals in info_dict.items():
            info_dict[key] = [val[0, :] if isinstance(val, np.ndarray)
                              else None for val in vals]
>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d

        return actions, info_dict

    def get_actions(self, obs_np, **kwargs):
<<<<<<< HEAD
        pol_idxs = kwargs['pol_idxs']

        actions, info_dict = self.eval_np(obs_np, **kwargs)

        if len(pol_idxs) > 1:
            actions = [np_ify(tensor) for tensor in actions]
        else:
            actions = np_ify(actions)

        for key, vals in info_dict.items():
            if len(pol_idxs) > 1:
                info_dict[key] = [np_ify(val) for val in vals]
            else:
                info_dict[key] = np_ify(vals)
=======
        actions, info_dict = self.eval_np(obs_np, **kwargs)

        actions = [np_ify(tensor) for tensor in actions]

        for key, vals in info_dict.items():
            info_dict[key] = [np_ify(val) for val in vals]
>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d

        return actions, info_dict

    def forward(
            self,
            obs,
            pol_idxs=None,
            deterministic=False,
            return_log_prob=False,
    ):
        """

        Args:
            obs (Tensor): Observation(s)
            pol_idxs (iterator):
            deterministic (bool):
            return_log_prob (bool):

        Returns:
            action (Tensor):
            pol_info (dict):

        """
        if pol_idxs is None:
            pol_idxs = list(range(self._n_policies))

        h = obs
        # Shared Layers
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))

        hs = [h for _ in pol_idxs]

        # Unshared Layers
        if len(self.ufcs) > 0:
            for ii, idx in enumerate(pol_idxs):
                for i, fc in enumerate(self.ufcs[idx]):
                    hs[ii] = self.hidden_activation(fc(hs[ii]))

        means = [self.last_fcs[idx](hs[ii])
                 for ii, idx in enumerate(pol_idxs)]

        if self.stds is None:
            log_stds = [None for _ in pol_idxs]
            stds = [None for _ in pol_idxs]
            for ii, idx in enumerate(pol_idxs):
                log_stds[ii] = self.last_fc_log_stds[idx](hs[ii])
                log_stds[ii] = torch.clamp(log_stds[ii],
                                           min=LOG_SIG_MIN, max=LOG_SIG_MAX)
                stds[ii] = torch.exp(log_stds[ii])
        else:
            stds = self.stds
            log_stds = self.log_std

        log_probs = [None for _ in pol_idxs]
        expected_log_probs = [None for _ in pol_idxs]
        mean_action_log_probs = [None for _ in pol_idxs]
        pre_tanh_values = [None for _ in pol_idxs]

        if deterministic:
            actions = [torch.tanh(mean) for mean in means]
        else:
            actions = [None for _ in means]
            for ii in range(len(pol_idxs)):
                mean = means[ii]
                std = stds[ii]
                tanh_normal = TanhNormal(mean, std)
                if return_log_prob:
                    actions[ii], pre_tanh_values[ii] = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                    log_probs[ii] = tanh_normal.log_prob(
                        actions[ii],
                        pre_tanh_value=pre_tanh_values[ii]
                    )
                    log_probs[ii] = log_probs[ii].sum(dim=-1, keepdim=True)
                else:
                    actions[ii], pre_tanh_values[ii] = \
                        tanh_normal.rsample(return_pretanh_value=True)

<<<<<<< HEAD
        if len(pol_idxs) == 1:
            actions = actions[0]
            means = means[0]
            log_stds = log_stds[0]
            log_probs = log_probs[0]
            expected_log_probs = expected_log_probs[0]
            stds = stds[0]
            mean_action_log_probs = mean_action_log_probs[0]
            pre_tanh_values = pre_tanh_values[0]

=======
>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d
        info_dict = dict(
            mean=means,
            log_std=log_stds,
            log_prob=log_probs,
            expected_log_prob=expected_log_probs,
            std=stds,
            mean_action_log_prob=mean_action_log_probs,
            pre_tanh_value=pre_tanh_values,
        )

        return actions, info_dict

    def log_action(self, actions, obs, pol_idxs=None):
        if pol_idxs is None:
            pol_idxs = list(range(self._n_policies))
        assert len(pol_idxs) == len(actions)

        h = obs
        # Shared Layers
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))

        hs = [h for _ in pol_idxs]

        # Unshared Layers
        if len(self.ufcs) > 0:
            for ii, idx in enumerate(pol_idxs):
                for i, fc in enumerate(self.ufcs[idx]):
                    hs[ii] = self.hidden_activation(fc(hs[ii]))

        means = [self.last_fcs[idx](hs[ii])
                 for ii, idx in enumerate(pol_idxs)]

        if self.stds is None:
            log_stds = [None for _ in pol_idxs]
            stds = [None for _ in pol_idxs]
            for ii, idx in enumerate(pol_idxs):
                log_stds[ii] = self.last_fc_log_stds[idx](hs[ii])
                log_stds[ii] = torch.clamp(log_stds[ii],
                                           min=LOG_SIG_MIN, max=LOG_SIG_MAX)
                stds[ii] = torch.exp(log_stds[ii])
        else:
            stds = self.stds

        log_probs = [None for _ in pol_idxs]

        for ii in range(len(pol_idxs)):
            mean = means[ii]
            std = stds[ii]
            tanh_normal = TanhNormal(mean, std)
            log_probs[ii] = torch.sum(tanh_normal.log_prob(actions),
                                      dim=-1, keepdim=True)

        return log_probs

        # z = (actions - mean)/stds
        # return -0.5 * torch.sum(torch.mul(z, z), dim=-1, keepdim=True)

    @property
    def n_heads(self):
        return self._n_policies
