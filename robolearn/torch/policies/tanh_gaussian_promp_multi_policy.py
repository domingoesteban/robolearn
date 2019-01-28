import math
import torch
from torch import nn as nn
from torch.distributions import Normal
from robolearn.torch.core import PyTorchModule
from robolearn.torch.core import np_ify
from torch.nn.modules.normalization import LayerNorm
import robolearn.torch.utils.pytorch_util as ptu
from robolearn.models.policies import ExplorationPolicy
from collections import OrderedDict
from itertools import chain

# LOG_SIG_MAX = 2
# LOG_SIG_MIN = -3.0
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

# SIG_MAX = 7.38905609893065
# SIG_MIN = 0.049787068367863944

# LOG_MIX_COEFF_MIN = -10
# LOG_MIX_COEFF_MAX = -1e-6  #-4.5e-5
# LOG_MIX_COEFF_MIN = -1
# LOG_MIX_COEFF_MAX = 1  #-4.5e-5

# EPS = 1e-12
EPS = 1e-8


class TanhGaussianPrompMultiPolicy(PyTorchModule, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPrompMultiPolicy(...)
    action, policy_dict = policy(obs)
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
            shared_hidden_sizes=None,
            unshared_hidden_sizes=None,
            unshared_mix_hidden_sizes=None,
            stds=None,
            hidden_activation='relu',
            hidden_w_init='xavier_normal',
            hidden_b_init_val=0,
            output_w_init='xavier_normal',
            output_b_init_val=0,
            pol_output_activation='linear',
            mix_output_activation='linear',
            input_norm=False,
            shared_layer_norm=False,
            policies_layer_norm=False,
            mixture_layer_norm=False,
            softmax_weights=False,
            mixing_temperature=1.,
            **kwargs
    ):
        self.save_init_params(locals())
        PyTorchModule.__init__(self)
        ExplorationPolicy.__init__(self, action_dim)

        self._input_size = obs_dim
        self._output_sizes = action_dim
        self._n_subpolicies = n_policies
        # Activation Fcns
        self._hidden_activation = ptu.get_activation(hidden_activation)
        self._pol_output_activation = ptu.get_activation(pol_output_activation)
        self._mix_output_activation = ptu.get_activation(mix_output_activation)
        # Normalization Layer Flags
        self._shared_layer_norm = shared_layer_norm
        self._policies_layer_norm = policies_layer_norm
        self._mixture_layer_norm = mixture_layer_norm
        # Layers Lists
        self._sfcs = []  # Shared Layers
        self._sfc_norms = []  # Norm. Shared Layers
        self._pfcs = [list() for _ in range(self._n_subpolicies)]  # Policies Layers
        self._pfc_norms = [list() for _ in range(self._n_subpolicies)]  # N. Pol. L.
        self._pfc_lasts = []  # Last Policies Layers
        self._mfcs = []  # Mixing Layers
        self._norm_mfcs = []  # Norm. Mixing Layers
        # self.mfc_last = None  # Below is instantiated

        self._mixing_temperature = mixing_temperature  # Hyperparameter for exp.
        self._softmax_weights = softmax_weights

        # Initial size = Obs size
        in_size = self._input_size

        # Ordered Dictionaries for specific modules/parameters
        self._shared_modules = OrderedDict()
        self._shared_parameters = OrderedDict()
        self._policies_modules = [OrderedDict() for _ in range(n_policies)]
        self._policies_parameters = [OrderedDict() for _ in range(n_policies)]
        self._mixing_modules = OrderedDict()
        self._mixing_parameters = OrderedDict()

        # ############# #
        # Shared Layers #
        # ############# #
        if input_norm:
            ln = nn.BatchNorm1d(in_size)
            self.sfc_input = ln
            self.add_shared_module("sfc_input", ln)
        else:
            self.sfc_input = None

        if shared_hidden_sizes is not None:
            for ii, next_size in enumerate(shared_hidden_sizes):
                sfc = nn.Linear(in_size, next_size)
                ptu.layer_init(
                    layer=sfc,
                    option=hidden_w_init,
                    activation=hidden_activation,
                    b=hidden_b_init_val,
                )
                self.__setattr__("sfc{}".format(ii), sfc)
                self._sfcs.append(sfc)
                self.add_shared_module("sfc{}".format(ii), sfc)

                if self._shared_layer_norm:
                    ln = LayerNorm(next_size)
                    # ln = nn.BatchNorm1d(next_size)
                    self.__setattr__("sfc{}_norm".format(ii), ln)
                    self._sfc_norms.append(ln)
                    self.add_shared_module("sfc{}_norm".format(ii), ln)
                in_size = next_size

        # Get the output_size of the shared layers (assume same for all)
        multipol_in_size = in_size
        mixture_in_size = in_size

        # ############### #
        # Unshared Layers #
        # ############### #
        # Unshared Multi-Policy Hidden Layers
        if unshared_hidden_sizes is not None:
            for ii, next_size in enumerate(unshared_hidden_sizes):
                for pol_idx in range(self._n_subpolicies):
                    pfc = nn.Linear(multipol_in_size, next_size)
                    ptu.layer_init(
                        layer=pfc,
                        option=hidden_w_init,
                        activation=hidden_activation,
                        b=hidden_b_init_val,
                    )
                    self.__setattr__("pfc{}_{}".format(pol_idx, ii), pfc)
                    self._pfcs[pol_idx].append(pfc)
                    self.add_policies_module("pfc{}_{}".format(pol_idx, ii),
                                             pfc, idx=pol_idx)

                    if self._policies_layer_norm:
                        ln = LayerNorm(next_size)
                        # ln = nn.BatchNorm1d(next_size)
                        self.__setattr__("pfc{}_{}_norm".format(pol_idx, ii),
                                         ln)
                        self._pfc_norms[pol_idx].append(ln)
                        self.add_policies_module("pfc{}_{}_norm".format(pol_idx,
                                                                        ii),
                                                 ln, idx=pol_idx)
                multipol_in_size = next_size

        # Multi-Policy Last Layers
        for pol_idx in range(self._n_subpolicies):
            last_pfc = nn.Linear(multipol_in_size, action_dim)
            ptu.layer_init(
                layer=last_pfc,
                option=output_w_init,
                activation=pol_output_activation,
                b=output_b_init_val,
            )
            self.__setattr__("pfc{}_last".format(pol_idx), last_pfc)
            self._pfc_lasts.append(last_pfc)
            self.add_policies_module("pfc{}_last".format(pol_idx), last_pfc,
                                     idx=pol_idx)

        # Multi-Policy Log-Stds Last Layers
        self.stds = stds
        self.log_std = list()
        if stds is None:
            self._pfc_log_std_lasts = list()
            for pol_idx in range(self._n_subpolicies):
                last_pfc_log_std = nn.Linear(multipol_in_size, action_dim)
                ptu.layer_init(
                    layer=last_pfc_log_std,
                    option=output_w_init,
                    activation=pol_output_activation,
                    b=output_b_init_val,
                )
                self.__setattr__("pfc{}_log_std_last".format(pol_idx),
                                 last_pfc_log_std)
                self._pfc_log_std_lasts.append(last_pfc_log_std)
                self.add_policies_module("pfc{}_log_std_last".format(pol_idx),
                                         last_pfc_log_std, idx=pol_idx)

        else:
            for std in stds:
                self.log_std.append(torch.log(stds))
                assert LOG_SIG_MIN <= self.log_std[-1] <= LOG_SIG_MAX

        # ############# #
        # Mixing Layers #
        # ############# #
        # Unshared Mixing-Weights Hidden Layers
        if unshared_mix_hidden_sizes is not None:
            for ii, next_size in enumerate(unshared_mix_hidden_sizes):
                mfc = nn.Linear(mixture_in_size, next_size)
                ptu.layer_init(
                    layer=mfc,
                    option=hidden_w_init,
                    activation=hidden_activation,
                    b=hidden_b_init_val,
                )
                self.__setattr__("mfc{}".format(ii), mfc)
                self._mfcs.append(mfc)
                # Add it to specific dictionaries
                self.add_mixing_module("mfc{}".format(ii), mfc)

                if self._mixture_layer_norm:
                    ln = LayerNorm(next_size)
                    # ln = nn.BatchNorm1d(next_size)
                    self.__setattr__("mfc{}_norm".format(ii), ln)
                    self._norm_mfcs.append(ln)
                    self.add_mixing_module("mfc{}_norm".format(ii), ln)
                mixture_in_size = next_size

        # Unshared Mixing-Weights Last Layers
        mfc_last = nn.Linear(mixture_in_size, self._n_subpolicies * action_dim)
        ptu.layer_init(
            layer=mfc_last,
            option=output_w_init,
            activation=mix_output_activation,
            b=output_b_init_val,
        )
        self.__setattr__("mfc_last", mfc_last)
        self.mfc_last = mfc_last
        # Add it to specific dictionaries
        self.add_mixing_module("mfc_last", mfc_last)

        self.mfc_sigmoid = nn.Sigmoid()

        self._normal_dist = Normal(loc=ptu.zeros(action_dim),
                                   scale=ptu.ones(action_dim))

        self._pols_idxs = ptu.arange(self._n_subpolicies)

    def get_action(self, obs_np, **kwargs):
        """
        """
        actions, info_dict = self.get_actions(obs_np[None], **kwargs)

        for key, val in info_dict.items():
            info_dict[key] = val[0, :]

        # Get [0, :] vals (Because it has dimension 1xdA)
        return actions[0, :], info_dict

    def get_actions(self, obs_np, **kwargs):
        """
        """
        actions, torch_info_dict = self.eval_np(obs_np, **kwargs)

        info_dict = dict()
        for key, vals in torch_info_dict.items():
            if key in ['mixing_coeff']:
                info_dict[key] = np_ify(torch_info_dict[key])

        return actions, info_dict

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=False,
            pol_idx=None,
            optimize_policies=True,
    ):
        """

        Args:
            obs (Tensor): Observation(s)
            deterministic (bool): True for using mean. False, sample from dist.
            return_log_prob (bool):
            pol_idx (int):
            optimize_policies (bool):

        Returns:
            action (Tensor):
            pol_info (dict):

        """
        h = obs
        nbatch = obs.shape[0]

        # ############# #
        # Shared Layers #
        # ############# #
        if self.sfc_input is not None:
            # h = self.sfc_input(h)
            if nbatch > 1:
                h = self.sfc_input(h)
            else:
                h = torch.batch_norm(
                    h,
                    self.sfc_input.weight,
                    self.sfc_input.bias,
                    self.sfc_input.running_mean,
                    self.sfc_input.running_var,
                    True,  # TODO: True or False??
                    self.sfc_input.momentum,
                    self.sfc_input.eps,
                    torch.backends.cudnn.enabled
                )

        for ss, fc in enumerate(self._sfcs):
            h = fc(h)

            if self._shared_layer_norm:
                h = self._sfc_norms[ss](h)

            h = self._hidden_activation(h)

        # ############## #
        # Multi Policies #
        # ############## #
        hs = [h.clone() for _ in range(self._n_subpolicies)]

        # Hidden Layers
        if len(self._pfcs) > 0:
            for pp in range(self._n_subpolicies):
                for ii, fc in enumerate(self._pfcs[pp]):
                    hs[pp] = fc(hs[pp])

                    if self._policies_layer_norm:
                        hs[pp] = self._pfc_norms[pp][ii](hs[pp])

                    hs[pp] = self._hidden_activation(hs[pp])

        # Last Mean Layers
        means = torch.cat(
            [(
                 self._pol_output_activation(self._pfc_lasts[pp](hs[pp]))
             ).unsqueeze(dim=1)
             for pp in range(self._n_subpolicies)
             ],
            dim=1
        )  # Batch x Npols x dA

        # Last Log-Std Layers
        if self.stds is None:
            log_stds = torch.cat(
                [(
                  self._pol_output_activation(
                      self._pfc_log_std_lasts[pp](hs[pp])
                  )
                 ).unsqueeze(dim=1)
                 for pp in range(self._n_subpolicies)
                 ],
                dim=1
            )  # Batch x Npols x dA

            # # log_std option 1:
            # log_stds = torch.clamp(log_stds, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            # log_std option 2:
            log_stds = torch.tanh(log_stds)
            log_stds = \
                LOG_SIG_MIN + 0.5 * (LOG_SIG_MAX - LOG_SIG_MIN)*(log_stds + 1)

            stds = torch.exp(log_stds)
            variances = stds**2

        else:
            log_stds = self.log_std
            stds = self.stds
            variances = stds**2

        # ############## #
        # Mixing Weigths #
        # ############## #
        mh = h.clone()

        if len(self._mfcs) > 0:
            for mm, mfc in enumerate(self._mfcs):
                mh = mfc(mh)

                if self._mixture_layer_norm:
                    mh = self._norm_mfcs[mm](mh)

                mh = self._hidden_activation(mh)

        # NO nonlinear transformation
        mixture_coeff = \
            self.mfc_last(mh).reshape(-1, self._n_subpolicies, self.action_dim)

        mixture_coeff = self.mfc_sigmoid(mixture_coeff)

        # if torch.isnan(mixture_coeff).any():
        #     raise ValueError('Some mixture coeff(s) is(are) NAN: %s' %
        #                      mixture_coeff)
        #
        # if torch.isnan(means).any():
        #     raise ValueError('Some means are NAN: %s' %
        #                      means)
        #
        # if torch.isnan(stds).any():
        #     raise ValueError('Some stds are NAN: %s' %
        #                      stds)

        if pol_idx is None:
            # Calculate weighted means and stds (and log_stds)
            if optimize_policies:
                sig_invs = mixture_coeff/variances
            else:
                sig_invs = mixture_coeff/variances.detach()

            variance = 1./torch.sum(sig_invs, dim=1, keepdim=False)

            if optimize_policies:
                mean = variance*torch.sum(
                    means*sig_invs,
                    dim=1,
                    keepdim=False
                )
            else:
                mean = variance*torch.sum(
                    means.detach()*sig_invs,
                    dim=1,
                    keepdim=False
                )

            # log_std option 1:
            std = torch.sqrt(variance)
            std = torch.clamp(std,
                              min=math.exp(LOG_SIG_MIN),
                              max=math.exp(LOG_SIG_MAX))
            log_std = torch.log(std)
            # # log_std option 2:
            # variance = torch.tanh(variance)
            # variance = (
            #     math.exp(LOG_SIG_MIN)**2 +
            #     0.5*(math.exp(LOG_SIG_MAX)**2 - math.exp(LOG_SIG_MIN)**2) *
            #     (variance + 1)
            # )
            # std = torch.sqrt(variance)
            # log_std = torch.log(std)

            # TODO: Remove the following?
            # log_std = torch.logsumexp(
            #     log_stds + log_mixture_coeff.reshape(-1,
            #                                          self.action_dim,
            #                                          self._n_subpolicies),
            #     dim=-1,
            #     keepdim=False
            # ) - torch.logsumexp(log_mixture_coeff, dim=-1, keepdim=True)

            # log_std = torch.log(std)

        else:
            index = self._pols_idxs[pol_idx]
            mean = \
                torch.index_select(means, dim=1, index=index).squeeze(1)
            std = \
                torch.index_select(stds, dim=1, index=index).squeeze(1)
            log_std = \
                torch.index_select(log_stds, dim=1, index=index).squeeze(1)
            variance = \
                torch.index_select(variances, dim=1, index=index).squeeze(1)

        pre_tanh_value = None
        log_prob = None
        pre_tanh_values = None
        log_probs = None

        if deterministic:
            action = torch.tanh(mean)
            actions = torch.tanh(means)
        else:
            # # Using this distribution instead of TanhMultivariateNormal
            # # because it has Diagonal Covariance.
            # # Then, a collection of n independent Gaussian r.v.
            # tanh_normal = TanhNormal(mean, std)
            #
            # # # It is the Lower-triangular factor of covariance because it is
            # # # Diagonal Covariance
            # # scale_trils = torch.stack([torch.diag(m) for m in std])
            # # tanh_normal = TanhMultivariateNormal(mean, scale_tril=scale_trils)
            #
            # if return_log_prob:
            #     log_prob = tanh_normal.log_prob(
            #         action,
            #         pre_tanh_value=pre_tanh_value
            #     )
            #     log_prob = log_prob.sum(dim=-1, keepdim=True)

            noise = self._normal_dist.sample((nbatch,))

            pre_tanh_value = std*noise + mean
            pre_tanh_values = stds*noise.unsqueeze(1) + means

            action = torch.tanh(pre_tanh_value)
            actions = torch.tanh(pre_tanh_values)

            if return_log_prob:
                # Log probability: Main Policy
                log_prob = -((pre_tanh_value - mean) ** 2) / (2*variance) \
                           - log_std - math.log(math.sqrt(2*math.pi))
                log_prob -= torch.log(
                    # torch.clamp(1. - action**2, 0, 1)
                    clip_but_pass_gradient(1. - action**2, 0, 1)
                    + 1.e-6
                )
                log_prob = log_prob.sum(dim=-1, keepdim=True)

                # Log probability: Sub-Policies
                log_probs = -((pre_tanh_values - means) ** 2) / (2*variances)\
                            - log_stds - math.log(math.sqrt(2*math.pi))
                log_probs -= torch.log(
                    # torch.clamp(1. - actions**2, 0, 1)
                    clip_but_pass_gradient(1. - actions**2, 0, 1)
                    + 1.e-6
                )
                log_probs = log_probs.sum(dim=-1, keepdim=True)

        # if torch.isnan(action).any():
        #     raise ValueError('ACTION NAN')
        #
        # if torch.isnan(actions).any():
        #     raise ValueError('ACTION NAN')

        info_dict = dict(
            mean=mean,
            std=std,
            log_std=log_std,
            log_prob=log_prob,
            pre_tanh_value=pre_tanh_value,
            # log_mixture_coeff=log_mixture_coeff,
            mixing_coeff=mixture_coeff,
            pol_actions=actions,
            pol_means=means,
            pol_stds=stds,
            pol_log_stds=log_stds,
            pol_log_probs=log_probs,
            pol_pre_tanh_values=pre_tanh_values,
        )

        return action, info_dict

    def log_action(self, actions, obs, pol_idx=None):
        raise NotImplementedError

    @property
    def n_heads(self):
        return self._n_subpolicies

    @property
    def n_subpolicies(self):
        return self._n_subpolicies

    # ################# #
    # Shared parameters #
    # ################# #

    def shared_parameters(self):
        """Returns an iterator over the shared parameters.
        """
        for name, param in self.named_shared_parameters():
            yield param

    def named_shared_parameters(self, **kwargs):
        """Returns an iterator over shared module parameters, yielding both the
        name of the parameter as well as the parameter itself
        """
        return ptu.named_parameters(self._shared_modules,
                                    self._shared_parameters,
                                    **kwargs)

    def add_shared_module(self, name, module):
        ptu.add_module(self._shared_modules, name, module)

    # ####################### #
    # Sub-Policies parameters #
    # ####################### #

    def policies_parameters(self, idx=None):
        """Returns an iterator over the policies parameters.
        """
        if idx is None:
            idx_list = list(range(self._n_subpolicies))
        elif isinstance(idx, list) or isinstance(idx, tuple):
            idx_list = idx
        else:
            idx_list = [idx]

        for name, param in self.named_policies_parameters(idx_list):
            yield param

    def named_policies_parameters(self, idx=None, **kwargs):
        """Returns an iterator over policies module parameters, yielding both the
        name of the parameter as well as the parameter itself
        """
        if idx is None:
            idx_list = list(range(self._n_subpolicies))
        elif isinstance(idx, list) or isinstance(idx, tuple):
            idx_list = idx
        else:
            idx_list = [idx]

        return chain(*[ptu.named_parameters(self._policies_modules[idx],
                                            self._policies_parameters[idx],
                                            **kwargs)
                       for idx in idx_list])

    def add_policies_module(self, name, module, idx=None):
        if idx is None:
            idx_list = list(range(self._n_subpolicies))
        elif isinstance(idx, list) or isinstance(idx, tuple):
            idx_list = idx
        else:
            idx_list = [idx]

        for idx in idx_list:
            ptu.add_module(self._policies_modules[idx], name, module)

    # ################# #
    # Mixing parameters #
    # ################# #

    def mixing_parameters(self):
        """Returns an iterator over the mixing parameters.
        """
        for name, param in self.named_mixing_parameters():
            yield param

    def named_mixing_parameters(self, **kwargs):
        """Returns an iterator over mixing module parameters, yielding both the
        name of the parameter as well as the parameter itself
        """
        return ptu.named_parameters(self._mixing_modules,
                                    self._mixing_parameters,
                                    **kwargs)

    def add_mixing_module(self, name, module):
        ptu.add_module(self._mixing_modules, name, module)


def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = (x > u).to(ptu.device, dtype=torch.float32)
    clip_low = (x < l).to(ptu.device, dtype=torch.float32)
    return x + ((u - x)*clip_up + (l - x)*clip_low).detach()
