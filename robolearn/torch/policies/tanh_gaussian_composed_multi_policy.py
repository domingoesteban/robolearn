import math
import torch
from torch import nn as nn
from torch.distributions import Normal
from robolearn.torch.core import PyTorchModule
from robolearn.torch.utils.pytorch_util import np_ify
from torch.nn.modules.normalization import LayerNorm
import robolearn.torch.utils.pytorch_util as ptu
from robolearn.models.policies import ExplorationPolicy
from collections import OrderedDict
from itertools import chain

LOG_SIG_MAX = 2
# LOG_SIG_MIN = -20
LOG_SIG_MIN = -3.0

SIG_MAX = 7.38905609893065
SIG_MIN = 0.049787068367863944

LOG_MIX_COEFF_MIN = -10
LOG_MIX_COEFF_MAX = -1e-6  #-4.5e-5
LOG_MIX_COEFF_MIN = -1
LOG_MIX_COEFF_MAX = 1  #-4.5e-5

EPS = 1e-12


class TanhGaussianComposedMultiPolicy(PyTorchModule, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianComposedMultiPolicy(...)
    action, policy_dict = policy(obs)
    action, policy_dict = policy(obs, deterministic=True)
    action, policy_dict = policy(obs, return_log_prob=True)
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
            latent_dim,
            shared_hidden_sizes=None,
            unshared_hidden_sizes=None,
            unshared_mix_hidden_sizes=None,
            unshared_policy_hidden_sizes=None,
            stds=None,
            hidden_activation='relu',
            hidden_w_init='xavier_normal',
            hidden_b_init_val=1e-2,
            output_w_init='xavier_normal',
            output_b_init_val=1e-2,
            pol_output_activation='linear',
            mix_output_activation='linear',
            final_pol_output_activation='linear',
            input_norm=False,
            shared_layer_norm=False,
            policies_layer_norm=False,
            mixture_layer_norm=False,
            final_policy_layer_norm=False,
            epsilon=1e-6,
            softmax_weights=False,
            **kwargs
    ):
        self.save_init_params(locals())
        TanhGaussianComposedMultiPolicy.__init__(self)
        ExplorationPolicy.__init__(self, action_dim)

        self._input_size = obs_dim
        self._output_sizes = action_dim
        self._n_subpolicies = n_policies
        self._latent_size = latent_dim
        # Activation Fcns
        self._hidden_activation = ptu.get_activation(hidden_activation)
        self._pol_output_activation = ptu.get_activation(pol_output_activation)
        self._mix_output_activation = ptu.get_activation(mix_output_activation)
        self._final_pol_output_activation = ptu.get_activation(final_pol_output_activation)
        # Normalization Layer Flags
        self._shared_layer_norm = shared_layer_norm
        self._policies_layer_norm = policies_layer_norm
        self._mixture_layer_norm = mixture_layer_norm
        self._final_policy_layer_norm = final_policy_layer_norm
        # Layers Lists
        self._sfcs = []  # Shared Layers
        self._sfc_norms = []  # Norm. Shared Layers
        self._pfcs = [list() for _ in range(self._n_subpolicies)]  # Policies Layers
        self._pfc_norms = [list() for _ in range(self._n_subpolicies)]  # N. Pol. L.
        self._pfc_lasts = []  # Last Policies Layers
        self._mfcs = []  # Mixing Layers
        self._norm_mfcs = []  # Norm. Mixing Layers
        # self.mfc_last = None  # Below is instantiated
        self._fpfcs = []  # Final Policy Layers
        self._norm_fpfcs = []  # Norm. Mixing Layers

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
        self._final_policy_modules = OrderedDict()
        self._final_policy_parameters = OrderedDict()

        # ############# #
        # Shared Layers #
        # ############# #
        if input_norm:
            ln = nn.BatchNorm1d(in_size)
            self.sfc_input = ln
            self.add_shared_module("sfc_input", ln)
            self.__setattr__("sfc_input", ln)
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
                        b=hidden_b_init_val
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
            last_pfc = nn.Linear(multipol_in_size, latent_dim)
            ptu.layer_init(
                layer=last_pfc,
                option=output_w_init,
                activation=pol_output_activation,
                b=output_b_init_val
            )
            self.__setattr__("pfc{}_last".format(pol_idx), last_pfc)
            self._pfc_lasts.append(last_pfc)
            self.add_policies_module("pfc{}_last".format(pol_idx), last_pfc,
                                     idx=pol_idx)

        # ############# #
        # Mixing Layers #
        # ############# #
        mixture_in_size = in_size + latent_dim*self._n_subpolicies
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
        mfc_last = nn.Linear(mixture_in_size, latent_dim)
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

        if softmax_weights:
            raise ValueError("Check if it is correct a softmax")
            # self.mfc_softmax = nn.Softmax(dim=1)
        else:
            self.mfc_softmax = None

        # ################### #
        # Final Policy Layers #
        # ################### #
        final_pol_in_size = latent_dim
        if unshared_policy_hidden_sizes is not None:
            for ii, next_size in enumerate(unshared_policy_hidden_sizes):
                fpfc = nn.Linear(final_pol_in_size, next_size)
                ptu.layer_init(
                    layer=fpfc,
                    option=hidden_w_init,
                    activation=hidden_activation,
                    b=hidden_b_init_val
                )
                self.__setattr__("fpfc{}".format(ii), fpfc)
                self._fpfcs.append(fpfc)
                # Add it to specific dictionaries
                self.add_final_policy_module("fpfc{}".format(ii), fpfc)

                if self._mixture_layer_norm:
                    ln = LayerNorm(next_size)
                    # ln = nn.BatchNorm1d(next_size)
                    self.__setattr__("fpfc{}_norm".format(ii), ln)
                    self._norm_fpfcs.append(ln)
                    self.add_final_policy_module("fpfc{}_norm".format(ii), ln)
                final_pol_in_size = next_size

        # Unshared Final Policy Last Layer
        fpfc_last = nn.Linear(final_pol_in_size, action_dim)
        ptu.layer_init(
            layer=fpfc_last,
            option=output_w_init,
            activation=final_pol_output_activation,
            b=output_b_init_val
        )
        self.__setattr__("fpfc_last", fpfc_last)
        self.fpfc_last = fpfc_last
        # Add it to specific dictionaries
        self.add_final_policy_module("fpfc_last", fpfc_last)

        # ########## #
        # Std Layers #
        # ########## #
        # Multi-Policy Log-Stds Last Layers
        fpfc_last_log_std = nn.Linear(final_pol_in_size, action_dim)
        ptu.layer_init(
            layer=fpfc_last_log_std,
            option=output_w_init,
            activation=final_pol_output_activation,
            b=output_b_init_val
        )
        self.__setattr__("fpfc_last_log_std", fpfc_last_log_std)
        self.fpfc_last_log_std = fpfc_last_log_std
        # Add it to specific dictionaries
        self.add_final_policy_module("fpfc_last_log_std", fpfc_last_log_std)

        self._normal_dist = Normal(loc=ptu.zeros(action_dim),
                                   scale=ptu.ones(action_dim))
        self._epsilon = epsilon

        self._pols_idxs = ptu.arange(self._n_subpolicies)
        self._compo_pol_idx = torch.tensor([self._n_subpolicies],
                                           dtype=torch.int64,
                                           device=ptu.device)

    def get_action(self, obs_np, **kwargs):
        actions, info_dict = self.get_actions(obs_np[None], **kwargs)

        for key, val in info_dict.items():
            info_dict[key] = val[0, :]

        # Get [0, :] vals (Because it has dimension 1xdA)
        return actions[0, :], info_dict

    def get_actions(self, obs_np, **kwargs):
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
            deterministic (bool):
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

            if self._mixture_layer_norm:
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

        subpol_means = \
            [self._pol_output_activation(self._pfc_lasts[pp](hs[pp]))
             for pp in range(self._n_subpolicies)]
        subpols = torch.cat(subpol_means, dim=-1)

        if torch.isnan(subpols).any():
            raise ValueError('Some subpols are NAN:',
                             subpols)

        # ############## #
        # Mixing Weigths #
        # ############## #
        mh = torch.cat([h.clone(), subpols], dim=-1)  # N x dZ
        if not optimize_policies:
            mh = mh.detach()

        if len(self._mfcs) > 0:
            for mm, mfc in enumerate(self._mfcs):
                mh = mfc(mh)

                if self._mixture_layer_norm:
                    mh = self._norm_mfcs[mm](mh)

                mh = self._hidden_activation(mh)

        # NO nonlinear transformation
        mpol_mean = self.mfc_last(mh)

        if self.mfc_softmax is not None:
            raise NotImplementedError
            # mixture_coeff = self.mfc_softmax(mixture_coeff)


        # Final Policy
        final_pol_inputs = [ii.unsqueeze(-2)
                            for ii in (subpol_means + [mpol_mean])]
        fph = torch.cat(
            final_pol_inputs,
            dim=-2,
        )

        for ff, fpfc in enumerate(self._fpfcs):
            fph = fpfc(fph)

            if self._final_policy_layer_norm:
                fph = self._norm_mfcs[ff](fph)

            fph = self._hidden_activation(fph)

        means = self._final_pol_output_activation(
            self.fpfc_last(fph)
        )

        log_stds = self._final_pol_output_activation(
            self.fpfc_last_log_std(fph)
        )
        log_stds = torch.clamp(log_stds, LOG_SIG_MIN, LOG_SIG_MAX)
        stds = torch.exp(log_stds)
        variances = torch.pow(stds, 2)

        if pol_idx is None:
            index = self._compo_pol_idx
        else:
            index = self._pols_idxs[pol_idx]

        mean = \
            torch.index_select(means, dim=-2, index=index).squeeze(-2)
        std = \
            torch.index_select(stds, dim=-2, index=index).squeeze(-2)
        log_std = \
            torch.index_select(log_stds, dim=-2, index=index).squeeze(-2)
        variance = \
            torch.index_select(variances, dim=-2, index=index).squeeze(-2)

        means = \
            torch.index_select(means, dim=-2, index=self._pols_idxs).squeeze(-2)
        stds = \
            torch.index_select(stds, dim=-2, index=self._pols_idxs).squeeze(-2)
        log_stds = \
            torch.index_select(log_stds, dim=-2, index=self._pols_idxs).squeeze(-2)
        variances = \
            torch.index_select(variances, dim=-2, index=self._pols_idxs).squeeze(-2)

        pre_tanh_value = None
        log_prob = None
        entropy = None
        mean_action_log_prob = None
        log_probs = None
        pre_tanh_values = None

        mixture_coeff = ptu.ones((nbatch, self.n_heads, self.action_dim))

        if deterministic:
            action = torch.tanh(mean)
            actions = torch.tanh(means)
        else:
            noise = self._normal_dist.sample((nbatch,))
            pre_tanh_value = std*noise + mean
            pre_tanh_values = stds*noise.unsqueeze(1) + means
            action = torch.tanh(pre_tanh_value)
            actions = torch.tanh(pre_tanh_values)

            if return_log_prob:
                # Log probability: Main Policy
                log_prob = -((pre_tanh_value - mean) ** 2) / (2 * variance) \
                           - torch.log(std) - math.log(math.sqrt(2 * math.pi))
                log_prob -= torch.log(1. - action**2 + self._epsilon)
                log_prob = log_prob.sum(dim=-1, keepdim=True)

                # Log probability: Sub-Policies
                log_probs = -((pre_tanh_values - means) ** 2) / (2 * variances) \
                            - torch.log(stds) - math.log(math.sqrt(2 * math.pi))
                log_probs -= torch.log(1. - actions**2 + self._epsilon)
                log_probs = log_probs.sum(dim=-1, keepdim=True)

        if torch.isnan(action).any():
            raise ValueError('ACTION NAN')

        if torch.isnan(actions).any():
            raise ValueError('ACTION NAN')

        info_dict = dict(
            mean=mean,
            log_std=log_std,
            log_prob=log_prob,
            entropy=entropy,
            std=std,
            mean_action_log_prob=mean_action_log_prob,
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

    # ####################### #
    # Final policy parameters #
    # ####################### #

    def final_policy_parameters(self):
        """Returns an iterator over the final policy parameters.
        """
        for name, param in self.named_final_policy_parameters():
            yield param

    def named_final_policy_parameters(self, **kwargs):
        """Returns an iterator over final policy module parameters, yielding
        both the name of the parameter as well as the parameter itself
        """
        return ptu.named_parameters(self._final_policy_modules,
                                    self._final_policy_parameters,
                                    **kwargs)

    def add_final_policy_module(self, name, module):
        ptu.add_module(self._final_policy_modules, name, module)
