import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Multinomial
from robolearn.torch.core import PyTorchModule
from robolearn.torch.core import np_ify
# from robolearn.torch.nn import LayerNorm
from torch.nn.modules.normalization import LayerNorm
import robolearn.torch.pytorch_util as ptu
from robolearn.policies.base import ExplorationPolicy
from robolearn.torch.nn import identity
from robolearn.torch.distributions import TanhNormal
from collections import OrderedDict
from itertools import chain

LOG_SIG_MAX = 2
# LOG_SIG_MIN = -20
# LOG_SIG_MAX = 0.0  # 2
LOG_SIG_MIN = -3.0  # 20

SIG_MAX = 7.38905609893065
SIG_MIN = 0.049787068367863944

LOG_MIX_COEFF_MIN = -10
LOG_MIX_COEFF_MAX = -1e-6  #-4.5e-5
LOG_MIX_COEFF_MIN = -1
LOG_MIX_COEFF_MAX = 1  #-4.5e-5

EPS = 1e-12


class TanhGaussianWeightedMultiPolicy(PyTorchModule, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianWeightedMultiPolicy(...)
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
            shared_hidden_sizes=None,
            unshared_hidden_sizes=None,
            unshared_mix_hidden_sizes=None,
            stds=None,
            hidden_activation='relu',
            hidden_w_init=ptu.xavier_init,
            hidden_b_init_val=1e-2,
            output_w_init=ptu.xavier_init,
            output_b_init_val=1e-2,
            pol_output_activation='linear',
            mix_output_activation='linear',
            shared_layer_norm=False,
            policies_layer_norm=False,
            mixture_layer_norm=False,
            mixing_temperature=1.,
            reparameterize=True,
            **kwargs
    ):
        self.save_init_params(locals())
        super(TanhGaussianWeightedMultiPolicy, self).__init__()
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
        if shared_hidden_sizes is not None:
            for ii, next_size in enumerate(shared_hidden_sizes):
                sfc = nn.Linear(in_size, next_size)
                ptu.layer_init_xavier_normal(layer=sfc,
                                             activation=hidden_activation,
                                             b=hidden_b_init_val)
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

        # Unshared Multi-Policy Hidden Layers
        if unshared_hidden_sizes is not None:
            for ii, next_size in enumerate(unshared_hidden_sizes):
                for pol_idx in range(self._n_subpolicies):
                    pfc = nn.Linear(multipol_in_size, next_size)
                    ptu.layer_init_xavier_normal(layer=pfc,
                                                 activation=hidden_activation,
                                                 b=hidden_b_init_val)
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
            ptu.layer_init_xavier_normal(layer=last_pfc,
                                         activation=pol_output_activation,
                                         b=output_b_init_val)
            self.__setattr__("pfc_last{}".format(pol_idx), last_pfc)
            self._pfc_lasts.append(last_pfc)
            self.add_policies_module("pfc_last{}".format(pol_idx), last_pfc,
                                     idx=pol_idx)

        # Unshared Mixing-Weights Hidden Layers
        if unshared_mix_hidden_sizes is not None:
            for ii, next_size in enumerate(unshared_mix_hidden_sizes):
                mfc = nn.Linear(mixture_in_size, next_size)
                ptu.layer_init_xavier_normal(layer=mfc,
                                             activation=hidden_activation,
                                             b=hidden_b_init_val)
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
        ptu.layer_init_xavier_normal(layer=mfc_last,
                                     activation=mix_output_activation,
                                     b=output_b_init_val)
        self.__setattr__("mfc_last", mfc_last)
        self.mfc_last = mfc_last
        # Add it to specific dictionaries
        self.add_mixing_module("mfc_last", mfc_last)

        # Multi-Policy Log-Stds Last Layers
        self.stds = stds
        self.log_std = list()
        if stds is None:
            self._pfc_log_std_lasts = list()
            for pol_idx in range(self._n_subpolicies):
                last_pfc_log_std = nn.Linear(multipol_in_size, action_dim)
                ptu.layer_init_xavier_normal(layer=last_pfc_log_std,
                                             activation=pol_output_activation,
                                             b=output_b_init_val)
                self.__setattr__("pfc_log_std_last{}".format(pol_idx),
                                 last_pfc_log_std)
                self._pfc_log_std_lasts.append(last_pfc_log_std)
                self.add_policies_module("pfc_log_std_last{}".format(pol_idx),
                                         last_pfc_log_std, idx=pol_idx)

        else:
            for std in stds:
                self.log_std.append(np.log(stds))
                assert LOG_SIG_MIN <= self.log_std[-1] <= LOG_SIG_MAX

        self._reparameterize = reparameterize

    def get_action(self, obs_np, **kwargs):
        action, info_dict = self.get_actions(obs_np[None], **kwargs)

        for key, val in info_dict.items():
            info_dict[key] = val[0, :]

        return action[0, :], info_dict

    def get_actions(self, obs_np, **kwargs):
        action, torch_info_dict = self.eval_np(obs_np, **kwargs)

        info_dict = dict()
        for key, vals in torch_info_dict.items():
            if key in ['mixing_coeff']:
                info_dict[key] = np_ify(torch_info_dict[key])

        return action, info_dict

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=False,
            pol_idx=None,
            optimize_policies=True,
            print_debug=False,
    ):
        """

        Args:
            obs (Tensor): Observation(s)
            deterministic (bool):
            return_log_prob (bool):
            pol_idx (int):
            optimize_policies (bool):
            print_debug (bool):

        Returns:
            action (Tensor):
            pol_info (dict):

        """
        h = obs

        # ############# #
        # Shared Layers #
        # ############# #
        if print_debug:
            print('***', 'OBS', '***')
            print(h)
            print('***', 'SFCS', '***')
        for ss, fc in enumerate(self._sfcs):
            h = fc(h)

            if self._mixture_layer_norm:
                h = self._sfc_norms[ss](h)

            h = self._hidden_activation(h)
            if print_debug:
                print(h)

        # ############## #
        # Multi Policies #
        # ############## #
        hs = [h.clone() for _ in range(self._n_subpolicies)]

        if print_debug:
            print('***', 'HS', '***')
            for hh in hs:
                print(hh)

        if print_debug:
            print('***', 'PFCS', '***')
        # Hidden Layers
        if len(self._pfcs) > 0:
            for pp in range(self._n_subpolicies):
                if print_debug:
                    print(pp)

                for ii, fc in enumerate(self._pfcs[pp]):
                    hs[pp] = fc(hs[pp])

                    if self._policies_layer_norm:
                       hs[pp] = self._pfc_norms[pp][ii](hs[pp])

                    hs[pp] = self._hidden_activation(hs[pp])

                    if print_debug:
                        print(hs[pp])

        # Last Mean Layers
        means_list = \
            [(self._pol_output_activation(self._pfc_lasts[pp](hs[pp]))).unsqueeze(dim=-1)
             for pp in range(self._n_subpolicies)]

        if print_debug:
            print('***', 'LAST_PFCS', '***')
            for tt, tensor in enumerate(means_list):
                print(tt, '\n', tensor)

        means = torch.cat(means_list, dim=-1)

        if print_debug:
            print('***', 'CONCATENATED MEANS', '***')
            print(means)

        # Last Log-Std Layers
        if self.stds is None:
            stds_list = [
                (torch.clamp(
                    self._pol_output_activation(self._pfc_log_std_lasts[pp](hs[pp])),
                    min=LOG_SIG_MIN, max=LOG_SIG_MAX)
                ).unsqueeze(dim=-1)
                for pp in range(self._n_subpolicies)]

            log_stds = torch.cat(stds_list, dim=-1)
            stds = torch.exp(log_stds)

        else:
            stds = self.stds
            log_stds = self.log_std

        # Detach gradients if we don't want to optimize policies parameters
        if not optimize_policies:
            means = means.detach()
            stds = stds.detach()
            log_stds = log_stds.detach()

        # ############## #
        # Mixing Weigths #
        # ############## #
        mh = h.clone()

        if print_debug:
            print('***', 'MH', '***')
            print(mh)
            print('***', 'MFCS', '***')
        if len(self._mfcs) > 0:
            for mm, mfc in enumerate(self._mfcs):
                mh = mfc(mh)

                if self._mixture_layer_norm:
                    mh = self._norm_mfcs[mm](mh)

                mh = self._hidden_activation(mh)
                if print_debug:
                    print(mh)

        # log_mixture_coeff = self.mix_output_activation(self.mfc_last(mh))
        # if print_debug:
        #     print('***', 'LAST_MFC', '***')
        #     print(log_mixture_coeff)
        #

        # log_mixture_coeff = torch.clamp(log_mixture_coeff,
        #                                 min=LOG_MIX_COEFF_MIN,
        #                                 max=LOG_MIX_COEFF_MAX)  # NxK

        # mixture_coeff = nn.Softmax(dim=-1)(self._mixing_temperature *
        #     log_mixture_coeff.reshape(-1, self.action_dim, self._n_subpolicies)
        # )

        mixture_coeff = \
            self.mfc_last(mh).reshape(-1, self.action_dim, self._n_subpolicies)

        # mixture_coeff = nn.Softmax(dim=-1)(self._mixing_temperature *
        #                                    log_mixture_coeff.reshape(-1, self.action_dim, self._n_subpolicies)
        #                                    )

        # # NO nonlinear transformation
        # mixture_coeff = self.mfc_last(mh).reshape(-1, self.action_dim, self._n_subpolicies)



        # # TODO: UNCOMMENT FOR DEBUGGING
        # if torch.isnan(mixture_coeff).any():
        #     for name, param in self.named_parameters():
        #         print(name, '\n', param)
        #         print('-')
        #     print('\n***'*5)
        #     for name, param in self.named_parameters():
        #         print('grad_'+name, '\n', param.grad)
        #         print('-')
        #     print('\n***'*5)
        #     print('---')
        #     print('h:', h)
        #     print('mh:', mh.reshape(-1, self.action_dim, self._n_subpolicies))
        #     print('mfc_last(mh)',self.mfc_last(mh).reshape(-1, self.action_dim, self._n_subpolicies))
        #     raise ValueError('Some mixture coeff(s) is(are) NAN:',
        #                      mixture_coeff)


        if pol_idx is None:
            # Calculate weighted means and stds (and log_stds)
            mean = torch.sum(means*mixture_coeff, dim=-1, keepdim=False)

            if print_debug:
                print('***', 'WEIGHTED MEAN', '***')
                print(mean)

            # # BEFORE 23/09
            # log_std = torch.clamp(
            #     torch.logsumexp(log_stds +
            #               torch.log(torch.sqrt(mixture_coeff**2) + EPS),
            #               dim=-1, keepdim=False),
            #     min=LOG_SIG_MIN, max=LOG_SIG_MAX
            # )
            #
            # std = torch.exp(log_std)

            variance = torch.sum((stds*mixture_coeff)**2, dim=-1, keepdim=False)
            std = torch.sqrt(variance)
            std = torch.clamp(std, min=SIG_MIN, max=SIG_MAX)
            log_std = torch.log(std)

            # log_std = \
            #     torch.logsumexp(log_stds + log_mixture_coeff.reshape(-1, self.action_dim, self._n_subpolicies), dim=-1,
            #               keepdim=False) \
            #     - torch.logsumexp(log_mixture_coeff, dim=-1, keepdim=True)

            # log_std = torch.log(std)

        else:
            indices = ptu.LongTensor([pol_idx])
            mean = \
                torch.index_select(means, dim=-1, index=indices).squeeze(-1)
            std = \
                torch.index_select(stds, dim=-1, index=indices).squeeze(-1)
            log_std = \
                torch.index_select(log_stds, dim=-1, index=indices).squeeze(-1)

        pre_tanh_value = None
        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None

        if deterministic:
            action = torch.tanh(mean)
            if print_debug:
                print('***', 'ACTION', '***')
                print(action)
        else:
            tanh_normal = TanhNormal(mean, std)

            if self._reparameterize:
                action, pre_tanh_value = tanh_normal.rsample(
                    return_pretanh_value=True
                )
            else:
                action, pre_tanh_value = tanh_normal.sample(
                    return_pretanh_value=True
                )

            if return_log_prob:
                log_prob = tanh_normal.log_prob(action,
                                                pre_tanh_value=pre_tanh_value)
                log_prob = log_prob.sum(dim=-1, keepdim=True)

        info_dict = dict(
            mean=mean,
            log_std=log_std,
            log_prob=log_prob,
            expected_log_prob=expected_log_prob,
            std=std,
            mean_action_log_prob=mean_action_log_prob,
            pre_tanh_value=pre_tanh_value,
            # log_mixture_coeff=log_mixture_coeff,
            mixing_coeff=mixture_coeff,
            pol_means=means_list,
        )

        return action, info_dict

    def log_action(self, actions, obs, pol_idx=None):
        raise NotImplementedError
        # h = obs
        # # Shared Layers
        # for fc in self.sfcs:
        #     h = self.hidden_activation(fc(h))
        #
        # # Mixing Weigths
        # mh = h
        # if len(self.mfcs) > 0:
        #     for mfc in self.mfcs:
        #         mh = self.hidden_activation(mfc(mh))
        # log_mixture_coeff = self.output_activation(self.mfc_last(mh))
        #
        # log_mixture_coeff = torch.clamp(log_mixture_coeff,
        #                                 min=LOG_MIX_COEFF_MIN)  # NxK
        #
        # mixture_coeff = torch.exp(log_mixture_coeff) \
        #                 / torch.sum(torch.exp(log_mixture_coeff), dim=-1,
        #                             keepdim=True)
        #
        # # Multi Policies
        # hs = [h for _ in range(self._n_subpolicies)]
        #
        # if len(self.pfcs) > 0:
        #     for ii in range(self._n_subpolicies):
        #         for i, fc in enumerate(self.pfcs[ii]):
        #             hs[ii] = self.hidden_activation(fc(hs[ii]))
        #
        # means = torch.cat(
        #     [(self.output_activation(self.pfc_lasts[ii](hs[ii]))).unsqueeze(dim=-1)
        #      for ii in range(self._n_subpolicies)
        #      ], dim=-1)
        #
        # if self.stds is None:
        #     log_stds = torch.cat(
        #         [torch.clamp((self.output_activation(
        #             self.last_pfc_log_stds[ii](hs[ii])).unsqueeze(dim=-1)
        #                       ), min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        #          for ii in range(self._n_subpolicies)],
        #         dim=-1
        #     )
        #     stds = torch.exp(log_stds)
        # else:
        #     stds = self.stds
        #
        # # Calculate weighted means and stds (and log_stds)
        # weighted_mean = \
        #     torch.sum(means*mixture_coeff.unsqueeze(-2),
        #               dim=-1)
        #
        # weighted_std = \
        #     torch.sum(stds*mixture_coeff.unsqueeze(-2),
        #               dim=-1)
        #
        # if pol_idx is None:
        #     mean = weighted_mean
        #     std = weighted_std
        # else:
        #     indices = ptu.LongTensor([pol_idx])
        #     mean = \
        #         torch.index_select(means, dim=-1, index=indices).squeeze(-1)
        #     std = \
        #         torch.index_select(stds, dim=-1, index=indices).squeeze(-1)
        #
        # tanh_normal = TanhNormal(mean, std)
        #
        # log_prob = torch.sum(tanh_normal.log_prob(actions),
        #                      dim=-1, keepdim=True)
        #
        # return log_prob



        # z = (actions - mean)/stds
        # return -0.5 * torch.sum(torch.mul(z, z), dim=-1, keepdim=True)

    @property
    def n_heads(self):
        return self._n_subpolicies

    @property
    def n_subpolicies(self):
        return self._n_subpolicies

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

    @property
    def reparameterize(self):
        return self._reparameterize
