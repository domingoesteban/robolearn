import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.distributions import Multinomial
from robolearn.torch.core import PyTorchModule
from robolearn.torch.core import np_ify
from robolearn.torch.nn import LayerNorm
import robolearn.torch.pytorch_util as ptu
from robolearn.policies.base import ExplorationPolicy
from robolearn.torch.nn import identity
from robolearn.torch.distributions import TanhNormal
from robolearn.torch.ops import logsumexp
from collections import OrderedDict
from itertools import chain

# LOG_SIG_MAX = 2
# LOG_SIG_MIN = -20
LOG_SIG_MAX = 0.0  # 2
LOG_SIG_MIN = -3.0  # 20

# LOG_MIX_COEFF_MIN = -10
# LOG_MIX_COEFF_MAX = -1e-6  #-4.5e-5
LOG_MIX_COEFF_MIN = -1
LOG_MIX_COEFF_MAX = 1  #-4.5e-5


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
            embedding_dim=None,
            shared_hidden_sizes=None,
            embedding_hidden_sizes=None,
            composition_hidden_sizes=None,
            policy_hidden_sizes=None,
            stds=None,
            hidden_activation=F.relu,
            hidden_w_init=ptu.xavier_init,
            hidden_b_init_val=1e-6,
            output_w_init=ptu.xavier_init,
            output_b_init_val=1e-6,
            pol_output_activation=identity,
            mix_output_activation=F.tanh,
            shared_layer_norm=False,
            policies_layer_norm=False,
            mixture_layer_norm=False,
            policy_layer_norm=False,
            mixing_temperature=1.,
            reparameterize=True,
            **kwargs
    ):
        self.save_init_params(locals())
        super(TanhGaussianComposedMultiPolicy, self).__init__()
        ExplorationPolicy.__init__(self, action_dim)

        self.input_size = obs_dim
        self.output_sizes = action_dim
        self._n_policies = n_policies
        if embedding_dim is None:
            embedding_dim = action_dim
        self._embedding_dim = embedding_dim

        # Activation Fcns
        self.hidden_activation = hidden_activation
        self.pol_output_activation = pol_output_activation
        self.mix_output_activation = mix_output_activation
        # Normalization Layer Flags
        self.shared_layer_norm = shared_layer_norm
        self.policies_layer_norm = policies_layer_norm
        self.mixture_layer_norm = mixture_layer_norm
        # Layers Lists
        self.sfcs = []  # Shared Layers
        self.norm_sfcs = []  # Norm. Shared Layers
        self.efcs = [list() for _ in range(self._n_policies)]  # Policies Layers
        self.norm_efcs = [list() for _ in range(self._n_policies)]  # N. Pol. L.
        self.last_efcs = []  # Last Policies Layers
        self.mfcs = []  # Mixing Layers
        self.norm_mfcs = []  # Norm. Mixing Layers
        # self.last_mfc = None

        self._mixing_temperature = mixing_temperature  # Hyperparameter for exp.

        # Initial size = Obs size
        in_size = self.input_size

        # Ordered Dictionaries for specific modules/parameters
        self._shared_modules = OrderedDict()
        self._shared_parameters = OrderedDict()
        self._embedding_modules = [OrderedDict() for _ in range(n_policies)]
        self._embedding_parameters = [OrderedDict() for _ in range(n_policies)]
        self._composition_modules = OrderedDict()
        self._composition_parameters = OrderedDict()
        self._policy_modules = OrderedDict()
        self._policy_parameters = OrderedDict()

        # ############# #
        # Shared Layers #
        # ############# #
        if shared_hidden_sizes is not None:
            for i, next_size in enumerate(shared_hidden_sizes):
                sfc = nn.Linear(in_size, next_size)
                nn.init.xavier_normal_(sfc.weight.data,
                                       gain=nn.init.calculate_gain('relu'))
                ptu.fill(sfc.bias, hidden_b_init_val)
                self.__setattr__("sfc{}".format(i), sfc)
                self.sfcs.append(sfc)
                self.add_shared_module("sfc{}".format(i), sfc)

                if self.shared_layer_norm:
                    ln = LayerNorm(next_size)
                    # ln = nn.BatchNorm1d(next_size)
                    self.__setattr__("norm_sfc{}".format(i), ln)
                    self.norm_sfcs.append(ln)
                    self.add_shared_module("norm_sfc{}".format(i), ln)
                in_size = next_size

        # Get the output_size of the shared layers
        embedding_in_size = in_size
        mixture_in_size = in_size

        # Embeddings Hidden Layers
        if embedding_hidden_sizes is not None:
            for i, next_size in enumerate(embedding_hidden_sizes):
                for pol_idx in range(self._n_policies):
                    efc = nn.Linear(embedding_in_size, next_size)
                    nn.init.xavier_normal_(efc.weight.data,
                                           gain=nn.init.calculate_gain('relu'))
                    ptu.fill(efc.bias, hidden_b_init_val)
                    self.__setattr__("efc{:02}_{}".format(pol_idx, i), efc)
                    self.efcs[pol_idx].append(efc)
                    self.add_policies_module("efc{:02}_{}".format(pol_idx, i), efc,
                                             idx=pol_idx)

                    if self.policies_layer_norm:
                        ln = LayerNorm(next_size)
                        # ln = nn.BatchNorm1d(next_size)
                        self.__setattr__("norm_efc{:02}_{}".format(pol_idx, i), ln)
                        self.norm_efcs[pol_idx].append(ln)
                        self.add_policies_module("norm_efc{:02}_{}".format(pol_idx,
                                                                        i),
                                                 ln, idx=pol_idx)
                embedding_in_size = next_size

        # Embeddings Last Layers
        for pol_idx in range(self._n_policies):
            last_efc = nn.Linear(embedding_in_size, self._embedding_dim)
            nn.init.xavier_normal_(last_efc.weight.data,
                                   gain=nn.init.calculate_gain('linear'))
            ptu.fill(last_efc.bias, output_b_init_val)
            self.__setattr__("last_efc{}".format(pol_idx), last_efc)
            self.last_efcs.append(last_efc)
            self.add_policies_module("last_efc{}".format(pol_idx), last_efc,
                                     idx=pol_idx)

        print(self)
        input('wuuu')

        # Unshared Mixing-Weights Hidden Layers
        if composition_hidden_sizes is not None:
            for i, next_size in enumerate(composition_hidden_sizes):
                mfc = nn.Linear(mixture_in_size, next_size)
                nn.init.xavier_normal_(mfc.weight.data,
                                       gain=nn.init.calculate_gain('relu'))
                ptu.fill(mfc.bias, hidden_b_init_val)
                self.__setattr__("mfc{}".format(i), mfc)
                self.mfcs.append(mfc)
                # Add it to specific dictionaries
                self.add_mixing_module("mfc{}".format(i), mfc)

                if self.mixture_layer_norm:
                    ln = LayerNorm(next_size)
                    # ln = nn.BatchNorm1d(next_size)
                    self.__setattr__("norm_mfc{}".format(i), ln)
                    self.norm_mfcs.append(ln)
                    self.add_mixing_module("norm_mfc{}".format(i), ln)
                mixture_in_size = next_size

        # Unshared Mixing-Weights Last Layers
        last_mfc = nn.Linear(mixture_in_size, self._n_policies)
        nn.init.xavier_normal_(last_mfc.weight.data,
                               gain=nn.init.calculate_gain('linear'))
        ptu.fill(last_mfc.bias, output_b_init_val)
        self.__setattr__("last_mfc", last_mfc)
        self.last_mfc = last_mfc
        # Add it to specific dictionaries
        self.add_mixing_module("last_mfc", last_mfc)

        self.stds = stds
        self.log_std = list()
        if stds is None:
            self.last_pfc_log_stds = list()
            for pol_idx in range(self._n_policies):
                last_hidden_size = obs_dim
                if embedding_hidden_sizes is None:
                    if len(shared_hidden_sizes) > 0:
                        last_hidden_size = shared_hidden_sizes[-1]
                else:
                    last_hidden_size = embedding_hidden_sizes[-1]
                last_pfc_log_std = nn.Linear(last_hidden_size,
                                             action_dim)
                nn.init.xavier_normal_(last_pfc_log_std.weight.data,
                                       gain=nn.init.calculate_gain('linear'))
                ptu.fill(last_pfc_log_std.bias, hidden_b_init_val)
                self.__setattr__("last_pfc_log_std{}".format(pol_idx),
                                 last_pfc_log_std)
                self.last_pfc_log_stds.append(last_pfc_log_std)
                self.add_policies_module("last_pfc_log_std{}".format(pol_idx),
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
        for ss, fc in enumerate(self.sfcs):
            h = fc(h)

            if self.mixture_layer_norm:
                h = self.norm_sfcs[ss](h)

            h = self.hidden_activation(h)
            if print_debug:
                print(h)

        # ############## #
        # Multi Policies #
        # ############## #
        hs = [h.clone() for _ in range(self._n_policies)]

        if print_debug:
            print('***', 'HS', '***')
            for hh in hs:
                print(hh)

        if print_debug:
            print('***', 'PFCS', '***')
        # Hidden Layers
        if len(self.pfcs) > 0:
            for pp in range(self._n_policies):
                if print_debug:
                    print(pp)

                for ii, fc in enumerate(self.pfcs[pp]):
                    hs[pp] = fc(hs[pp])

                    if self.policies_layer_norm:
                       hs[pp] = self.norm_pfcs[pp][ii](hs[pp])

                    hs[pp] = self.hidden_activation(hs[pp])

                    if print_debug:
                        print(hs[pp])

        # Last Mean Layers
        means_list = \
            [(self.pol_output_activation(self.last_pfcs[pp](hs[pp]))).unsqueeze(dim=-1)
             for pp in range(self._n_policies)]

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
                    self.pol_output_activation(self.last_pfc_log_stds[pp](hs[pp])),
                    min=LOG_SIG_MIN, max=LOG_SIG_MAX)
                ).unsqueeze(dim=-1)
                for pp in range(self._n_policies)]

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
        if len(self.mfcs) > 0:
            for mm, mfc in enumerate(self.mfcs):
                mh = mfc(mh)

                if self.mixture_layer_norm:
                    mh = self.norm_mfcs[mm](mh)

                mh = self.hidden_activation(mh)
                if print_debug:
                    print(mh)
        log_mixture_coeff = self.mix_output_activation(self.last_mfc(mh))
        if print_debug:
            print('***', 'LAST_MFC', '***')
            print(log_mixture_coeff)

        log_mixture_coeff = torch.clamp(log_mixture_coeff,
                                        min=LOG_MIX_COEFF_MIN,
                                        max=LOG_MIX_COEFF_MAX)  # NxK

        mixture_coeff = \
            nn.Softmax(dim=-1)(self._mixing_temperature*log_mixture_coeff)

        if torch.isnan(mixture_coeff).any():
            raise ValueError('Some mixture coeff(s) is(are) NAN:',
                             mixture_coeff)

        z = mixture_coeff  # NxK

        pre_tanh_value = None
        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None

        if pol_idx is None:
            # Composed (Intentional) Policy
            mean = torch.sum(means*z.unsqueeze(-2), dim=-1)
            std = torch.sum(stds*z.unsqueeze(-2), dim=-1)
            log_std = torch.sum(log_stds*z.unsqueeze(-2), dim=-1)
        else:
            # A specific Composable (Unintentional) policy
            indices = ptu.LongTensor([pol_idx])
            mean = \
                torch.index_select(means, dim=-1, index=indices).squeeze(-1)
            std = \
                torch.index_select(stds, dim=-1, index=indices).squeeze(-1)
            log_std = \
                torch.index_select(log_stds, dim=-1, index=indices).squeeze(-1)

        if deterministic:
            actions = torch.tanh(means)

            if pol_idx is None:
                action = torch.sum(actions*z.unsqueeze(-2), dim=-1)
            else:
                indices = ptu.LongTensor([pol_idx])
                action = \
                    torch.index_select(actions, dim=-1, index=indices).squeeze(-1)

            if print_debug:
                print('***', 'ACTION', '***')
                print(action)

        else:
            tanh_normals = TanhNormal(means, stds)

            if self._reparameterize:
                actions, pre_tanh_values = \
                    tanh_normals.rsample(return_pretanh_value=True)  # N x dA x K
            else:
                actions, pre_tanh_values = \
                    tanh_normals.sample(return_pretanh_value=True)  # N x dA x K

            log_probs = \
                tanh_normals.log_prob(actions,
                                      pre_tanh_value=pre_tanh_values)  # N x dA x K

            # Sum over actions
            log_probs = log_probs.sum(dim=-2, keepdim=True)  # N x 1 x K

            if pol_idx is None:
                action = torch.sum(actions*z.unsqueeze(-2), dim=-1)
                pre_tanh_value = \
                    torch.sum(pre_tanh_values*z.unsqueeze(-2), dim=-1)
                log_prob = torch.sum(log_probs*z.unsqueeze(-2), dim=-1)

                # log_prob = \
                #     logsumexp(log_probs + log_mixture_coeff.unsqueeze(-2),
                #               dim=-1, keepdim=False) \
                #     - logsumexp(log_mixture_coeff, dim=-1, keepdim=True)

            else:
                indices = ptu.LongTensor([pol_idx])
                action = \
                    torch.index_select(actions, dim=-1, index=indices).squeeze(-1)
                pre_tanh_value = \
                    torch.index_select(pre_tanh_values, dim=-1, index=indices).squeeze(-1)
                log_prob = \
                    torch.index_select(log_probs, dim=-1, index=indices).squeeze(-1)

        info_dict = dict(
            mean=mean,
            log_std=log_std,
            log_prob=log_prob,
            expected_log_prob=expected_log_prob,
            std=std,
            mean_action_log_prob=mean_action_log_prob,
            pre_tanh_value=pre_tanh_value,
            log_mixture_coeff=log_mixture_coeff,
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
        # log_mixture_coeff = self.output_activation(self.last_mfc(mh))
        #
        # log_mixture_coeff = torch.clamp(log_mixture_coeff,
        #                                 min=LOG_MIX_COEFF_MIN)  # NxK
        #
        # mixture_coeff = torch.exp(log_mixture_coeff) \
        #                 / torch.sum(torch.exp(log_mixture_coeff), dim=-1,
        #                             keepdim=True)
        #
        # # Multi Policies
        # hs = [h for _ in range(self._n_policies)]
        #
        # if len(self.pfcs) > 0:
        #     for ii in range(self._n_policies):
        #         for i, fc in enumerate(self.pfcs[ii]):
        #             hs[ii] = self.hidden_activation(fc(hs[ii]))
        #
        # means = torch.cat(
        #     [(self.output_activation(self.last_pfcs[ii](hs[ii]))).unsqueeze(dim=-1)
        #      for ii in range(self._n_policies)
        #      ], dim=-1)
        #
        # if self.stds is None:
        #     log_stds = torch.cat(
        #         [torch.clamp((self.output_activation(
        #             self.last_pfc_log_stds[ii](hs[ii])).unsqueeze(dim=-1)
        #                       ), min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        #          for ii in range(self._n_policies)],
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
        return self._n_policies

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
            idx_list = list(range(self._n_policies))
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
            idx_list = list(range(self._n_policies))
        elif isinstance(idx, list) or isinstance(idx, tuple):
            idx_list = idx
        else:
            idx_list = [idx]

        return chain(*[ptu.named_parameters(self._embedding_modules[idx],
                                            self._embedding_parameters[idx],
                                            **kwargs)
                       for idx in idx_list])

    def add_policies_module(self, name, module, idx=None):
        if idx is None:
            idx_list = list(range(self._n_policies))
        elif isinstance(idx, list) or isinstance(idx, tuple):
            idx_list = idx
        else:
            idx_list = [idx]

        for idx in idx_list:
            ptu.add_module(self._embedding_modules[idx], name, module)

    def mixing_parameters(self):
        """Returns an iterator over the mixing parameters.
        """
        for name, param in self.named_mixing_parameters():
            yield param

    def named_mixing_parameters(self, **kwargs):
        """Returns an iterator over mixing module parameters, yielding both the
        name of the parameter as well as the parameter itself
        """
        return ptu.named_parameters(self._composition_modules,
                                    self._composition_parameters,
                                    **kwargs)

    def add_mixing_module(self, name, module):
        ptu.add_module(self._composition_modules, name, module)

    @property
    def reparameterize(self):
        return self._reparameterize
