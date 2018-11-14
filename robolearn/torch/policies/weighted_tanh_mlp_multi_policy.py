import torch
from torch import nn as nn
from robolearn.torch.core import PyTorchModule
from robolearn.torch.core import np_ify
# from robolearn.torch.nn import LayerNorm
from torch.nn.modules.normalization import LayerNorm
import robolearn.torch.utils.pytorch_util as ptu
from robolearn.models.policies import Policy
from collections import OrderedDict
from itertools import chain

LOG_MIX_COEFF_MIN = -10
LOG_MIX_COEFF_MAX = -1e-6  #-4.5e-5
LOG_MIX_COEFF_MIN = -1
LOG_MIX_COEFF_MAX = 1  #-4.5e-5

EPS = 1e-12


class WeightedTanhMlpMultiPolicy(PyTorchModule, Policy):
    """
    Usage:

    ```
    policy = WeightedTanhMlpMultiPolicy(...)
    action, policy_dict = policy(obs)
    ```

    """
    def __init__(
            self,
            obs_dim,
            action_dim,
            n_policies,
            shared_hidden_sizes=None,
            unshared_hidden_sizes=None,
            unshared_mix_hidden_sizes=None,
            hidden_activation='relu',
            hidden_w_init='xavier_normal',
            hidden_b_init_val=1e-2,
            output_w_init='xavier_normal',
            output_b_init_val=1e-2,
            pol_output_activation='linear',
            mix_output_activation='linear',
            shared_layer_norm=False,
            policies_layer_norm=False,
            mixture_layer_norm=False,
            reparameterize=True,
            softmax_weights=False,
            mixing_temperature=1.,
            **kwargs
    ):
        self.save_init_params(locals())
        PyTorchModule.__init__(self)
        Policy.__init__(self, action_dim)

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
        if softmax_weights:
            self._softmax_fcn = nn.Softmax(dim=1)
        else:
            self._softmax_fcn = None

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

        self._reparameterize = reparameterize

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
            pol_idx=None,
            optimize_policies=True,
            print_debug=False,
    ):
        """

        Args:
            obs (Tensor): Observation(s)
            pol_idx (int):
            optimize_policies (bool):
            print_debug (bool):

        Returns:
            action (Tensor):
            pol_info (dict):

        """
        h = obs
        nbatch = obs.shape[0]

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

        # Last Layers
        outputs_list = \
            [(
                 self._pol_output_activation(self._pfc_lasts[pp](hs[pp]))
              ).unsqueeze(dim=1)
             for pp in range(self._n_subpolicies)]

        if print_debug:
            print('***', 'LAST_PFCS', '***')
            for tt, tensor in enumerate(outputs_list):
                print(tt, '\n', tensor)

        outputs = torch.cat(outputs_list, dim=1)

        if print_debug:
            print('***', 'CONCATENATED OUTPUTS', '***')
            print(outputs)

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

        # NO nonlinear transformation
        mixture_coeff = \
            self.mfc_last(mh).reshape(-1, self._n_subpolicies, self.action_dim)

        if self._softmax_fcn is not None:
            mixture_coeff = self._softmax_fcn(mixture_coeff)

        if pol_idx is None:
            # Calculate weighted output

            if optimize_policies:
                output = torch.sum(outputs*mixture_coeff, dim=1, keepdim=False)
            else:
                output = torch.sum(outputs.detach()*mixture_coeff, dim=1,
                                   keepdim=False)

            if print_debug:
                print('***', 'WEIGHTED MEAN', '***')
                print(output)

        else:
            indices = ptu.LongTensor([pol_idx])
            output = \
                torch.index_select(outputs, dim=1, index=indices).squeeze(1)

        action = torch.tanh(output)
        actions = torch.tanh(outputs)
        if print_debug:
            print('***', 'ACTION', '***')
            print(action)

        info_dict = dict(
            pre_tanh_value=output,
            mixing_coeff=mixture_coeff,
            pol_actions=actions,
            pol_pre_tanh_values=outputs,
        )

        return action, info_dict

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

    @property
    def reparameterize(self):
        return self._reparameterize
