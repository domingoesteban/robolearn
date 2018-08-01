import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.distributions import Multinomial
from robolearn.torch.core import PyTorchModule
from robolearn.torch.core import np_ify
import robolearn.torch.pytorch_util as ptu
from robolearn.policies.base import ExplorationPolicy
from robolearn.torch.nn import identity
from robolearn.torch.distributions import TanhNormal
from robolearn.torch.ops import logsumexp

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
LOG_MIX_COEFF_MIN = -10
LOG_MIX_COEFF_MAX = -4.5e-5



class MixtureTanhGaussianMultiPolicy(PyTorchModule, ExplorationPolicy):
    def __init__(self, multipolicy, mix_hidden_sizes, pol_idxs=None,
                 mix_hidden_w_init=ptu.xavier_init,
                 mix_hidden_b_init_val=0,
                 mix_hidden_activation=F.relu,
                 optimize_multipolicy=False,
                 reuse_shared=False):
        self.save_init_params(locals())
        super(MixtureTanhGaussianMultiPolicy, self).__init__()

        self._multipolicy = multipolicy

        self.input_size = self._multipolicy.input_size

        if pol_idxs is None:
            n_heads = self._multipolicy.n_heads
            pol_idxs = list(range(n_heads))
        self.pol_idxs = pol_idxs

        # TODO: ASSUMING SAME ACTION DIMENSION
        self._action_dim = self._multipolicy.action_dim

        # Mixture Coefficients
        # TODO: MAYBE WE CAN REUSE LATER THE SHARED LAYERS OF THE MULTIPOLICY
        self.mix_hidden_activation = mix_hidden_activation
        self.mixfcs = list()
        in_size = self.input_size
        for i, next_size in enumerate(mix_hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            # mix_hidden_w_init(fc.weight)
            nn.init.xavier_normal_(fc.weight.data,
                                   gain=nn.init.calculate_gain('relu'))
            ptu.fill(fc.bias, mix_hidden_b_init_val)
            self.__setattr__("mixfc{}".format(i), fc)
            self.mixfcs.append(fc)
            in_size = next_size

        self.last_mixfc = nn.Linear(in_size, self._multipolicy.n_heads)
        nn.init.xavier_normal_(self.last_mixfc.weight.data,
                               gain=nn.init.calculate_gain('linear'))

        # if init_mixt_coeff is None:
        #     init_mixt_coeff = np.array([1. / len(self.pol_idxs)
        #                                 for _ in pol_idxs])
        # mixture_coeff = FloatTensor([1.0, 1.0])
        # self._mixture_coeff = nn.Parameter(mixture_coeff, requires_grad=True)

        # Label to detach gradients from multipolicy
        self._optimize_multipolicy = optimize_multipolicy

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

    def forward(self,
                obs,
                deterministic=False,
                return_log_prob=False,
                optimize_policies=False,
                **nn_kwargs
                ):

        # Get Values from multipolicy
        pol_idxs = self.pol_idxs
        actions, policy_infos = \
            self._multipolicy(obs,
                              deterministic=deterministic,
                              return_log_prob=return_log_prob,
                              **nn_kwargs)

        log_actions = policy_infos['log_prob']

        actions_concat = torch.cat([action.unsqueeze(dim=-1)
                                    for action in actions], dim=-1)  # NxAxK
        # print('actions_concat', actions_concat.shape)

        if not self._optimize_multipolicy:
            actions_concat = actions_concat.detach()

        # Compute mixture coefficients
        h = obs
        for i, fc in enumerate(self.mixfcs):
            h = self.mix_hidden_activation(fc(h))
        log_mixture_coeff = self.last_mixfc(h)

        log_mixture_coeff = torch.clamp(log_mixture_coeff,
                                        min=LOG_MIX_COEFF_MIN,
                                        max=LOG_MIX_COEFF_MAX)  # NxK

        mixture_coeff = torch.exp(log_mixture_coeff) \
                        / torch.sum(torch.exp(log_mixture_coeff), dim=-1,
                                    keepdim=True)

        if torch.isnan(mixture_coeff).any():
            raise ValueError('Any mixture coeff is NAN:',
                             mixture_coeff)

        # print(log_mixture_coeff)
        # print(mixture_coeff)
        # print('--')

        # TODO: CHECK IF NOT PROPAGATING GRADIENTS HERE IS A PROBLEM
        # Sample latent variables
        z = Multinomial(logits=log_mixture_coeff).sample()  # NxK

        # Choose mixture component corresponding
        weighted_action = torch.sum(actions_concat*z.unsqueeze(-2), dim=-1)

        # weighted_action = \
        #     torch.sum(actions_concat * log_mixture_coeff.unsqueeze(-2), dim=-1) \
        #     / torch.sum(log_mixture_coeff, dim=-1, keepdim=True)

        if return_log_prob is True:
            log_actions_concat = \
                torch.cat([log_action.unsqueeze(dim=-1)
                           for log_action in log_actions], dim=-1)

            if not self._optimize_multipolicy:
                log_actions_concat = log_actions_concat.detach()

            log_actions_concat = torch.sum(log_actions_concat*z.unsqueeze(-1),
                                           dim=-2)
            weighted_log_action = \
                logsumexp(log_actions_concat + log_mixture_coeff, dim=-1,
                          keepdim=True) \
                - logsumexp(log_mixture_coeff, dim=-1, keepdim=True)

            # weighted_log_action = \
            #     torch.sum(log_actions_concat * log_mixture_coeff.unsqueeze(-2),
            #               dim=-1) \
            #     / torch.sum(log_mixture_coeff, dim=-1, keepdim=True)
        else:
            weighted_log_action = None

        """
        mixture_coeff = torch.exp(log_mixture_coeff) \
                        / torch.sum(torch.exp(log_mixture_coeff), dim=-1,
                                    keepdim=True)

        weighted_action = \
            torch.sum(actions_concat*mixture_coeff.unsqueeze(-2),
                      dim=-1)

        if return_log_prob is True:
            log_actions_concat = \
                torch.cat([log_action.unsqueeze(dim=-1)
                           for log_action in log_actions], dim=-1)

            if not self._optimize_multipolicy:
                log_actions_concat = log_actions_concat.detach()

            log_actions_concat = \
                torch.sum(log_actions_concat*mixture_coeff.unsqueeze(-1),
                          dim=-2)

            weighted_log_action = \
                logsumexp(log_actions_concat + log_mixture_coeff, dim=-1,
                          keepdim=True) \
                - logsumexp(log_mixture_coeff, dim=-1, keepdim=True)
        else:
            weighted_log_action = None
        """

        info_dict = dict(
            log_action=weighted_log_action,
            mixing_coeff=mixture_coeff,
        )

        return weighted_action, info_dict

    @property
    def n_heads(self):
        return self._multipolicy.n_heads

