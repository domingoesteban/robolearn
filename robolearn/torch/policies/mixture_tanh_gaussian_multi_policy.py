import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions import Multinomial
from robolearn.torch.core import PyTorchModule
import robolearn.torch.pytorch_util as ptu
from robolearn.policies.base import ExplorationPolicy
from robolearn.torch.nn import identity
from robolearn.torch.distributions import TanhNormal
from robolearn.torch.ops import logsumexp

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
LOG_MIX_COEFF_MIN = -10


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
            mix_hidden_w_init(fc.weight)
            ptu.fill(fc.bias, mix_hidden_b_init_val)
            self.__setattr__("mixfc{}".format(i), fc)
            self.mixfcs.append(fc)
            in_size = next_size

        self.last_mixfc = nn.Linear(in_size, self._multipolicy.n_heads)

        # if init_mixt_coeff is None:
        #     init_mixt_coeff = np.array([1. / len(self.pol_idxs)
        #                                 for _ in pol_idxs])
        # mixture_coeff = FloatTensor([1.0, 1.0])
        # self._mixture_coeff = nn.Parameter(mixture_coeff, requires_grad=True)

        # Label to detach gradients from multipolicy
        self._optimize_multipolicy = optimize_multipolicy

    def get_action(self, *args, **kwargs):
        return self.get_actions(*args, **kwargs), dict()

    # def get_actions(self, obs_np, deterministic=False):
    def get_actions(self, *args, **kwargs):
        # Return only action
        return self.eval_np(*args, **kwargs)[0]

    def forward(self, *nn_input, return_log_prob=False, **nn_kwargs):
        # Get Values from multipolicy
        nn_kwargs['pol_idxs'] = self.pol_idxs
        actions, policy_infos = \
            self._multipolicy(*nn_input, return_log_prob=return_log_prob,
                              **nn_kwargs)

        log_actions = policy_infos['log_prob']

        actions_concat = torch.cat([action.unsqueeze(dim=-1)
                                    for action in actions], dim=-1)  # NxAxK
        # print('actions_concat', actions_concat.shape)

        if not self._optimize_multipolicy:
            actions_concat = actions_concat.detach()

        # Compute mixture coefficients
        h = nn_input[0]
        for i, fc in enumerate(self.mixfcs):
            h = self.mix_hidden_activation(fc(h))
        log_mixture_coeff = self.last_mixfc(h)

        log_mixture_coeff = torch.clamp(log_mixture_coeff,
                                        max=LOG_MIX_COEFF_MIN)  # NxK
        # print('log_mix_coef', log_mixture_coeff.shape)

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
                - logsumexp(log_actions_concat, dim=-1, keepdim=True)

            # weighted_log_action = \
            #     torch.sum(log_actions_concat * log_mixture_coeff.unsqueeze(-2),
            #               dim=-1) \
            #     / torch.sum(log_mixture_coeff, dim=-1, keepdim=True)
        else:
            weighted_log_action = None

        dict_output = dict(
            log_action=weighted_log_action)

        return weighted_action, dict_output

    @staticmethod
    def get_output_labels():
        return ['action', 'pol_dict']

