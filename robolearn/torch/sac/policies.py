import numpy as np
import torch
from torch import nn as nn

import robolearn.torch.pytorch_util as ptu
from robolearn.torch.pytorch_util import FloatTensor
from robolearn.torch.core import PyTorchModule, np_ify
from robolearn.torch.nn import LayerNorm
from robolearn.policies import ExplorationPolicy, Policy
from robolearn.torch.distributions import TanhNormal
from torch.distributions import Normal
from torch.distributions import Bernoulli
from torch.distributions import Multinomial
from robolearn.torch.nn import Mlp
from robolearn.torch.nn import identity
from robolearn.torch.ops import logsumexp

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
LOG_MIX_COEFF_MIN = -10


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
            shared_hidden_sizes,
            obs_dim,
            action_dims,
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

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = obs_dim
        self.output_sizes = action_dims
        self._n_policies = len(action_dims)
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

        for pol_idx, output_size in enumerate(self.output_sizes):
            last_fc = nn.Linear(in_size, output_size)
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
                                            action_dims[pol_idx])
                hidden_w_init(last_fc_log_std.weight)
                ptu.fill(last_fc_log_std.bias, hidden_b_init_val)
                self.__setattr__("last_fc_log_std{}".format(pol_idx),
                                 last_fc_log_std)
                self.last_fc_log_stds.append(last_fc_log_std)

        else:
            for std in stds:
                self.log_std.append(np.log(stds))
                assert LOG_SIG_MIN <= self.log_std[-1] <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False, pol_idxs=None):
        if pol_idxs is None:
            pol_idxs = list(range(self._n_policies))

        outputs = self.get_actions(obs_np[None], deterministic=deterministic,
                                   pol_idxs=pol_idxs)
        actions = [np_ify(outputs[0][idx][0, :]) for idx in range(len(pol_idxs))]
        means = [np_ify(outputs[1][idx]) for idx in range(len(pol_idxs))]
        log_stds = [np_ify(outputs[2][idx]) for idx in range(len(pol_idxs))]
        log_probs = [np_ify(outputs[3][idx]) for idx in range(len(pol_idxs))]
        return actions, {'mean': means,
                         'log_std': log_stds,
                         'log_prob': log_probs}

    def get_actions(self, obs_np, deterministic=False, pol_idxs=None):
        if pol_idxs is None:
            pol_idxs = list(range(self._n_policies))

        outputs = self.eval_np(obs_np, deterministic=deterministic,
                               pol_idxs=pol_idxs)

        outputs = [[np_ify(tensor) for tensor in data]
                   for data in outputs]

        return tuple(outputs)

    def forward(
            self,
            obs,
            pol_idxs=None,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
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
                                          LOG_SIG_MIN, LOG_SIG_MAX)
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
                    # print('ACAA_NOOO')
                    # input('LO SABIA')
                    # action, pre_tanh_value = tanh_normal.sample(
                    actions[ii], pre_tanh_values[ii] = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                    log_probs[ii] = tanh_normal.log_prob(
                        actions[ii],
                        pre_tanh_value=pre_tanh_values[ii]
                    )
                    log_probs[ii] = log_probs[ii].sum(dim=-1, keepdim=True)
                else:
                    # print('ACAA')
                    # actions[ii] = tanh_normal.sample()
                    actions[ii], pre_tanh_values[ii] = \
                        tanh_normal.rsample(return_pretanh_value=True)

                # print(obs)
                # print('$$$')
                # print('mean', mean)
                # print('std', std)
                # print('action %02d' % ii, actions[ii])
                # input('PISDSDA')

        return (
            actions, means, log_stds, log_probs, expected_log_probs, stds,
            mean_action_log_probs, pre_tanh_values,
        )

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
                                           LOG_SIG_MIN, LOG_SIG_MAX)
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

    @staticmethod
    def get_output_labels():
        return ['action', 'mean', 'log_std', 'log_prob', 'expected_log_prob',
                'stds', 'mean_action_log_prob', 'pre_tanh_value']

    @property
    def n_heads(self):
        return len(self.last_fcs)


class WeightedTanhGaussianMultiPolicy(PyTorchModule, ExplorationPolicy):
    def __init__(self, multipolicy, mix_hidden_sizes, pol_idxs=None,
                 mix_hidden_w_init=ptu.xavier_init,
                 mix_hidden_b_init_val=0,
                 mix_hidden_activation=F.relu,
                 optimize_multipolicy=False):
        self.save_init_params(locals())
        super(WeightedTanhGaussianMultiPolicy, self).__init__()

        self._multipolicy = multipolicy

        self.input_size = self._multipolicy.input_size

        if pol_idxs is None:
            n_heads = self._multipolicy.n_heads
            pol_idxs = list(range(n_heads))
        self.pol_idxs = pol_idxs

        # TODO: ASSUMING SAME ACTION DIMENSION
        self._action_dim = self._multipolicy.output_sizes[0]

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
        outputs = self._multipolicy(*nn_input, return_log_prob=return_log_prob,
                                    **nn_kwargs)

        # Detach gradients of multipolicy
        if isinstance(outputs, tuple):
            # Actions are only the first (THEY SHOULD)
            actions = outputs[0]
            log_actions = outputs[3]
        else:
            actions = outputs

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
        # print('z', z.shape)

        # Choose mixture component corresponding
        # weighted_action = torch.sum(actions_concat*z.unsqueeze(-1), dim=-2)

        weighted_action = torch.sum(actions_concat*z.unsqueeze(-2), dim=-1)
        # print('z_unsq', (z.unsqueeze(-2)).shape)
        # print('weight_act', weighted_action.shape)
        # input('wuuu')

        # print('ete-->', actions_concat.ndimension())
        # if actions_concat.ndimension() > 2:
        #     print(actions_concat[0, :, 0])
        #     print(actions_concat[0, :, 1])
        #     print('^^')
        #     print(actions_concat[:4, :, :])
        #     print('###')
        #     print((z.unsqueeze(-2))[:4, :, :])
        #     print('@@@@@')
        #     print(weighted_action[:4, :])
        #     print(actions_concat.shape)
        #     input('pero')
        # else:
        #     print(actions_concat[:, 0])
        #     print(actions_concat[:, 1])
        #     print('^^')
        #     print(actions_concat[:, :])
        #     print('###')
        #     print((z.unsqueeze(-2))[:, :])
        #     print('@@@@@')
        #     print(weighted_action[:])
        #     print(actions_concat.shape)
        #     # input('pero')
        #     print('###\n\n')

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


class BernoulliTanhGaussianMultiPolicy(PyTorchModule, ExplorationPolicy):
    def __init__(self, multipolicy, prob=0.5):
        self.save_init_params(locals())
        super(BernoulliTanhGaussianMultiPolicy, self).__init__()

        self._multipolicy = multipolicy

        from torch.distributions import Binomial
        self._uni_dist = Bernoulli(probs=FloatTensor([prob]))

    def get_action(self, *args, **kwargs):
        kwargs['pol_idxs'] = [int(self._uni_dist.sample())]
        # outputs = self._multivalue.get_action(obs_np,
        #                                         deterministic=deterministic,
        #                                         _val_idxs=self._val_idxs)
        actions, info = self._multipolicy.get_action(*args, **kwargs)

        action = actions[0]

        return action, info

    # def get_actions(self, obs_np, deterministic=False):
    def get_actions(self, *args, **kwargs):
        kwargs['pol_idxs'] = [int(self._uni_dist.sample())]
        outputs = self._multipolicy.get_actions(*args, **kwargs)

        actions = outputs[0]

        action = actions[0]

        return action

    def forward(self, *nn_input):
        pol_idxs = [int(self._uni_dist.sample())]
        outputs = self._multipolicy(*nn_input, pol_idxs=pol_idxs)

        actions = outputs[0]

        action = actions[0]

        return action

    @staticmethod
    def get_output_labels():
        return ['action']


class MultiPolicySelector(PyTorchModule, ExplorationPolicy):
    def __init__(self, multipolicy, idx):
        self.save_init_params(locals())
        super(MultiPolicySelector, self).__init__()

        self._multipolicy = multipolicy
        self.idx = [idx]

    def get_action(self, *args, **kwargs):
        # print('pipipi')
        kwargs['pol_idxs'] = self.idx
        a, info = self._multipolicy.get_action(*args, **kwargs)

        a = a[0]
        info = dict(zip(info.keys(), [value[0] for value in info.values()]))
        # print('ACTIONpipipi', a)
        return a, info

    def get_actions(self, *args, **kwargs):
        # print('jojojo')
        kwargs['pol_idxs'] = self.idx
        a = self._multipolicy.get_actions(*args, **kwargs)

        a = a[0][0]  # Only return actions

        # print('ACTIONjojoj', a)
        return a

    def forward(self, *nn_input):
        # print('wuwuwuwuwu')
        outputs = self._multipolicy(*nn_input, pol_idxs=self.idx)
        action = outputs[0][0]

        # print('ACTIONwuwuwuwu', action)
        return action

    @staticmethod
    def get_output_labels():
        return ['action']
