import numpy as np
import torch
from torch import nn as nn

import robolearn.torch.pytorch_util as ptu
from robolearn.policies.base import ExplorationPolicy, Policy
from robolearn.torch.distributions import TanhNormal
from torch.distributions import Normal
from robolearn.torch.nn import Mlp, identity


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class GaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    _i_policy = GaussianPolicy(...)
    action, mean, log_std, _ = _i_policy(obs)
    action, mean, log_std, _ = _i_policy(obs, deterministic=True)
    action, mean, log_std, log_prob = _i_policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            hidden_w_init=ptu.xavier_init,
            hidden_b_init_val=0,
            output_w_init=ptu.xavier_init,
            output_b_init_val=0,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            hidden_w_init=ptu.xavier_init,
            hidden_b_init_val=0,
            output_w_init=ptu.xavier_init,
            output_b_init_val=0,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            hidden_w_init(self.last_fc_log_std.weight)
            ptu.fill(self.last_fc_log_std.bias, hidden_b_init_val)

        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0][0, :], {'mean': actions[1],
                                  'log_std': actions[2],
                                  'log_prob': actions[3]}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)

        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = mean
        else:
            normal = Normal(mean, std)
            action = normal.rsample()
            if return_log_prob:
                log_prob = normal.log_prob(action)
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            else:
                action = normal.rsample()

        return (
            action, mean, log_std, log_prob, expected_log_prob, std,
            mean_action_log_prob, pre_tanh_value,
        )

    def log_action(self, action, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))

        mean = self.last_fc(h)

        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        normal = Normal(mean, std)
        log_prob = torch.sum(normal.log_prob(action), dim=-1, keepdim=True)

        # if torch.isnan(log_prob):
        #     print(log_prob, )

        return log_prob

        # z = (action - mean)/stds
        # return -0.5 * torch.sum(torch.mul(z, z), dim=-1, keepdim=True)


    @staticmethod
    def get_output_labels():
        return ['action', 'mean', 'log_std', 'log_prob', 'expected_log_prob',
                'stds', 'mean_action_log_prob', 'pre_tanh_value']

class MakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)

    def get_actions(self, observations):
        return self.stochastic_policy.get_actions(observations,
                                                  deterministic=True)


class StochasticPolicy(Mlp, ExplorationPolicy):
    """Stochastic NN _i_policy."""
    def __init__(self,
                 hidden_sizes,
                 obs_dim,
                 action_dim,
                 squash=True,
                 **kwargs
                 ):
        self.save_init_params(locals())  # For serialization


        # MLP Init
        super().__init__(
            hidden_sizes,
            input_size=obs_dim + action_dim,  # +action_dim id for stochasticity
            output_size=action_dim,
            **kwargs
        )

        self._action_dim = action_dim
        self._latent_dist = Normal(torch.Tensor([0]), torch.Tensor([1]))

        self._squash = squash

        # # TODO: WE ARE INITIALIZING LAST LAYER WEIGHTS WITH XAVIER
        # nn_pol.init.xavier_normal_(self.last_fcs.weight.data)
        # # self.last_fcs.bias.data.zero_()

    def get_action(self, obs_np, deterministic=False):
        # TODO: CHECK IF INDEX 0
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)

    def forward(
            self,
            obs,
            deterministic=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        """
        # TODO: HOW TO DETERMINISTIC???
        latent_shape = (*list(obs.shape)[:-1], self._action_dim)
        if deterministic:
            latent = torch.zeros(latent_shape)
        else:
            latent = self._latent_dist.sample(latent_shape).squeeze(-1)

        if ptu.gpu_enabled():
            latent = latent.cuda()

        h = torch.cat([obs, latent], dim=-1)
        # print('--- INPUT ---')
        # print(torch.cat([obs, latent], dim=-1)[:5, :])
        for i, fc in enumerate(self.fcs):
            # h = self.hidden_activation(fc(h))
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)

        action = self.last_fc(h)

        if self._squash:
            action = torch.tanh(action)

        return action
