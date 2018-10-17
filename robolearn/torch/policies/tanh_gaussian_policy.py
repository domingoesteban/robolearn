"""
This code is based on: https://github.com/vitchyr/rlkit
"""
import numpy as np
import torch
from torch import nn as nn

import robolearn.torch.pytorch_util as ptu
from robolearn.torch.nn import Mlp
from robolearn.policies.base import ExplorationPolicy
from robolearn.torch.distributions import TanhNormal
from robolearn.torch.distributions import TanhMultivariateNormal

LOG_SIG_MAX = 0.0  # 2
LOG_SIG_MIN = -3.0  # 20


class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, meanTwoGoalEnv, log_std, log_prob = policy(obs, return_log_prob=True)
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
            hidden_sizes,
            std=None,
            hidden_w_init='xavier_normal',
            hidden_b_init_val=0,
            output_w_init='xavier_normal',
            output_b_init_val=0,
            reparameterize=True,
            **kwargs
    ):
        """

        Args:
            obs_dim:
            action_dim:
            hidden_sizes:
            std:
            hidden_w_init:
            hidden_b_init_val:
            output_w_init:
            output_b_init_val:
            reparameterize: If True, gradients will flow directly through
                the action samples.
            **kwargs:
        """
        self.save_init_params(locals())
        super(TanhGaussianPolicy, self).__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            hidden_w_init=hidden_w_init,
            hidden_b_init_val=hidden_b_init_val,
            output_w_init=output_w_init,
            output_b_init_val=output_b_init_val,
            **kwargs
        )
        ExplorationPolicy.__init__(self, action_dim)

        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            ptu.layer_init_xavier_normal(layer=self.last_fc_log_std,
                                         activation='linear',
                                         b=output_b_init_val)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

        self._reparameterize = reparameterize

    def get_action(self, obs_np, **kwargs):
        """
        """
        actions, info_dict = self.get_actions(obs_np[None], **kwargs)
        for key, val in info_dict.items():
            if isinstance(val, np.ndarray):
                info_dict[key] = val[0, :]

        # Get [0, :] vals (Because it has dimension 1xdA)
        return actions[0, :], info_dict

    def get_actions(self, obs_np, **kwargs):
        """
        """
        return self.eval_np(obs_np, **kwargs)

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

        pre_tanh_value = None
        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None

        if deterministic:
            action = torch.tanh(mean)
        else:
            # Using this distribution instead of TanhMultivariateNormal
            # because it has Diagonal Covariance.
            # Then, a collection of n independent Gaussian r.v.
            tanh_normal = TanhNormal(mean, std)

            # # It is the Lower-triangular factor of covariance because it is
            # # Diagonal Covariance
            # scale_trils = torch.stack([torch.diag(m) for m in std])
            # tanh_normal = TanhMultivariateNormal(mean, scale_tril=scale_trils)

            if self._reparameterize:
                action, pre_tanh_value = tanh_normal.rsample(
                    return_pretanh_value=True
                )
            else:
                action, pre_tanh_value = tanh_normal.sample(
                    return_pretanh_value=True
                )

            if return_log_prob:
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )

                # THE FOLLOWING ONLY WITH TanhNormal
                log_prob = log_prob.sum(dim=-1, keepdim=True)

        info_dict = dict(
            mean=mean,
            log_std=log_std,
            log_prob=log_prob,
            expected_log_prob=expected_log_prob,
            std=std,
            mean_action_log_prob=mean_action_log_prob,
            pre_tanh_value=pre_tanh_value,
        )
        return action, info_dict

    def log_action(self, action, obs):
        """

        Args:
            action:
            obs:

        Returns:

        """
        #TODO: CHECK THIS FUNCTION
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

        # tanh_normal = TanhNormal(mean, std)
        # log_prob = torch.sum(tanh_normal.log_prob(action), dim=-1, keepdim=True)

        scale_trils = torch.stack([torch.diag(m) for m in std])
        tanh_normal = TanhMultivariateNormal(mean, scale_tril=scale_trils)
        log_prob = tanh_normal.log_prob(action).unsqueeze_(-1)

        return log_prob

        # z = (action - mean)/stds
        # return -0.5 * torch.sum(torch.mul(z, z), dim=-1, keepdim=True)

    @property
    def reparameterize(self):
        return self._reparameterize

