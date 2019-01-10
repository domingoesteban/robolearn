"""
This code is based on: https://github.com/vitchyr/rlkit
"""
import math
import numpy as np
import torch
from torch import nn as nn

import robolearn.torch.utils.pytorch_util as ptu
from robolearn.torch.utils.nn import Mlp
from robolearn.models.policies import ExplorationPolicy
from torch.distributions import Normal
from robolearn.torch.utils.distributions import TanhNormal
from robolearn.torch.utils.distributions import TanhMultivariateNormal

# LOG_SIG_MAX = 0.0  # 2
# LOG_SIG_MIN = -3.0  # 20
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

EPS = 1e-8


class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
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
            ptu.layer_init(
                layer=self.last_fc_log_std,
                option=output_w_init,
                activation='linear',
                b=output_b_init_val
            )
        else:
            self.log_std = math.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

        self._reparameterize = reparameterize

        self._normal_dist = Normal(loc=ptu.zeros(action_dim),
                                   scale=ptu.ones(action_dim))

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

        Args:
            obs (Tensor): Observation(s)
            deterministic (bool): True for using mean. False, sample from dist.
            return_log_prob (bool):

        Returns:
            action (Tensor):
            pol_info (dict):

        """
        h = obs
        nbatch = obs.shape[0]
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)

        if self.std is None:
            log_std = self.last_fc_log_std(h)
            # # log_std option 1:
            # log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            # log_std option 2:
            log_std = torch.tanh(log_std)
            log_std = \
                LOG_SIG_MIN + 0.5 * (LOG_SIG_MAX - LOG_SIG_MIN) * (log_std + 1)

            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        pre_tanh_value = None
        log_prob = None

        if deterministic:
            action = torch.tanh(mean)
        else:
            """
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
            """

            noise = self._normal_dist.sample((nbatch,))

            pre_tanh_value = std*noise + mean

            action = torch.tanh(pre_tanh_value)

            if return_log_prob:
                log_prob = -0.5 * (((pre_tanh_value-mean)/(std+EPS))**2
                                   + 2*log_std + math.log(2*math.pi))
                log_prob = log_prob.sum(dim=-1, keepdim=True)

                log_prob -= (
                    torch.log(
                        # torch.clamp(1. - action**2, 0, 1)
                        clip_but_pass_gradient(1. - action**2, 0, 1)
                        + 1.e-6
                    )
                ).sum(dim=-1, keepdim=True)

        info_dict = dict(
            mean=mean,
            std=std,
            log_std=log_std,
            log_prob=log_prob,
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
        raise NotImplementedError
        # #TODO: CHECK THIS FUNCTION
        # h = obs
        # for i, fc in enumerate(self.fcs):
        #     h = self.hidden_activation(fc(h))
        #
        # mean = self.last_fc(h)
        #
        # if self.std is None:
        #     log_std = self.last_fc_log_std(h)
        #     log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        #     std = torch.exp(log_std)
        # else:
        #     std = self.std
        #
        # # tanh_normal = TanhNormal(mean, std)
        # # log_prob = torch.sum(tanh_normal.log_prob(action), dim=-1, keepdim=True)
        #
        # scale_trils = torch.stack([torch.diag(m) for m in std])
        # tanh_normal = TanhMultivariateNormal(mean, scale_tril=scale_trils)
        # log_prob = tanh_normal.log_prob(action).unsqueeze_(-1)
        #
        # return log_prob
        #
        # # z = (action - mean)/stds
        # # return -0.5 * torch.sum(torch.mul(z, z), dim=-1, keepdim=True)

    @property
    def reparameterize(self):
        return self._reparameterize


def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = (x > u).to(ptu.device, dtype=torch.float32)
    clip_low = (x < l).to(ptu.device, dtype=torch.float32)
    return x + ((u - x)*clip_up + (l - x)*clip_low).detach()
