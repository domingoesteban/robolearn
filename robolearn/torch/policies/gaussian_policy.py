"""
This code is based on: https://github.com/vitchyr/rlkit
"""
import numpy as np
import torch
from torch import nn as nn

import robolearn.torch.utils.pytorch_util as ptu
from robolearn.torch.utils.nn import Mlp
from robolearn.models.policies import ExplorationPolicy
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class GaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = GaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = mean.
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
            hidden_b_init_val=0.1,
            output_w_init='xavier_normal',
            output_b_init_val=0.1,
            **kwargs
    ):
        self.save_init_params(locals())
        super(GaussianPolicy, self).__init__(
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

            if output_w_init == 'xavier_normal':
                ptu.layer_init(layer=self.last_fc_log_std,
                               activation='linear',
                               b=output_b_init_val)
            elif output_w_init == 'xavier_normal_0.1':
                ptu.layer_init(layer=self.last_fc_log_std,
                               activation='0.1',
                               b=output_b_init_val)
            elif output_w_init == 'xavier_normal_0.01':
                ptu.layer_init(layer=self.last_fc_log_std,
                               activation='0.01',
                               b=output_b_init_val)
            elif output_w_init == 'xavier_normal_0.001':
                ptu.layer_init(layer=self.last_fc_log_std,
                               activation='0.001',
                               b=output_b_init_val)
            elif output_w_init == 'xavier_normal_0.003':
                ptu.layer_init(layer=self.last_fc_log_std,
                               activation='0.003',
                               b=output_b_init_val)
            else:
                raise ValueError("Wrong init value:%s" % output_w_init)

        else:
            self.log_std = torch.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        """
        """
        actions, info_dict = self.get_actions(obs_np[None],
                                              deterministic=deterministic)
        for key, val in info_dict.items():
            if isinstance(val, np.ndarray):
                info_dict[key] = val[0, :]

        # Get [0, :] vals (Because it has dimension 1xdA)
        return actions[0, :], info_dict

    def get_actions(self, obs_np, deterministic=False):
        """
        """
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

        if deterministic:
            action = mean
        else:
            normal = Normal(mean, std)
            action = normal.rsample()
            if return_log_prob:
                log_prob = normal.log_prob(action,)
                log_prob = log_prob.sum(dim=-1, keepdim=True)

        info_dict = dict(
            mean=mean,
            log_std=log_std,
            log_prob=log_prob,
            expected_log_prob=expected_log_prob,
            std=std,
            mean_action_log_prob=mean_action_log_prob,
        )
        return action, info_dict

    def log_action(self, action, obs):
        """

        Args:
            action:
            obs:

        Returns:

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

        normal = Normal(mean, std)
        log_prob = torch.sum(normal.log_prob(action), dim=-1, keepdim=True)
        return log_prob

        # z = (action - mean)/stds
        # return -0.5 * torch.sum(torch.mul(z, z), dim=-1, keepdim=True)

