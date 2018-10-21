"""
This code is based on TanhNormal class.
https://github.com/vitchyr/rlkit
"""
import torch
from torch.distributions import Distribution
from torch.distributions import Normal
import robolearn.torch.pytorch_util as ptu


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """

        Args:
            normal_mean (Tensor): Mean of the normal distribution
            normal_std (Tensor): Std of the normal distribution
            epsilon (Double): Numerical stability epsilon when computing
                log-prob.
        """
        super(TanhNormal, self).__init__()
        self._normal_mean = normal_mean
        self._normal_std = normal_std
        self._normal = Normal(normal_mean, normal_std)
        self._epsilon = epsilon

    @property
    def mean(self):
        return self._normal.mean

    @property
    def variance(self):
        return self._normal.variance

    @property
    def stddev(self):
        return self._normal.stddev

    @property
    def epsilon(self):
        return self._epsilon

    def sample(self, return_pretanh_value=False):
        # z = self._normal.sample()
        z = self._normal.sample().detach()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        z = self._normal.rsample()
        # z = (
        #     self._normal_mean +
        #     self._normal_std *
        #     Normal(
        #         ptu.zeros(self._normal_mean.size()),
        #         ptu.ones(self._normal_std.size()),
        #     ).sample()
        # )
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self._normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        Returns the log of the probability density function evaluated at
        `value`.

        Args:
            value (Tensor):
            pre_tanh_value (Tensor): arctan(value)

        Returns:
            log_prob (Tensor)

        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2

        return self._normal.log_prob(pre_tanh_value) - \
            torch.log(1. - value * value + self._epsilon)
        # return self.normal.log_prob(pre_tanh_value) - \
        #     torch.log(1. - torch.tanh(pre_tanh_value)**2 + self._epsilon)

    def cdf(self, value, pre_tanh_value=None):
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self._normal.cdf(pre_tanh_value)


