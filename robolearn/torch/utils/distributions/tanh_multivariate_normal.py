"""
This code is based on TanhNormal class.
https://github.com/vitchyr/rlkit
"""
import torch
from torch.distributions import Distribution
from torch.distributions.utils import lazy_property

# from torch.distributions import MultivariateNormal
from robolearn.torch.utils.distributions.multivariate_normal import MultivariateNormal


class TanhMultivariateNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, *args, **kwargs):
        """

        Args:
            mvn_mean (Tensor): Mean of the normal distribution
            mvn_covar (Tensor): Std of the normal distribution
            epsilon (Double): Numerical stability epsilon when computing
                log-prob.
        """
        epsilon = kwargs.pop('epsilon', 1.e-6)
        super(TanhMultivariateNormal, self).__init__()

        self._multivariate_normal = MultivariateNormal(*args, **kwargs)
        self._epsilon = epsilon

    @property
    def mean(self):
        return self._multivariate_normal.mean

    @property
    def variance(self):
        return self._multivariate_normal.variance

    @property
    def stddev(self):
        return self._multivariate_normal.stddev

    @lazy_property
    def covariance_matrix(self):
        return self._multivariate_normal.covariance_matrix

    @property
    def epsilon(self):
        return self._epsilon

    def sample(self, return_pretanh_value=False):
        z = self._multivariate_normal.sample()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        z = self._multivariate_normal.rsample()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self._multivariate_normal.sample_n(n)
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

        return self._multivariate_normal.log_prob(pre_tanh_value) - \
               torch.sum(torch.log(1. - value * value + self._epsilon), dim=-1)
        # return self.normal.log_prob(pre_tanh_value) - \
        #     torch.log(1. - torch.tanh(pre_tanh_value)**2 + self._epsilon)

    def cdf(self, value, pre_tanh_value=None):
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        # TODO: MAYBE THE FOLLOWING IS NOT CORRECT - maybe apply pretanh?:
        return self._multivariate_normal.cdf(pre_tanh_value)


