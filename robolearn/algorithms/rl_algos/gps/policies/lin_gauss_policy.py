import numpy as np
from robolearn.policies.base import Policy


class LinearGaussianPolicy(Policy):
    """
    Time-varying linear Gaussian policy.
    U = K*x + k + noise, where noise ~ N(0, chol_pol_covar)
    """
    def __init__(self, K, k, pol_covar, chol_pol_covar, inv_pol_covar,
                 max_std=0.1):
        action_dim = K.shape[0]
        Policy.__init__(self, action_dim=action_dim)

        self.T = K.shape[0]
        self.dU = K.shape[1]
        self.dX = K.shape[2]

        self.max_std = max_std
        self.max_var = max_std**2

        check_shape(k, (self.T, self.dU))
        check_shape(pol_covar, (self.T, self.dU, self.dU))
        check_shape(chol_pol_covar, (self.T, self.dU, self.dU))
        check_shape(inv_pol_covar, (self.T, self.dU, self.dU))

        self.K = K
        self.k = k
        self.pol_covar = pol_covar
        self.chol_pol_covar = chol_pol_covar
        self.inv_pol_covar = inv_pol_covar

    def get_action(self, state, t, noise):
        x = state
        u = self.K[t].dot(x) + self.k[t]

        # u += self.chol_pol_covar[t].T.dot(noise)
        u += (np.clip(self.chol_pol_covar[t],
                      -self.max_std, self.max_std)).T.dot(noise)

        return u, dict()

    def nans_like(self):
        """
        Returns:
            A new linear Gaussian policy object with the same dimensions
            but all values filled with NaNs.
        """
        policy = LinearGaussianPolicy(
            np.zeros_like(self.K), np.zeros_like(self.k),
            np.zeros_like(self.pol_covar), np.zeros_like(self.chol_pol_covar),
            np.zeros_like(self.inv_pol_covar)
        )
        policy.K.fill(np.nan)
        policy.k.fill(np.nan)
        policy.pol_covar.fill(np.nan)
        policy.chol_pol_covar.fill(np.nan)
        policy.inv_pol_covar.fill(np.nan)
        return policy


def check_shape(value, expected_shape, name=''):
    """
    Throws a ValueError if value.shape != expected_shape.
    Args:
        value: Matrix to shape check.
        expected_shape: A tuple or list of integers.
        name: An optional name to add to the exception message.
    AUTHOR: github.com:cbfinn/gps.git
    """
    if value.shape != tuple(expected_shape):
        raise ValueError('Shape mismatch %s: Expected %s, got %s' %
                         (name, str(expected_shape), str(value.shape)))
