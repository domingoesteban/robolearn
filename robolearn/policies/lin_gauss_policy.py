"""
Adapted by robolearn team.
Based on lin_gaus_policy from github.com:cbfinn/gps.git
"""

import numpy as np

from robolearn.policies.policy import Policy
from robolearn.utils.general import check_shape

class LinearGaussianPolicy(Policy):
    """
    Time-varying linear Gaussian policy.
    U = K*x + k + noise, where noise ~ N(0, chol_pol_covar)
    """
    def __init__(self, K, k, pol_covar, chol_pol_covar, inv_pol_covar):
        """
        :param K: dim=(T,dU,dX)
        :param k: dim=(T,dU)
        :param pol_covar: dim=(T,dU,dU)
        :param chol_pol_covar: dim=(T,dU,dU)
        :param inv_pol_covar:dim=(T,dU,dU)
        """
        Policy.__init__(self)
        #super(LinearGaussianPolicy, self).__init__()

        # Assume K has the correct shape, and make sure others match.
        self.T = K.shape[0]
        self.dU = K.shape[1]
        self.dX = K.shape[2]

        check_shape(k, (self.T, self.dU))
        check_shape(pol_covar, (self.T, self.dU, self.dU))
        check_shape(chol_pol_covar, (self.T, self.dU, self.dU))
        check_shape(inv_pol_covar, (self.T, self.dU, self.dU))

        self.K = K
        self.k = k
        self.pol_covar = pol_covar
        self.chol_pol_covar = chol_pol_covar
        self.inv_pol_covar = inv_pol_covar

    def eval(self, x, obs, t, noise=None):
        """
        Return an action for a state.
        :param x: State vector. 
        :param obs: Observation vector.
        :param t: Time step.
        :param noise: Action noise. This will be scaled by the variance.
        :return: Action u.
        """
        u = self.K[t].dot(x) + self.k[t]
        u += self.chol_pol_covar[t].T.dot(noise)
        return u

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

    def get_params(self):
        """
        Get the parameters of the policy 
        :return: A dictionary with parameters
        """
        return {'K': self.K, 'k': self.k, 'pol_covar': self.pol_covar,
                'chol_pol_covar': self.chol_pol_covar, 'inv_pol_covar': self.inv_pol_covar}
