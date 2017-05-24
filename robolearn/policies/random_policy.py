# Random Multivariate Normal policy

import numpy as np

from robolearn.policies.policy import Policy

class RandomPolicy(Policy):
    """
    Random Multivariate Normal policy.
    """
    def __init__(self, action_dim, action_mean=None, action_cov=None):
        Policy.__init__(self)
        #super(LinearGaussianPolicy, self).__init__()

        # Assume K has the correct shape, and make sure others match.
        self.dU = action_dim
        if action_mean is None:
            self.mean = np.zeros(action_dim)
        else:
            self.mean = action_mean

        if action_cov is None:
            self.cov = np.eye(action_dim)
        else:
            self.cov = action_cov

    def act(self, x, obs, t, noise):
        u = np.random.multivariate_normal(self.mean, self.cov)
        return u

    def nans_like(self):
        return np.empty(self.dU)*np.nan

    def get_params(self):
        """
        Get the parameters of the policy 
        :return: A dictionary with parameters
        """
        return {'mean': self.mean, 'cov': self.cov}

