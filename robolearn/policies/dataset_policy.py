import numpy as np

from robolearn.policies.policy import Policy


class DataSetPolicy(Policy):
    """
    DataSet policy.
    """
    def __init__(self, action_dim, dataset):
        Policy.__init__(self)
        #super(LinearGaussianPolicy, self).__init__()

        # Assume K has the correct shape, and make sure others match.
        self.dU = action_dim
        self.dataset = dataset

    def eval(self, state=None, obs=None, t=None, noise=None):
        u = self.dataset[t, :]
        return u

    def nans_like(self):
        return np.empty(self.dU)*np.nan

    def get_params(self):
        """
        Get the parameters of the policy
        :return: A dictionary with parameters
        """
        return {'dataset': self.dataset}
