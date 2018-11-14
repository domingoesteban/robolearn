import torch
import numpy as np
from robolearn.torch.core import PyTorchModule
from robolearn.utils.serializable import Serializable
import robolearn.torch.utils.pytorch_util as ptu


class GMM(PyTorchModule):
    def __init__(self, init_sequential=False, eigreg=False, warmstart=True):
        self._init_sequential = init_sequential
        self._eigreg = eigreg
        self._warmstart = warmstart
        self._sigma = None

        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        super(GMM, self).__init__()

    def inference(self, pts):
        """
        Evaluate dynamics prior.
        Args:
            pts: N x D array of points.

        Returns:
            mu0: mean
            Phi: covar
            m: number of
            n0: number of

        """
        # Compute posterior cluster weights
        logwts = self.clusterwts(pts)

        # Compute posterior mean and covariance
        mu0, Phi = self.moments(logwts)

        # Set hyperparameters
        m = self.N
        n0 = m - 2 - mu0.shape[0]

        # Normalize
        m = float(m) / self.N
        n0 = float(n0) / self.N

        return mu0, Phi, m, n0

    def estep(self, data):
        """
        Compute log observation probabilities under GMM.
        Args:
            data: N x D tensor of points

        Returns:
            logobs: N x K tensor of log probabilities (for each point on each
                    cluster)

        """
        # Constants
        N, D = data.shape
        K = self._sigma.shape[0]

        logobs = -0.5 * ptu.ones((N, K)) * D * np.log(2*np.pi)  # Constant
        for i in range(K):
            mu, sigma = self._mu[i], self._sigma[i]
            L = torch.potri(sigma, upper=False)  # Cholesky decomposition
            logobs[:, i] -= torch.sum(torch.log(torch.))

    @parameter
    def N(self):
        return self._N
