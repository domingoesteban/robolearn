import numpy as np
from robolearn.algos.gps.utils import gauss_fit_joint_prior


class DynamicsLRPrior(object):
    def __init__(self, prior, sigma_regu=1e-6):
        self.Fm = None
        self.fv = None
        self.dyn_covar = None
        self._prior = prior
        self._sigma_regu = sigma_regu

    def update_prior(self, X, U, state_idx=None, action_idx=None):
        """ Update dynamics prior. """
        if state_idx is not None:
            X = X[:, :, state_idx]

        if action_idx is not None:
            U = U[:, :, action_idx]

        self._prior.update(X, U)

    def get_prior(self):
        """ Return the dynamics prior. """
        return self._prior

    def fit(self, X, U):
        """ Fit dynamics. """
        N, T, dX = X.shape
        dU = U.shape[2]

        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 path")

        self.Fm = np.zeros([T, dX, dX+dU])
        self.fv = np.zeros([T, dX])
        self.dyn_covar = np.zeros([T, dX, dX])

        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)
        # Fit dynamics with least squares regression.
        dwts = (1.0 / N) * np.ones(N)

        for t in range(T - 1):
            Ys = np.c_[X[:, t, :], U[:, t, :], X[:, t+1, :]]

            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self._prior.eval(dX, dU, Ys)
            sig_reg = np.zeros((dX+dU+dX, dX+dU+dX))
            sig_reg[it, it] = self._sigma_regu

            Fm, fv, dyn_covar = gauss_fit_joint_prior(
                Ys, mu0, Phi, mm, n0, dwts, dX+dU, dX, sig_reg
            )

            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv
            self.dyn_covar[t, :, :] = dyn_covar

        return self.Fm, self.fv, self.dyn_covar

