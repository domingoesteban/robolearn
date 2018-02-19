"""
This file defines linear regression with an arbitrary prior.
Author: C. Finn et al. Code in https://github.com/cbfinn/gps
"""
import numpy as np

from robolearn.utils.dynamics.dynamics import Dynamics
from robolearn.algos.gps.gps_utils import gauss_fit_joint_prior
from robolearn.utils.print_utils import ProgressBar


class DynamicsLRPrior(Dynamics):
    """ Dynamics with linear regression, with arbitrary prior. """
    def __init__(self, hyperparams):
        Dynamics.__init__(self, hyperparams)
        self.Fm = None
        self.fv = None
        self.dyn_covar = None
        self.prior = self._hyperparams['prior']['type'](self._hyperparams['prior'])

    def update_prior(self, samples, state_idx=None, action_idx=None):
        """ Update dynamics prior. """
        if state_idx is None:
            X = samples.get_states()
        else:
            X = samples.get_states()[:, :, state_idx]
        if action_idx is None:
            U = samples.get_actions()
        else:
            U = samples.get_actions()[:, :, action_idx]
        self.prior.update(X, U)

    def get_prior(self):
        """ Return the dynamics prior. """
        return self.prior

    #TODO: Merge this with DynamicsLR.fit - lots of duplicated code.
    def fit(self, X, U):
        """ Fit dynamics. """
        N, T, dX = X.shape
        dU = U.shape[2]

        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        self.Fm = np.zeros([T, dX, dX+dU])
        self.fv = np.zeros([T, dX])
        self.dyn_covar = np.zeros([T, dX, dX])

        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)
        # Fit dynamics with least squares regression.
        dwts = (1.0 / N) * np.ones(N)

        #fit_bar = ProgressBar(T-1)
        for t in range(T - 1):
            Ys = np.c_[X[:, t, :], U[:, t, :], X[:, t+1, :]]
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.prior.eval(dX, dU, Ys)
            sig_reg = np.zeros((dX+dU+dX, dX+dU+dX))
            sig_reg[it, it] = self._hyperparams['regularization']
            Fm, fv, dyn_covar = gauss_fit_joint_prior(Ys,
                        mu0, Phi, mm, n0, dwts, dX+dU, dX, sig_reg)
            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv
            self.dyn_covar[t, :, :] = dyn_covar
            #fit_bar.update(t)
        #fit_bar.end()
        return self.Fm, self.fv, self.dyn_covar
