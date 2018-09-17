"""
iLQR
Based on C. Finn GPS
"""
import numpy as np
import scipy as sp
from numpy.linalg import LinAlgError
import copy

from .base import TrajOpt


DEFAULT_HYPERPARAMS = dict(
    # Dual variable updates for non-PD Q-function.
    del0=1e-4,
    eta_error_threshold=1e16,
    min_eta=1e-8,
    max_eta=1e16,
    cons_per_step=False,  # Whether or not to enforce separate KL constraints at each time step.
    use_prev_distr=False,  # Whether or not to measure expected KL under the previous traj distr.
    update_in_bwd_pass=True, # Whether or not to update the TVLG controller during the bwd pass.
)


class iLQR(object):
    """ iterative LQR """
    def __init__(self, horizon, state_dim, action_dim, cost_fcn, delta=1e-4):

        self._T = horizon
        self._dX = state_dim
        self._dU = action_dim
        self._cost_fcn = cost_fcn
        self._delta = delta

    def update(self, traj_distr, linear_dynamics, cost_fcn, x0mu, x0sigma):

        new_mu, new_sigma = self.forward(traj_distr, linear_dynamics,
                                         x0mu, x0sigma)

    def forward(self, traj_distr, linear_dynamics, x0mu, x0sigma):
        """
        Perform LQR forward pass. Computes state-action marginals from dynamics
        and policy.
        Args:
            traj_distr:

        Returns:
            mu: T x dX mean state-action vector
            sigma: T x dX x dX state-action covariance matrix

        """

        # Allocate space
        sigma = np.zeros((self._T, self._dX+self._dU, self._dX+self._dU))
        mu = np.zeros((self._T, self._dX+self._dU))

        # Get dynamics
        Fm = linear_dynamics.Fm
        fv = linear_dynamics.fv
        dyn_covar = linear_dynamics.dyn_covar

        # Get TVLGC params
        K = traj_distr.K
        k = traj_distr.k
        pol_covar = traj_distr.pol_covar

        # Indexes
        idx_x = slice(self._dX)

        # Set initial covariance (initial mu is always zero)
        sigma[0, idx_x, idx_x] = x0sigma
        mu[0, idx_x] = x0mu

        for t in range(self._T):
            # Update Covariance
            sigma[t, :, :] = np.vstack((
                # dX x dX+dU
                np.hstack((
                    sigma[t, idx_x, idx_x],
                    sigma[t, idx_x, idx_x].dot(K[t, :, :].T),
                )),
                # dU x dX+dU
                np.hstack((
                    K[t, :, :].dot(sigma[t, idx_x, idx_x]),
                    K[t, :, :].dot(sigma[t, idx_x, idx_x]).dot(K[t, :, :].T) +
                    pol_covar[t, :, :],
                ))
            ))
            # Update Action mean
            mu[t, :] = np.hstack((
                mu[t, idx_x],
                K[t, :, :].dot(mu[t, idx_x]) + k[t, :]
            ))

            # Forward Dynamics
            if t < self._T - 1:
                sigma[t+1, idx_x, idx_x] = \
                    Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T) + \
                    dyn_covar[t, :, :]
                mu[t+1, idx_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]

        return mu, sigma

    def backward(self, prev_traj_distr, linear_dynamics, costs):
        traj_distr = prev_traj_distr.copy()
        K = traj_distr.K
        k = traj_distr.k
        inv_pol_covar = traj_distr.inv_pol_covar
        pol_covar = traj_distr.pol_covar
        chol_pol_covar = traj_distr.chol_pol_covar

        # Indexes
        idx_x = slice(self._dX)
        idx_u = slice(self._dX, self._dX + self._dU)

        # Get Dynamics Matrix and Vector
        Fm = linear_dynamics.Fm
        fv = linear_dynamics.fv

        # Non-SPD correction terms
        delta = self._delta

        # Solve triangular Function
        solve_triangular = sp.linalg.solve_triangular

        # ################### #
        # Dynamic Programming #
        # ################### #

        # Allocate
        Vxx = np.zeros((self._T, self._dX, self._dX))
        Vx = np.zeros((self._T, self._dX))
        Qtt = np.zeros((self._T, self._dX+self._dU, self._dX+self._dU))
        Qt = np.zeros((self._T, self._dX+self._dU))

        # Gradient and Hessian of Reward
        fCm = costs.fCm
        fcv = costs.fcv

        # Compute state-action-state function at each time step
        for t in range(self._T - 1, -1, -1):
            # Add in the cost gradient and Hessian respectively
            Qtt[t] = fCm[t, :, :]  # (X+U) x (X+U)
            Qt[t] = fcv[t, :, :]  # (X+U) x 1

            # Add in the state value function from the next time step
            if t < self._T - 1:
                Qtt[t] += Fm[t, :, :].T.dot(Vxx[t+1, :, :]).dot(Fm[t, :, :])
                Qt[t] += Fm[t, :, :].T.dot(
                    Vx[t+1, :] + Vxx[t+1, :, :].dot(fv[t, :])
                )
            # Symmetrize quadratic component
            Qtt[t] = 0.5 * (Qtt[t] + Qtt[t].T)

            inv_term = Qtt[t, idx_u, idx_u]  # Quu
            k_term = Qt[t, idx_u]  # Qu
            K_term = Qtt[t, idx_u, idx_u]

            # Compute Cholesky decomposition of Q-value action component
            U = sp.linalg.cholesky(inv_term)  # chol(Quu)
            L = U.T

            # Conditional covariance, inverse, and Cholesky decomposition
            inv_pol_covar[t, :, :] = inv_term
            pol_covar[t, :, :] = solve_triangular(
                U, solve_triangular(L, np.eye(self._dU), lower=True)
            )
            chol_pol_covar[t, :, :] = sp.linalg.cholesky(pol_covar[t, :, :])

            # Compute mean terms
            k[t] = -solve_triangular(
                U, solve_triangular(L, k_term, lower=True)
            )
            K[t] = -solve_triangular(
                U, solve_triangular(L, K_term, lower=True)
            )

            # State value gradient
            Vx[t] = Qt[t, idx_x] + Qtt[t, idx_x, idx_u].dot(k[t])
            # State value Hessian
            Vxx[t] = Qtt[t, idx_x, idx_x] + Qtt[t, idx_x, idx_u].dot(K[t])
            # Symmetrize quadratic component
            Vxx[t] = 0.5 * (Vxx[t] + Vxx[t].T)



