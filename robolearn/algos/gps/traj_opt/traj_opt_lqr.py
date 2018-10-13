import numpy as np
from numpy.linalg import LinAlgError
import scipy as sp
import logging

from robolearn.algos.gps.traj_opt.traj_opt_utils import traj_distr_kl, traj_distr_kl_alt

print_DGD_log = False

MAX_ALL_DGD = 20
DGD_MAX_ITER = 50
DGD_MAX_LS_ITER = 20
DGD_MAX_GD_ITER = 50  #500 #200

ALPHA, BETA1, BETA2, EPS = 0.005, 0.9, 0.999, 1e-8  # Adam parameters


class TrajOptLQR(object):
    """ LQR trajectory optimization """
    def __init__(self,
                 cons_per_step=False,
                 use_prev_distr=False,
                 update_in_bwd_pass=True,
                 min_eta=1e-8,
                 max_eta=1e16,
                 del0=1e-4,
                 ):

        self.cons_per_step = cons_per_step
        self._use_prev_distr = use_prev_distr
        self._update_in_bwd_pass = update_in_bwd_pass
        self._min_eta = min_eta
        self._max_eta = max_eta
        self._del0 = del0

        if not self._use_prev_distr:
            self._traj_distr_kl_fcn = traj_distr_kl_alt
        else:
            self._traj_distr_kl_fcn = traj_distr_kl

        self.logger = logging.getLogger(__name__)

    # TODO - Add arg and return spec on this function.
    def update(self, m, algorithm, prev_type='nn_pol'):
        """
        Run dual gradient descent to optimize trajectories.
        It returns (optimized) new trajectory and eta.
        :param m: Condition number.
        :param algorithm: GPS algorithm to get info
        :param
        :return: traj_distr, eta
        """
        T = algorithm.T

        # Get current eta
        eta = algorithm.cur[m].eta
        if self.cons_per_step and type(eta) in (int, float):
            eta = np.ones(T) * eta

        # Get current step_mult and traj_info
        step_mult = algorithm.cur[m].step_mult
        traj_info = algorithm.cur[m].traj_info

        # Get the trajectory distribution that is going to be used as constraint
        # gps_algo = type(algorithm).__name__
        # if gps_algo == 'MDGPS':
        if prev_type == 'nn_pol':
            # For MDGPS, constrain to previous NN linearization
            prev_traj_distr = algorithm.cur[m].pol_info.traj_distr()
        else:
            # For BADMM/trajopt, constrain to previous LG controller
            prev_traj_distr = algorithm.cur[m].traj_distr

        # Set KL-divergence step size (epsilon) using step multiplier.
        kl_step = algorithm.base_kl_step * step_mult
        if not self.cons_per_step:
            kl_step *= T

        # We assume at min_eta, kl_div > kl_step, opposite for max_eta.
        if not self.cons_per_step:
            min_eta = self._min_eta
            max_eta = self._max_eta
            print('_'*60)
            print("Running DGD for trajectory(condition) %d, eta: %f"
                  % (m, eta))
        else:
            min_eta = np.ones(T) * self._min_eta
            max_eta = np.ones(T) * self._max_eta
            self.logger.debug("Running DGD for trajectory %d, avg eta: %f",
                              m, np.mean(eta[:-1]))

        max_itr = (DGD_MAX_LS_ITER if self.cons_per_step else DGD_MAX_ITER)  # Less iterations if cons_per_step=True

        for itr in range(max_itr):
            if not self.cons_per_step:
                self.logger.debug("Iteration %d, bracket: (%.2e , %.2e , %.2e)",
                                  itr, min_eta, eta, max_eta)

            # Run fwd/bwd pass, note that eta may be updated.
            # Compute KL divergence constraint violation.
            traj_distr, eta = self.backward(prev_traj_distr, traj_info, eta,
                                            algorithm, m)

            # Compute KL divergence constraint violation.
            if not self._use_prev_distr:
                traj_distr_to_check = traj_distr
            else:
                traj_distr_to_check = prev_traj_distr

            mu_to_check, sigma_to_check = self.forward(traj_distr_to_check,
                                                       traj_info)

            kl_div = self._traj_distr_kl_fcn(mu_to_check, sigma_to_check,
                                             traj_distr, prev_traj_distr,
                                             tot=(not self.cons_per_step))

            con = kl_div - kl_step

            # Convergence check - constraint satisfaction.
            if self._conv_check(con, kl_step):
                if not self.cons_per_step:
                    self.logger.debug("KL: %f / %f, converged iteration %d",
                                      kl_div, kl_step, itr)
                else:
                    self.logger.debug("KL: %f / %f, converged iteration %d",
                                      np.mean(kl_div[:-1]),
                                      np.mean(kl_step[:-1]), itr)
                break

            if not self.cons_per_step:
                # Choose new eta (bisect bracket or multiply by constant)
                if con < 0:  # Eta was too big.
                    max_eta = eta
                    geom = np.sqrt(min_eta*max_eta)  # Geometric mean.
                    new_eta = max(geom, 0.1*max_eta)
                    self.logger.debug("KL: %f / %f, eta too big, new eta: %f",
                                      kl_div, kl_step, new_eta)
                else:  # Eta was too small.
                    min_eta = eta
                    geom = np.sqrt(min_eta*max_eta)  # Geometric mean.
                    new_eta = min(geom, 10.0*min_eta)
                    self.logger.debug("KL: %f / %f, eta too small, new eta: %f",
                                      kl_div, kl_step, new_eta)

                # Logarithmic mean: log_mean(x,y) = (y - x)/(log(y) - log(x))
                eta = new_eta
            else:
                for t in range(T):
                    if con[t] < 0:  # Eta was too big.
                        max_eta[t] = eta[t]
                        geom = np.sqrt(min_eta[t]*max_eta[t])  # Geometric mean.
                        eta[t] = max(geom, 0.1*max_eta[t])
                    else:
                        min_eta[t] = eta[t]
                        geom = np.sqrt(min_eta[t]*max_eta[t])  # Geometric mean.
                        eta[t] = min(geom, 10.0*min_eta[t])
                if itr % 10 == 0:
                    self.logger.debug("avg KL: %f / %f, avg new eta: %f",
                                      np.mean(kl_div[:-1]),
                                      np.mean(kl_step[:-1]), np.mean(eta[:-1]))

        # ADAM Gradient Descent
        if self.cons_per_step and not self._conv_check(con, kl_step):
            m_b, v_b = np.zeros(T-1), np.zeros(T-1)

            for itr in range(DGD_MAX_GD_ITER):
                traj_distr, eta = self.backward(prev_traj_distr, traj_info, eta,
                                                algorithm, m)

                mu_to_check, sigma_to_check = self.forward(traj_distr_to_check,
                                                           traj_info)

                kl_div = self._traj_distr_kl_fcn(mu_to_check, sigma_to_check,
                                                 traj_distr, prev_traj_distr,
                                                 tot=(not self.cons_per_step))

                con = kl_div - kl_step
                if self._conv_check(con, kl_step):
                    self.logger.debug("KL: %f / %f, converged iteration %d",
                                      np.mean(kl_div[:-1]),
                                      np.mean(kl_step[:-1]), itr)
                    break

                m_b = (BETA1 * m_b + (1-BETA1) * con[:-1])
                m_u = m_b / (1 - BETA1 ** (itr+1))
                v_b = (BETA2 * v_b + (1-BETA2) * np.square(con[:-1]))
                v_u = v_b / (1 - BETA2 ** (itr+1))
                eta[:-1] = np.minimum(
                    np.maximum(eta[:-1] + ALPHA * m_u / (np.sqrt(v_u) + EPS),
                               self._min_eta),
                    self._max_eta)

                if itr % 10 == 0:
                    self.logger.debug("avg KL: %f / %f, avg new eta: %f",
                                      np.mean(kl_div[:-1]),
                                      np.mean(kl_step[:-1]), np.mean(eta[:-1]))

        if np.mean(kl_div) > np.mean(kl_step) \
                and not self._conv_check(con, kl_step):
            self.logger.warning("Final KL divergence after DGD convergence is "
                                "too high (eta: %f)." % eta)
        else:
            self.logger.info("Final eta: %f." % eta)

        return traj_distr, eta

    def estimate_cost(self, traj_distr, traj_info):
        """ Compute Laplace approximation to expected cost. """
        # Constants.
        T = traj_distr.T

        # Perform forward pass (note that we repeat this here, because
        # traj_info may have different dynamics from the ones that were
        # used to compute the distribution already saved in traj).
        mu, sigma = self.forward(traj_distr, traj_info)

        # Compute cost.
        predicted_cost = np.zeros(T)
        for t in range(T):
            predicted_cost[t] = traj_info.cc[t] + \
                                0.5 * np.sum(sigma[t, :, :] * traj_info.Cm[t, :, :]) + \
                                0.5 * mu[t, :].T.dot(traj_info.Cm[t, :, :]).dot(mu[t, :]) + \
                                mu[t, :].T.dot(traj_info.cv[t, :])
        return predicted_cost

    @staticmethod
    def forward(traj_distr, traj_info):
        """
        Perform LQR forward pass. Computes state-action marginals from dynamics and policy.
        Args:
            traj_distr: A linear Gaussian policy object.
            traj_info: A TrajectoryInfo object.
        Returns:
            mu: A T x dX mean action vector.
            sigma: A T x dX x dX covariance matrix.
        """
        # Compute state-action marginals from specified conditional
        # parameters and current traj_info.
        T = traj_distr.T
        dU = traj_distr.dU
        dX = traj_distr.dX

        # Constants.
        idx_x = slice(dX)

        # Allocate space.
        sigma = np.zeros((T, dX+dU, dX+dU))
        mu = np.zeros((T, dX+dU))

        # Pull out dynamics.
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv
        dyn_covar = traj_info.dynamics.dyn_covar

        # Set initial covariance (initial mu is always zero).
        sigma[0, idx_x, idx_x] = traj_info.x0sigma
        mu[0, idx_x] = traj_info.x0mu

        for t in range(T):
            sigma[t, :, :] = np.vstack([np.hstack([
                sigma[t, idx_x, idx_x],
                sigma[t, idx_x, idx_x].dot(traj_distr.K[t, :, :].T)
            ]),
                np.hstack([
                    traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]),
                    traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]).dot(
                        traj_distr.K[t, :, :].T
                    ) + traj_distr.pol_covar[t, :, :]
                ])
            ])
            mu[t, :] = np.hstack([mu[t, idx_x],
                                  traj_distr.K[t, :, :].dot(mu[t, idx_x]) + traj_distr.k[t, :]])
            if t < T - 1:
                sigma[t+1, idx_x, idx_x] = Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T) + dyn_covar[t, :, :]
                mu[t+1, idx_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]
        return mu, sigma

    def backward(self, prev_traj_distr, traj_info, eta, algorithm, m):
        """
        Perform LQR backward pass. This computes a new linear Gaussian policy
        object.
        Args:
            prev_traj_distr: A linear Gaussian policy object from previous iteration.
            traj_info: A TrajectoryInfo object.
            eta: Dual variable.
            algorithm: Algorithm object needed to compute costs.
            m: Condition number.
        Returns:
            traj_distr: A new linear Gaussian policy.
            new_eta: The updated dual variable. Updates happen if the Q-function is not PD.
        """
        # Constants.
        T = prev_traj_distr.T
        dU = prev_traj_distr.dU
        dX = prev_traj_distr.dX

        if self._update_in_bwd_pass:
            traj_distr = prev_traj_distr.nans_like()
        else:
            traj_distr = prev_traj_distr.copy()

        compute_cost_fcn = algorithm.compute_traj_cost

        # Store pol_wt if necessary
        gps_algo = type(algorithm).__name__
        if gps_algo.lower() == 'badmm':
            pol_wt = algorithm.cur[m].pol_info.pol_wt

        idx_x = slice(dX)
        idx_u = slice(dX, dX+dU)

        # Pull out dynamics.
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv

        # Non-SPD correction terms.
        del_ = self._del0
        if self.cons_per_step:
            del_ = np.ones(T) * del_
        eta0 = eta

        # Run dynamic programming.
        fail = True
        while fail:
            fail = False  # Flip to true on non-symmetric PD.

            # Allocate.
            Vxx = np.zeros((T, dX, dX))
            Vx = np.zeros((T, dX))
            Qtt = np.zeros((T, dX+dU, dX+dU))
            Qt = np.zeros((T, dX+dU))

            if not self._update_in_bwd_pass:
                new_K, new_k = np.zeros((T, dU, dX)), np.zeros((T, dU))
                new_pS = np.zeros((T, dU, dU))
                new_ipS, new_cpS = np.zeros((T, dU, dU)), np.zeros((T, dU, dU))

            # Compute cost defined in RL algorithm
            fCm, fcv = compute_cost_fcn(m, eta, augment=(not self.cons_per_step))

            # Get Maximum Var
            max_precission = 1./traj_distr.max_var

            # Compute state-action-state function at each time step.
            for t in range(T - 1, -1, -1):
                # Add in the cost.
                Qtt[t] = fCm[t, :, :]  # (X+U) x (X+U)
                Qt[t] = fcv[t, :]  # (X+U) x 1

                # Add in the value function from the next time step.
                if t < T - 1:
                    gps_algo = type(algorithm).__name__
                    if gps_algo.lower() == 'badmm':
                        multiplier = (pol_wt[t+1] + eta)/(pol_wt[t] + eta)
                    else:
                        multiplier = 1.0

                    Qtt[t] += multiplier * Fm[t, :, :].T.dot(Vxx[t+1, :, :]).dot(Fm[t, :, :])
                    Qt[t] += multiplier * Fm[t, :, :].T.dot(Vx[t+1, :] + Vxx[t+1, :, :].dot(fv[t, :]))

                # Symmetrize quadratic component.
                Qtt[t] = 0.5 * (Qtt[t] + Qtt[t].T)

                if not self.cons_per_step:
                    inv_term = np.clip(Qtt[t, idx_u, idx_u],
                                       -max_precission, max_precission)
                    k_term = Qt[t, idx_u]
                    K_term = Qtt[t, idx_u, idx_x]
                else:
                    inv_term = (1.0 / eta[t]) * Qtt[t, idx_u, idx_u] + prev_traj_distr.inv_pol_covar[t]
                    k_term = (1.0 / eta[t]) * Qt[t, idx_u] - prev_traj_distr.inv_pol_covar[t].dot(prev_traj_distr.k[t])
                    K_term = (1.0 / eta[t]) * Qtt[t, idx_u, idx_x] - \
                             prev_traj_distr.inv_pol_covar[t].dot(prev_traj_distr.K[t])

                # Compute Cholesky decomposition of Q function action component.
                try:
                    U = sp.linalg.cholesky(inv_term)
                    L = U.T
                except LinAlgError as e:
                    # Error thrown when Qtt[idx_u, idx_u] is not symmetric positive definite.
                    self.logger.debug('LinAlgError: %s', e)
                    fail = t if self.cons_per_step else True
                    break

                if self._update_in_bwd_pass:
                    # Store conditional covariance, inverse, and Cholesky.
                    traj_distr.inv_pol_covar[t, :, :] = inv_term
                    traj_distr.pol_covar[t, :, :] = sp.linalg.solve_triangular(
                        U, sp.linalg.solve_triangular(L, np.eye(dU), lower=True)
                    )
                    traj_distr.chol_pol_covar[t, :, :] = sp.linalg.cholesky(
                        traj_distr.pol_covar[t, :, :]
                    )

                    # Compute mean terms.
                    traj_distr.k[t, :] = -sp.linalg.solve_triangular(
                        U, sp.linalg.solve_triangular(L, k_term, lower=True)
                    )
                    traj_distr.K[t, :, :] = -sp.linalg.solve_triangular(
                        U, sp.linalg.solve_triangular(L, K_term, lower=True)
                    )
                else:
                    # Store conditional covariance, inverse, and Cholesky.
                    new_ipS[t, :, :] = inv_term
                    new_pS[t, :, :] = sp.linalg.solve_triangular(
                        U, sp.linalg.solve_triangular(L, np.eye(dU), lower=True)
                    )
                    new_cpS[t, :, :] = sp.linalg.cholesky(
                        new_pS[t, :, :]
                    )

                    # Compute mean terms.
                    new_k[t, :] = -sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L, k_term, lower=True))
                    new_K[t, :, :] = -sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L, K_term, lower=True))

                # Compute value function.
                if (self.cons_per_step or
                        not self._update_in_bwd_pass):
                    Vxx[t, :, :] = Qtt[t, idx_x, idx_x] + \
                                   traj_distr.K[t].T.dot(Qtt[t, idx_u, idx_u]).dot(traj_distr.K[t]) + \
                                   (2 * Qtt[t, idx_x, idx_u]).dot(traj_distr.K[t])
                    Vx[t, :] = Qt[t, idx_x].T + \
                               Qt[t, idx_u].T.dot(traj_distr.K[t]) + \
                               traj_distr.k[t].T.dot(Qtt[t, idx_u, idx_u]).dot(traj_distr.K[t]) + \
                               Qtt[t, idx_x, idx_u].dot(traj_distr.k[t])
                else:
                    Vxx[t, :, :] = Qtt[t, idx_x, idx_x] + \
                                   Qtt[t, idx_x, idx_u].dot(traj_distr.K[t, :, :])
                    Vx[t, :] = Qt[t, idx_x] + \
                               Qtt[t, idx_x, idx_u].dot(traj_distr.k[t, :])
                Vxx[t, :, :] = 0.5 * (Vxx[t, :, :] + Vxx[t, :, :].T)

            if not self._update_in_bwd_pass:
                traj_distr.K, traj_distr.k = new_K, new_k
                traj_distr.pol_covar = new_pS
                traj_distr.inv_pol_covar = new_ipS
                traj_distr.chol_pol_covar = new_cpS

            # Increment eta on non-SPD Q-function.
            if fail:
                if not self.cons_per_step:
                    old_eta = eta
                    eta = eta0 + del_
                    self.logger.debug('Increasing eta: %f -> %f', old_eta, eta)
                    del_ *= 2  # Increase del_ exponentially on failure.
                else:
                    old_eta = eta[fail]
                    eta[fail] = eta0[fail] + del_[fail]
                    self.logger.debug('Increasing eta %d: %f -> %f',
                                      fail, old_eta, eta[fail])
                    del_[fail] *= 2  # Increase del_ exponentially on failure.
                if self.cons_per_step:
                    fail_check = (eta[fail] >= 1e16)
                else:
                    fail_check = (eta >= 1e16)
                if fail_check:
                    if np.any(np.isnan(Fm)) or np.any(np.isnan(fv)):
                        raise ValueError('NaNs encountered in dynamics!')
                    raise ValueError('Failed to find PD solution even for very '
                                     'large eta (check that dynamics and cost '
                                     'are reasonably well conditioned)!')
        return traj_distr, eta

    def _conv_check(self, con, kl_step):
        """Function that checks whether dual gradient descent has converged."""
        if self.cons_per_step:
            return all([abs(con[t]) < (0.1*kl_step[t]) for t in range(con.size)])
        return abs(con) < 0.1 * kl_step
