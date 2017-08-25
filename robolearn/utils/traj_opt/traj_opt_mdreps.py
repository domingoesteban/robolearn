"""
This file defines code for mDREPS-based trajectory optimization.
Based on: C. Finn et al. Code in https://github.com/cbfinn/gps
"""
import sys
import logging
import copy

import numpy as np
from numpy.linalg import LinAlgError
import scipy as sp

from robolearn.utils.traj_opt.traj_opt import TrajOpt
from robolearn.utils.traj_opt.config import default_traj_opt_mdreps_hyperparams
from robolearn.utils.traj_opt.traj_opt_utils import DGD_MAX_ITER, DGD_MAX_LS_ITER, DGD_MAX_GD_ITER, \
                                                    ALPHA, BETA1, BETA2, EPS, traj_distr_kl, traj_distr_kl_alt
from robolearn.utils.print_utils import ProgressBar

LOGGER = logging.getLogger(__name__)
# Logging into console AND file
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
LOGGER.addHandler(ch)


class TrajOptMDREPS(TrajOpt):
    """ MDREPS trajectory optimization """
    def __init__(self, hyperparams):
        config = copy.deepcopy(default_traj_opt_mdreps_hyperparams)
        config.update(hyperparams)

        TrajOpt.__init__(self, config)

        self.cons_per_step = config['cons_per_step']
        self._use_prev_distr = config['use_prev_distr']
        self._update_in_bwd_pass = config['update_in_bwd_pass']

    # TODO - Add arg and return spec on this function.
    def update(self, m, algorithm, a=None):
        """
        Run dual gradient descent to optimize trajectories.
        It returns (optimized) new trajectory and eta.
        :param m: Condition number.
        :param algorithm: GPS algorithm to get info
        :param a: Linear Act ID (Multiple local policies).
        :return: traj_distr, eta, omega, nu
        """
        T = algorithm.T

        # Get current eta
        if a is None:
            eta = algorithm.cur[m].eta
        else:
            eta = algorithm.cur[a][m].eta
        if self.cons_per_step and type(eta) in (int, float):
            eta = np.ones(T) * eta

        # Get current omega
        if a is None:
            omega = algorithm.cur[m].omega
        else:
            omega = algorithm.cur[a][m].omega
        if self.cons_per_step and type(omega) in (int, float):
            omega = np.ones(T) * omega

        # Get current nu
        if a is None:
            nu = algorithm.cur[m].nu
        else:
            nu = algorithm.cur[a][m].nu
        if self.cons_per_step and type(nu) in (int, float):
            nu = np.ones(T) * nu

        eta_conv = False
        nu_conv = False
        omega_conv = False
        total_gradient_descent_steps = 2
        for _ in range(total_gradient_descent_steps):
            print('_'*20)
            print(eta)
            eta, eta_conv = self._gradient_descent_eta(algorithm, a, m, eta, omega, nu)
            print(eta)
            print('^'*20)
            print(nu)
            nu, nu_conv = self._gradient_descent_nu(algorithm, a, m, eta, omega, nu)
            print(nu)
            print('^'*20)
            omega_conv = True
            # omega, omega_conv = self._gradient_descent_omega(algorithm, a, m, eta, omega, nu)

            print('eta_conv %d' % eta_conv)
            print('nu_conv %d' % nu_conv)
            print('omega_conv %d' % omega_conv)

        raw_input('probemos_PE')
        if self.cons_per_step and not self._conv_check(con, kl_step):
            print("IT DID NOT CONVERGED!!")
            m_b, v_b = np.zeros(T-1), np.zeros(T-1)

            for itr in range(DGD_MAX_GD_ITER):
                traj_distr, eta = self.backward(prev_traj_distr, traj_info, eta, algorithm, m, a)

                if not self._use_prev_distr:
                    new_mu, new_sigma = self.forward(traj_distr, traj_info)
                    kl_div = traj_distr_kl(new_mu, new_sigma, traj_distr, prev_traj_distr, tot=False)
                else:
                    prev_mu, prev_sigma = self.forward(prev_traj_distr, traj_info)
                    kl_div = traj_distr_kl_alt(prev_mu, prev_sigma, traj_distr, prev_traj_distr, tot=False)

                con = kl_div - kl_step
                if self._conv_check(con, kl_step):
                    LOGGER.debug("KL: %f / %f, converged iteration %d", np.mean(kl_div[:-1]), np.mean(kl_step[:-1]),
                                 itr)
                    break

                m_b = (BETA1 * m_b + (1-BETA1) * con[:-1])
                m_u = m_b / (1 - BETA1 ** (itr+1))
                v_b = (BETA2 * v_b + (1-BETA2) * np.square(con[:-1]))
                v_u = v_b / (1 - BETA2 ** (itr+1))
                eta[:-1] = np.minimum(np.maximum(eta[:-1] + ALPHA * m_u / (np.sqrt(v_u) + EPS),
                                                 self._hyperparams['min_eta']),
                                      self._hyperparams['max_eta'])

                if itr % 10 == 0:
                    LOGGER.debug("avg KL: %f / %f, avg new eta: %f", np.mean(kl_div[:-1]), np.mean(kl_step[:-1]),
                                 np.mean(eta[:-1]))

        if np.mean(kl_div) > np.mean(kl_step) and not self._conv_check(con, kl_step, con_good, kl_step_good, con_bad, kl_step_bad):
            LOGGER.warning("Final KL_step divergence after DGD convergence is too high.")

        if np.mean(kl_div_good) > np.mean(kl_step_good) and not self._conv_check(con, kl_step, con_good, kl_step_good, con_bad, kl_step_bad):
            LOGGER.warning("Final KL_good divergence after DGD convergence is too high.")

        if np.mean(kl_div_bad) < np.mean(kl_step_bad) and not self._conv_check(con, kl_step, con_good, kl_step_good, con_bad, kl_step_bad):
            LOGGER.warning("Final KL_bad divergence after DGD convergence is too low.")

        return traj_distr, eta, omega, nu

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

        forward_bar = ProgressBar(T, bar_title='Forward pass')
        for t in range(T):
            forward_bar.update(t)
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
        forward_bar.end()
        return mu, sigma

    def backward(self, prev_traj_distr, traj_info, eta, omega, nu, algorithm, m, a=None):
        """
        Perform LQR backward pass. This computes a new linear Gaussian policy object.
        Args:
            prev_traj_distr: A linear Gaussian policy object from previous iteration.
            traj_info: A TrajectoryInfo object.
            eta: Dual variable prev_traj, KL(p||q) <= epsilon.
            omega: Dual variable good_traj, KL(p||g) <= chi.
            nu: Dual variable bad_traj, KL(p||b) >= xi.
            algorithm: Algorithm object needed to compute costs.
            m: Condition number.
            a: Local trajectory number.
        Returns:
            traj_distr: A new linear Gaussian policy.
            new_eta: The updated dual variable. Updates happen if the Q-function is not PD.
        """
        # Constants.
        T = prev_traj_distr.T
        dU = prev_traj_distr.dU
        dX = prev_traj_distr.dX

        backward_bar = ProgressBar(T, bar_title='Backward pass')
        backward_bar_count = 0

        if self._update_in_bwd_pass:
            traj_distr = prev_traj_distr.nans_like()
        else:
            traj_distr = prev_traj_distr.copy()

        compute_cost_fcn = algorithm.compute_costs

        # Store pol_wt if necessary
        if algorithm.gps_algo.lower() == 'badmm':
            if a is None:
                pol_wt = algorithm.cur[m].pol_info.pol_wt
            else:
                pol_wt = algorithm.cur[a][m].pol_info.pol_wt

        idx_x = slice(dX)
        idx_u = slice(dX, dX+dU)

        # Pull out dynamics.
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv

        # Non-SPD correction terms.
        del_ = self._hyperparams['del0']
        if self.cons_per_step:
            del_ = np.ones(T) * del_
        eta0 = eta
        omega0 = omega
        nu0 = nu

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
            if a is None:
                fCm, fcv = compute_cost_fcn(m, eta, omega, nu, augment=(not self.cons_per_step))
            else:
                fCm, fcv = compute_cost_fcn(a, m, eta, omega, nu, augment=(not self.cons_per_step))

            # Compute state-action-state function at each time step.
            for t in range(T - 1, -1, -1):
                backward_bar_count = 0
                backward_bar.update(backward_bar_count)
                # Add in the cost.
                Qtt[t] = fCm[t, :, :]  # (X+U) x (X+U)
                Qt[t] = fcv[t, :]  # (X+U) x 1

                # Add in the value function from the next time step.
                if t < T - 1:
                    if algorithm.gps_algo.lower() == 'badmm':
                        # multiplier = (pol_wt[t+1] + eta)/(pol_wt[t] + eta)
                        raise NotImplementedError("not implemented badmm in mdreps")
                    else:
                        multiplier = 1.0

                    Qtt[t] += multiplier * Fm[t, :, :].T.dot(Vxx[t+1, :, :]).dot(Fm[t, :, :])
                    Qt[t] += multiplier * Fm[t, :, :].T.dot(Vx[t+1, :] + Vxx[t+1, :, :].dot(fv[t, :]))

                # Symmetrize quadratic component.
                Qtt[t] = 0.5 * (Qtt[t] + Qtt[t].T)

                if not self.cons_per_step:
                    inv_term = Qtt[t, idx_u, idx_u]
                    k_term = Qt[t, idx_u]
                    K_term = Qtt[t, idx_u, idx_x]
                else:
                    # inv_term = (1.0 / eta[t]) * Qtt[t, idx_u, idx_u] + prev_traj_distr.inv_pol_covar[t]
                    # k_term = (1.0 / eta[t]) * Qt[t, idx_u] - prev_traj_distr.inv_pol_covar[t].dot(prev_traj_distr.k[t])
                    # K_term = (1.0 / eta[t]) * Qtt[t, idx_u, idx_x] - \
                    #         prev_traj_distr.inv_pol_covar[t].dot(prev_traj_distr.K[t])
                    raise NotImplementedError('COS_PER_STEP=TRUE, NOT IMPLEMENTED')

                # Compute Cholesky decomposition of Q function action component.
                try:
                    U = sp.linalg.cholesky(inv_term)
                    L = U.T
                except LinAlgError as e:
                    # Error thrown when Qtt[idx_u, idx_u] is not symmetric positive definite.
                    LOGGER.debug('LinAlgError: %s', e)
                    fail = t if self.cons_per_step else True
                    break

                if self._hyperparams['update_in_bwd_pass']:
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
                    not self._hyperparams['update_in_bwd_pass']):
                    Vxx[t, :, :] = Qtt[t, idx_x, idx_x] + \
                            traj_distr.K[t].T.dot(Qtt[t, idx_u, idx_u]).dot(traj_distr.K[t]) + \
                            (2 * Qtt[t, idx_x, idx_u]).dot(traj_distr.K[t])
                    Vx[t, :] = Qt[t, idx_x].T + \
                            Qt[t, idx_u].T.dot(traj_distr.K[t]) + \
                            traj_distr.k[t].T.dot(Qtt[t, idx_u, idx_u]).dot(traj_distr.K[t]) + \
                            Qtt[t, idx_x, idx_u].dot(traj_distr.k[t])
                else:
                    Vxx[t, :, :] = Qtt[t, idx_x, idx_x] + Qtt[t, idx_x, idx_u].dot(traj_distr.K[t, :, :])
                    Vx[t, :] = Qt[t, idx_x] + Qtt[t, idx_x, idx_u].dot(traj_distr.k[t, :])
                Vxx[t, :, :] = 0.5 * (Vxx[t, :, :] + Vxx[t, :, :].T)

            if not self._hyperparams['update_in_bwd_pass']:
                traj_distr.K, traj_distr.k = new_K, new_k
                traj_distr.pol_covar = new_pS
                traj_distr.inv_pol_covar = new_ipS
                traj_distr.chol_pol_covar = new_cpS


            # Increment eta on non-SPD Q-function.
            if fail:
                if not self.cons_per_step:
                    old_eta = eta
                    eta = eta0 + del_
                    LOGGER.debug('Increasing eta: %f -> %f', old_eta, eta)
                    old_nu = nu
                    nu = nu0 + del_
                    LOGGER.debug('Increasing nu: %f -> %f', old_nu, nu)
                    old_omega = omega
                    omega = omega0 + del_
                    LOGGER.debug('Increasing omega: %f -> %f', old_omega, omega)

                    del_ *= 2  # Increase del_ exponentially on failure.
                else:
                    old_eta = eta[fail]
                    eta[fail] = eta0[fail] + del_[fail]
                    LOGGER.debug('Increasing eta %d: %f -> %f', fail, old_eta, eta[fail])
                    old_nu = nu[fail]
                    nu[fail] = nu0[fail] + del_[fail]
                    LOGGER.debug('Increasing nu %d: %f -> %f', fail, old_nu, nu[fail])
                    old_omega = omega[fail]
                    omega[fail] = omega0[fail] + del_[fail]
                    LOGGER.debug('Increasing omega %d: %f -> %f', fail, old_omega, omega[fail])

                    del_[fail] *= 2  # Increase del_ exponentially on failure.
                if self.cons_per_step:
                    fail_check = (eta[fail] >= 1e16 or nu[fail] >= 1e16 or omega[fail] >= 1e16)
                else:
                    fail_check = (eta >= 1e16 and nu >= 1e16 and omega >= 1e16)
                if fail_check:
                    if np.any(np.isnan(Fm)) or np.any(np.isnan(fv)):
                        raise ValueError('NaNs encountered in dynamics!')
                    raise ValueError('Failed to find PD solution even for very large eta '
                                     '(check that dynamics and cost are reasonably well conditioned)!')
        backward_bar.end()
        return traj_distr, eta, omega, nu

    def _conv_step_check(self, con, kl_step):
        """Function that checks whether ETA dual gradient descent has converged."""
        if self.cons_per_step:
            return all([abs(con[t]) < (0.1*kl_step[t]) for t in range(con.size)])

        print("Convergence check")
        print("kl_step %d | con %f, 0.1*kl_step %f" % (abs(con) < 0.1 * kl_step, con, 0.1*kl_step))
        return abs(con) < 0.1 * kl_step

    def _conv_good_check(self, con_good, kl_step_good):
        """Function that checks whether OMEGA dual gradient descent has converged."""
        if self.cons_per_step:
            return all([abs(con_good[t]) < (0.1*kl_step_good[t]) for t in range(con_good.size)])

        print("Convergence check")
        print("kl_good %d | con_good %f, 0.1*kl_step_good %f" % (abs(con_good) < 0.1 * kl_step_good, con_good, 0.1*kl_step_good))
        return abs(con_good) < 0.1 * kl_step_good

    def _conv_bad_check(self, con_bad, kl_step_bad):
        """Function that checks whether NU dual gradient descent has converged."""
        if self.cons_per_step:
            return all([abs(con_bad[t]) < (0.1*kl_step_bad[t]) for t in range(con_bad.size)])

        print("Convergence check")
        print("kl_bad %d | con_bad %f, 0.1*kl_step_bad %f" % (abs(con_bad) < 0.1 * kl_step_bad, con_bad, 0.1*kl_step_bad))
        return abs(con_bad) < 0.1 * kl_step_bad

    def _gradient_descent_eta(self, algorithm, a, m, eta, omega, nu):
        T = algorithm.T

        # Get current step_mult and traj_info
        if a is None:
            step_mult = algorithm.cur[m].step_mult
            traj_info = algorithm.cur[m].traj_info
        else:
            step_mult = algorithm.cur[a][m].step_mult
            traj_info = algorithm.cur[a][m].traj_info

        # Get the trajectory distribution that is going to be used as constraint
        if algorithm.gps_algo.lower() == 'mdgps':
            # For MDGPS, constrain to previous NN linearization
            if a is None:
                prev_traj_distr = algorithm.cur[m].pol_info.traj_distr()
            else:
                prev_traj_distr = algorithm.cur[a][m].pol_info.traj_distr()
        else:
            # For BADMM/trajopt, constrain to previous LG controller
            if a is None:
                prev_traj_distr = algorithm.cur[m].traj_distr
            else:
                prev_traj_distr = algorithm.cur[a][m].traj_distr

        # Set KL-divergence step size (epsilon) using step multiplier.
        kl_step = algorithm.base_kl_step * step_mult

        if not self.cons_per_step:
            kl_step *= T

        # We assume at min_eta, kl_div > kl_step, opposite for max_eta.
        if not self.cons_per_step:
            min_eta = self._hyperparams['min_eta']
            max_eta = self._hyperparams['max_eta']
            if a is None:
                LOGGER.debug('_'*60)
                LOGGER.debug("Running DGD for trajectory(condition) %d, eta: %f",
                             m, eta)
            else:
                LOGGER.debug("Running DGD for local agent %d, trajectory(condition) %d, eta: %f",
                             a, m, eta)
        else:
            min_eta = np.ones(T) * self._hyperparams['min_eta']
            max_eta = np.ones(T) * self._hyperparams['max_eta']
            if a is None:
                LOGGER.debug("Running DGD for trajectory %d, avg eta: %f",
                             m, np.mean(eta[:-1]))
            else:
                LOGGER.debug("Running DGD for local agent %d, trajectory %d, avg eta: %f",
                             a, m, np.mean(eta[:-1]))

        max_itr = (DGD_MAX_LS_ITER if self.cons_per_step else DGD_MAX_ITER)  # Less iterations if cons_per_step=True

        # Run ETA GD
        for itr in range(max_itr):
            if not self.cons_per_step:
                LOGGER.debug("Iteration %d, ETA bracket: (%.2e , %.2e , %.2e)", itr, min_eta, eta, max_eta)

            # Run fwd/bwd pass, note that eta may be updated.
            # Compute KL divergence constraint violation.
            traj_distr, eta, omega, nu = self.backward(prev_traj_distr, traj_info, eta, omega, nu, algorithm, m, a)

            if not self._use_prev_distr:
                new_mu, new_sigma = self.forward(traj_distr, traj_info)
                kl_div = traj_distr_kl(new_mu, new_sigma, traj_distr, prev_traj_distr,
                                       tot=(not self.cons_per_step))
            else:
                prev_mu, prev_sigma = self.forward(prev_traj_distr, traj_info)
                kl_div = traj_distr_kl_alt(prev_mu, prev_sigma, traj_distr, prev_traj_distr,
                                           tot=(not self.cons_per_step))

            con = kl_div - kl_step

            # Convergence check - constraint satisfaction.
            if self._conv_step_check(con, kl_step):
                if not self.cons_per_step:
                    LOGGER.debug("KL_epsilon: %f / %f, converged iteration %d", kl_div, kl_step, itr)
                else:
                    LOGGER.debug("KL_epsilon: %f / %f, converged iteration %d",
                                 np.mean(kl_div[:-1]), np.mean(kl_step[:-1]), itr)
                break

            if not self.cons_per_step:
                # Choose new eta (bisect bracket or multiply by constant)
                if con < 0:  # Eta was too big.
                    max_eta = eta
                    geom = np.sqrt(min_eta*max_eta)  # Geometric mean.
                    new_eta = max(geom, 0.1*max_eta)
                    LOGGER.debug("KL: %f / %f, eta too big, new eta: %f", kl_div, kl_step, new_eta)
                else:  # Eta was too small.
                    min_eta = eta
                    geom = np.sqrt(min_eta*max_eta)  # Geometric mean.
                    new_eta = min(geom, 10.0*min_eta)
                    LOGGER.debug("KL: %f / %f, eta too small, new eta: %f", kl_div, kl_step, new_eta)

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
                    LOGGER.debug("avg KL: %f / %f, avg new eta: %f", np.mean(kl_div[:-1]), np.mean(kl_step[:-1]),
                                 np.mean(eta[:-1]))
        if itr > max_itr - 1:
            LOGGER.debug("After %d iterations for ETA, the constraints have not been satisfied.", itr + 1)

        return eta, self._conv_step_check(con, kl_step)

    def _gradient_descent_nu(self, algorithm, a, m, eta, omega, nu):
        T = algorithm.T

        # Get current step_mult and traj_info
        if a is None:
            traj_info = algorithm.cur[m].traj_info
            bad_step_mult = algorithm.cur[m].bad_step_mult
            bad_traj_info = algorithm.bad_duality_info[m]
        else:
            traj_info = algorithm.cur[a][m].traj_info
            bad_step_mult = algorithm.cur[a][m].bad_step_mult
            bad_traj_info = algorithm.bad_duality_info[a][m]

        # Get the trajectory distribution that is going to be used as constraint
        if algorithm.gps_algo.lower() == 'mdgps':
            # For MDGPS, constrain to previous NN linearization
            if a is None:
                prev_traj_distr = algorithm.cur[m].pol_info.traj_distr()
            else:
                prev_traj_distr = algorithm.cur[a][m].pol_info.traj_distr()
        else:
            # For BADMM/trajopt, constrain to previous LG controller
            if a is None:
                prev_traj_distr = algorithm.cur[m].traj_distr
            else:
                prev_traj_distr = algorithm.cur[a][m].traj_distr

        # Good and Bad traj_dist
        if a is None:
            bad_traj_distr = algorithm.bad_duality_info[m].traj_dist
        else:
            bad_traj_distr = algorithm.bad_duality_info[a][m].traj_dist


        # Set KL-divergence step size (epsilon) using step multiplier.
        kl_step_bad = algorithm.base_kl_bad * bad_step_mult

        if not self.cons_per_step:
            kl_step_bad *= T

        # We assume at min_eta, kl_div > kl_step, opposite for max_eta.
        if not self.cons_per_step:
            min_nu = self._hyperparams['min_nu']
            max_nu = self._hyperparams['max_nu']
            if a is None:
                LOGGER.debug('_'*60)
                LOGGER.debug("Running DGD for trajectory(condition) %d, nu: %f",
                             m, nu)
            else:
                LOGGER.debug("Running DGD for local agent %d, trajectory(condition) %d, nu: %f",
                             a, m, nu)
        else:
            min_nu = np.ones(T) * self._hyperparams['min_nu']
            max_nu = np.ones(T) * self._hyperparams['max_nu']
            if a is None:
                LOGGER.debug("Running DGD for trajectory %d, avg nu: %f",
                             m, np.mean(nu[:-1]))
            else:
                LOGGER.debug("Running DGD for local agent %d, trajectory %d, avg nu: %f",
                             a, m, np.mean(nu[:-1]))

        max_itr = (DGD_MAX_LS_ITER if self.cons_per_step else DGD_MAX_ITER)  # Less iterations if cons_per_step=True

        for itr in range(max_itr):
            if not self.cons_per_step:
                LOGGER.debug("Iteration %d, NU bracket: (%.2e , %.2e , %.2e)", itr, min_nu, nu, max_nu)

            # Run fwd/bwd pass, note that nu may be updated.
            # Compute KL divergence constraint violation.
            traj_distr, eta, omega, nu = self.backward(prev_traj_distr, traj_info, eta, omega, nu, algorithm, m, a)

            if not self._use_prev_distr:
                new_mu, new_sigma = self.forward(traj_distr, traj_info)
                # Using kl_alt instead the original one
                kl_div_bad = traj_distr_kl_alt(new_mu, new_sigma, traj_distr, bad_traj_distr,
                                               tot=(not self.cons_per_step))
            else:
                prev_mu, prev_sigma = self.forward(prev_traj_distr, traj_info)
                kl_div_bad = traj_distr_kl_alt(prev_mu, prev_sigma, traj_distr, bad_traj_distr,
                                               tot=(not self.cons_per_step))

            # con_bad = kl_div_bad - kl_step_bad
            con_bad = kl_step_bad - kl_div_bad

            # Convergence check - constraint satisfaction.
            if self._conv_bad_check(con_bad, kl_step_bad):
                if not self.cons_per_step:
                    LOGGER.debug("KL_nu: %f / %f, converged iteration %d", kl_div_bad, kl_step_bad, itr)
                else:
                    LOGGER.debug("KL_nu: %f / %f, converged iteration %d",
                                 np.mean(kl_div_bad[:-1]), np.mean(kl_step_bad[:-1]), itr)
                break

            # In case it does not converged
            if not self.cons_per_step:
                # Choose new nu (bisect bracket or multiply by constant)
                if con_bad > 0:  # Nu was too big.
                    max_nu = nu
                    geom = np.sqrt(min_nu*max_nu)  # Geometric mean.
                    new_nu = max(geom, 0.1*max_nu)
                    LOGGER.debug("KL: %f / %f, nu too big, new nu: %f", kl_div_bad, kl_step_bad, new_nu)
                else:  # Nu was too small.
                    min_nu = nu
                    geom = np.sqrt(min_nu*max_nu)  # Geometric mean.
                    new_nu = min(geom, 10.0*min_nu)
                    LOGGER.debug("KL: %f / %f, nu too small, new nu: %f", kl_div_bad, kl_step_bad, new_nu)

                # Logarithmic mean: log_mean(x,y) = (y - x)/(log(y) - log(x))
                nu = new_nu
            else:
                for t in range(T):
                    if con_bad[t] < 0:  # Nu was too big.
                        max_nu[t] = nu[t]
                        geom = np.sqrt(min_nu[t]*max_nu[t])  # Geometric mean.
                        nu[t] = max(geom, 0.1*max_nu[t])
                    else:
                        min_nu[t] = nu[t]
                        geom = np.sqrt(min_nu[t]*max_nu[t])  # Geometric mean.
                        nu[t] = min(geom, 10.0*min_nu[t])
                if itr % 10 == 0:
                    LOGGER.debug("avg KL: %f / %f, avg new nu: %f", np.mean(kl_div_bad[:-1]), np.mean(kl_step_bad[:-1]),
                                 np.mean(nu[:-1]))
        if itr > max_itr - 1:
            LOGGER.debug("After %d iterations for NU, the constraints have not been satisfied.", itr + 1)

        return nu, self._conv_bad_check(con_bad, kl_step_bad)

    def _gradient_descent_omega(self, algorithm, a, m, eta, omega, nu):
        T = algorithm.T

        # Get current step_mult and traj_info
        if a is None:
            traj_info = algorithm.cur[m].traj_info
            good_step_mult = algorithm.cur[m].good_step_mult
            good_traj_info = algorithm.good_duality_info[m]
        else:
            traj_info = algorithm.cur[a][m].traj_info
            good_step_mult = algorithm.cur[a][m].good_step_mult
            good_traj_info = algorithm.good_duality_info[a][m]

        # Get the trajectory distribution that is going to be used as constraint
        if algorithm.gps_algo.lower() == 'mdgps':
            # For MDGPS, constrain to previous NN linearization
            if a is None:
                prev_traj_distr = algorithm.cur[m].pol_info.traj_distr()
            else:
                prev_traj_distr = algorithm.cur[a][m].pol_info.traj_distr()
        else:
            # For BADMM/trajopt, constrain to previous LG controller
            if a is None:
                prev_traj_distr = algorithm.cur[m].traj_distr
            else:
                prev_traj_distr = algorithm.cur[a][m].traj_distr

        # Good and Bad traj_dist
        if a is None:
            good_traj_distr = algorithm.good_duality_info[m].traj_dist
        else:
            good_traj_distr = algorithm.good_duality_info[a][m].traj_dist


        # Set KL-divergence step size (epsilon) using step multiplier.
        kl_step_good = algorithm.base_kl_good * good_step_mult

        if not self.cons_per_step:
            kl_step_good *= T

        # We assume at min_eta, kl_div > kl_step, opposite for max_eta.
        if not self.cons_per_step:
            min_omega = self._hyperparams['min_omega']
            max_omega = self._hyperparams['max_omega']
            if a is None:
                LOGGER.debug('_'*60)
                LOGGER.debug("Running DGD for trajectory(condition) %d, omega: %f",
                             m, omega)
            else:
                LOGGER.debug("Running DGD for local agent %d, trajectory(condition) %d, omega: %f",
                             a, m, omega)
        else:
            min_omega = np.ones(T) * self._hyperparams['min_omega']
            max_omega = np.ones(T) * self._hyperparams['max_omega']
            if a is None:
                LOGGER.debug("Running DGD for trajectory %d, avg omega: %f",
                             m, np.mean(omega[:-1]))
            else:
                LOGGER.debug("Running DGD for local agent %d, trajectory %d, avg omega: %f",
                             a, m, np.mean(omega[:-1]))

        max_itr = (DGD_MAX_LS_ITER if self.cons_per_step else DGD_MAX_ITER)  # Less iterations if cons_per_step=True

        for itr in range(max_itr):
            if not self.cons_per_step:
                LOGGER.debug("Iteration %d, NU bracket: (%.2e , %.2e , %.2e)", itr, min_omega, omega, max_omega)

            # Run fwd/bwd pass, note that omega may be updated.
            # Compute KL divergence constraint violation.
            traj_distr, eta, omega, nu = self.backward(prev_traj_distr, traj_info, eta, omega, nu, algorithm, m, a)

            if not self._use_prev_distr:
                new_mu, new_sigma = self.forward(traj_distr, traj_info)
                # Using kl_alt instead the original one
                kl_div_good = traj_distr_kl_alt(new_mu, new_sigma, traj_distr, good_traj_distr,
                                               tot=(not self.cons_per_step))
            else:
                prev_mu, prev_sigma = self.forward(prev_traj_distr, traj_info)
                kl_div_good = traj_distr_kl_alt(prev_mu, prev_sigma, traj_distr, good_traj_distr,
                                               tot=(not self.cons_per_step))

            con_good = kl_div_good - kl_step_good

            # Convergence check - constraint satisfaction.
            if self._conv_good_check(con_good, kl_step_good):
                if not self.cons_per_step:
                    LOGGER.debug("KL_omega: %f / %f, converged iteration %d", kl_div_good, kl_step_good, itr)
                else:
                    LOGGER.debug("KL_omega: %f / %f, converged iteration %d",
                                 np.mean(kl_div_good[:-1]), np.mean(kl_step_good[:-1]), itr)
                break

            # In case it does not converged
            if not self.cons_per_step:
                # Choose new omega (bisect bracket or multiply by constant)
                if con_good < 0:  # Nu was too big.
                    max_omega = omega
                    geom = np.sqrt(min_omega*max_omega)  # Geometric mean.
                    new_omega = max(geom, 0.1*max_omega)
                    LOGGER.debug("KL: %f / %f, omega too big, new omega: %f", kl_div_good, kl_step_good, new_omega)
                else:  # Nu was too small.
                    min_omega = omega
                    geom = np.sqrt(min_omega*max_omega)  # Geometric mean.
                    new_omega = min(geom, 10.0*min_omega)
                    LOGGER.debug("KL: %f / %f, omega too small, new omega: %f", kl_div_good, kl_step_good, new_omega)

                # Logarithmic mean: log_mean(x,y) = (y - x)/(log(y) - log(x))
                omega = new_omega
            else:
                for t in range(T):
                    if con_good[t] < 0:  # Nu was too big.
                        max_omega[t] = omega[t]
                        geom = np.sqrt(min_omega[t]*max_omega[t])  # Geometric mean.
                        omega[t] = max(geom, 0.1*max_omega[t])
                    else:
                        min_omega[t] = omega[t]
                        geom = np.sqrt(min_omega[t]*max_omega[t])  # Geometric mean.
                        omega[t] = min(geom, 10.0*min_omega[t])
                if itr % 10 == 0:
                    LOGGER.debug("avg KL: %f / %f, avg new omega: %f", np.mean(kl_div_good[:-1]), np.mean(kl_step_good[:-1]),
                                 np.mean(omega[:-1]))
        if itr > max_itr - 1:
            LOGGER.debug("After %d iterations for NU, the constraints have not been satisfied.", itr + 1)

        return omega, self._conv_good_check(con_good, kl_step_good)