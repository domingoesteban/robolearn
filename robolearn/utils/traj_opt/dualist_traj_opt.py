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
from scipy.optimize import minimize

from robolearn.utils.traj_opt.traj_opt import TrajOpt
from robolearn.utils.traj_opt.config import default_traj_opt_mdreps_hyperparams
from robolearn.utils.traj_opt.traj_opt_utils import traj_distr_kl, traj_distr_kl_alt
from robolearn.utils.print_utils import ProgressBar

print_DGD_log = False

MAX_ALL_DGD = 20
DGD_MAX_ITER = 50
DGD_MAX_LS_ITER = 20
DGD_MAX_GD_ITER = 200

ALPHA, BETA1, BETA2, EPS = 0.005, 0.9, 0.999, 1e-8  # Adam parameters


class DualistTrajOpt(TrajOpt):
    """ Dualist trajectory optimization """
    def __init__(self, hyperparams):
        config = copy.deepcopy(default_traj_opt_mdreps_hyperparams)
        config.update(hyperparams)

        TrajOpt.__init__(self, config)

        self.cons_per_step = config['cons_per_step']
        self._use_prev_distr = config['use_prev_distr']
        self._update_in_bwd_pass = config['update_in_bwd_pass']

        self.consider_good = config['good_const']
        self.consider_bad = config['bad_const']

        if not self.consider_bad:
            self._hyperparams['min_nu'] = 0
            self._hyperparams['max_nu'] = 0

        if not self.consider_good:
            self._hyperparams['min_omega'] = 0
            self._hyperparams['max_omega'] = 0

        if not self._use_prev_distr:
            self._traj_distr_kl_fcn = traj_distr_kl_alt
        else:
            self._traj_distr_kl_fcn = traj_distr_kl

        self.LOGGER = logging.getLogger(__name__)

    def set_logger(self, logger):
        self.LOGGER = logger

    # TODO - Add arg and return spec on this function.
    def update(self, m, algorithm):
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
        eta = algorithm.cur[m].eta
        if self.cons_per_step:
            eta = np.ones(T) * eta

        # Get current omega
        omega = algorithm.cur[m].omega
        if self.cons_per_step:
            omega = np.ones(T) * omega

        # Get current nu
        nu = algorithm.cur[m].nu
        if self.cons_per_step:
            nu = np.ones(T) * nu

        if self.consider_good is False:
            omega *= 0

        if self.consider_bad is False:
            nu *= 0

        # init_eta = self.cur[m].eta
        # init_nu = self.cur[m].nu
        # init_omega = self.cur[m].omega
        # init_duals = np.array([init_eta, init_nu, init_omega])
        # res = minimize(self.lagrangian_function, init_duals,
        #                args=(algorithm, a, m),
        #                bounds=((np.zeros(3), None),),
        #                jac=self.lagrangian_gradient,
        #                method=None)
        # eta = res.x[0]
        # omega = res.x[1]
        # nu = res.x[2]

        traj_distr, duals, convs = \
            self._gradient_descent_all(algorithm, m, eta, nu, omega)
        eta = duals[0]
        nu = duals[1]
        omega = duals[2]
        eta_conv = convs[0]
        nu_conv = convs[1]
        omega_conv = convs[2]

        if eta_conv and nu_conv and omega_conv:
            self.LOGGER.info("ALL DGD has converged.")
        else:
            self.LOGGER.info("ALL DGD has NOT converged.")

        self.LOGGER.info('TODO: commenting a lot of stuff after this print')
        # if self.cons_per_step and not self._conv_check(con, kl_step):
        #     self.LOGGER.info("IT DID NOT CONVERGED!!")
        #     m_b, v_b = np.zeros(T-1), np.zeros(T-1)

        #     for itr in range(DGD_MAX_GD_ITER):
        #         traj_distr, eta = self.backward(prev_traj_distr, traj_info, eta, algorithm, m, a)

        #         if not self._use_prev_distr:
        #             new_mu, new_sigma = self.forward(traj_distr, traj_info)
        #             kl_div = traj_distr_kl(new_mu, new_sigma, traj_distr, prev_traj_distr, tot=False)
        #         else:
        #             prev_mu, prev_sigma = self.forward(prev_traj_distr, traj_info)
        #             kl_div = traj_distr_kl_alt(prev_mu, prev_sigma, traj_distr, prev_traj_distr, tot=False)

        #         con = kl_div - kl_step
        #         if self._conv_check(con, kl_step):
        #             self.LOGGER.info("KL: %f / %f, converged iteration %d", np.mean(kl_div[:-1]), np.mean(kl_step[:-1]),
        #                          itr)
        #             break

        #         m_b = (BETA1 * m_b + (1-BETA1) * con[:-1])
        #         m_u = m_b / (1 - BETA1 ** (itr+1))
        #         v_b = (BETA2 * v_b + (1-BETA2) * np.square(con[:-1]))
        #         v_u = v_b / (1 - BETA2 ** (itr+1))
        #         eta[:-1] = np.minimum(np.maximum(eta[:-1] + ALPHA * m_u / (np.sqrt(v_u) + EPS),
        #                                          self._hyperparams['min_eta']),
        #                               self._hyperparams['max_eta'])

        #         if itr % 10 == 0:
        #             self.LOGGER.info("avg KL: %f / %f, avg new eta: %f", np.mean(kl_div[:-1]), np.mean(kl_step[:-1]),
        #                          np.mean(eta[:-1]))

        if not eta_conv:
            self.LOGGER.warning("Final KL_step divergence after "
                                "DGD convergence is too high.")

        if not nu_conv:
            self.LOGGER.warning("Final KL_bad divergence after "
                                "DGD convergence is too low.")

        if not omega_conv:
            self.LOGGER.warning("Final KL_good divergence after "
                                "DGD convergence is too high.")

        # # TODO: IMPROVE THIS
        # # Get Traj_info
        # traj_info = algorithm.cur[m].traj_info
        #
        # # Get the trajectory distribution that is going to be used as constraint
        # gps_algo = type(algorithm).__name__
        # if gps_algo in ['MDGPS' or 'DualGPS']:
        #     # For MDGPS, constrain to previous NN linearization
        #     prev_traj_distr = algorithm.cur[m].pol_info.traj_distr()
        # else:
        #     # For BADMM/trajopt, constrain to previous LG controller
        #     prev_traj_distr = algorithm.cur[m].traj_distr
        #
        # self.LOGGER.info("TODO: Running backward AGAIN with dual_to_check "
        #                  "omega, to get traj_dist")
        # traj_distr, eta, omega, nu = self.backward(prev_traj_distr, traj_info,
        #                                            eta, omega, nu, algorithm, m,
        #                                            dual_to_check='omega')

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
                    0.5 * np.sum(sigma[t, :, :] * traj_info.Cm[t, :, :]) +\
                    0.5 * mu[t, :].T.dot(traj_info.Cm[t, :, :]).dot(mu[t, :]) +\
                    mu[t, :].T.dot(traj_info.cv[t, :])
        return predicted_cost

    @staticmethod
    def forward(traj_distr, traj_info):
        """
        Perform LQR forward pass. Computes state-action marginals from
        dynamics and policy.
        Args:
            traj_distr: A linear Gaussian policy object.
            traj_info: A TrajectoryInfo object.
        Returns:
            mu: T x (dX+dU) mean vector.
            sigma: T x (dX+dU) x (dX+dU) covariance matrix.
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

        if print_DGD_log:
            forward_bar = ProgressBar(T, bar_title='Forward pass')
        for t in range(T):
            if print_DGD_log:
                forward_bar.update(t)
            sigma[t, :, :] = np.vstack([
                np.hstack([
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
        if print_DGD_log:
            forward_bar.end()
        return mu, sigma

    def backward(self, prev_traj_distr, traj_info, eta, omega, nu, algorithm, m,
                 dual_to_check='eta'):
        """
        Perform LQR backward pass. This computes a new linear Gaussian policy
        object.
        Args:
            prev_traj_distr: A linear Gaussian policy object from
                previous iteration.
            traj_info: A TrajectoryInfo object.
            eta: Dual variable prev_traj, KL(p||q) <= epsilon.
            omega: Dual variable good_traj, KL(p||g) <= chi.
            nu: Dual variable bad_traj, KL(p||b) >= xi.
            algorithm: Algorithm object needed to compute costs.
            m: Condition number.
            dual_to_check: Dual variable to check.
        Returns:
            traj_distr: A new linear Gaussian policy.
            new_eta: The (possibly) updated dual variable. Updates happen if
                the Q-function is not PD.
            new_omega: The (possibly) updated dual variable. Updates happen if
                the Q-function is not PD.
            new_nu: The (possibly) updated dual variable. Updates happen if
                the Q-function is not PD.
        """

        # Constants.
        T = prev_traj_distr.T
        dU = prev_traj_distr.dU
        dX = prev_traj_distr.dX

        if print_DGD_log:
            backward_bar = ProgressBar(T, bar_title='Backward pass')
            backward_bar_count = 0

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
        del_ = self._hyperparams['del0']
        del_good_ = self._hyperparams['del0_good']
        del_bad_ = self._hyperparams['del0_bad']
        if self.cons_per_step:
            del_ = np.ones(T) * del_
            del_good_ = np.ones(T) * del_good_
            del_bad_ = np.ones(T) * del_bad_

        eta0 = eta
        omega0 = omega
        nu0 = nu

        # Run Dynamic Programming.
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
            fCm, fcv = compute_cost_fcn(m, eta, omega, nu,
                                        augment=(not self.cons_per_step))

            # Compute state-action-state function at each time step.
            for t in range(T - 1, -1, -1):
                if print_DGD_log:
                    backward_bar.update(backward_bar_count)
                # Add in the cost.
                Qtt[t] = fCm[t, :, :]  # (X+U) x (X+U)
                Qt[t] = fcv[t, :]  # (X+U) x 1

                # Add in the value function from the next time step.
                if t < T - 1:
                    gps_algo = type(algorithm).__name__
                    if gps_algo.lower() == 'badmm':
                        # multiplier = (pol_wt[t+1] + eta)/(pol_wt[t] + eta)
                        raise NotImplementedError("not implemented badmm in "
                                                  "mdreps")
                    else:
                        multiplier = 1.0

                    Qtt[t] += multiplier * \
                            Fm[t, :, :].T.dot(Vxx[t+1, :, :]).dot(Fm[t, :, :])
                    Qt[t] += multiplier * \
                            Fm[t, :, :].T.dot(Vx[t+1, :] +
                                              Vxx[t+1, :, :].dot(fv[t, :]))

                # Symmetrize quadratic component.
                Qtt[t] = 0.5 * (Qtt[t] + Qtt[t].T)

                if not self.cons_per_step:
                    inv_term = Qtt[t, idx_u, idx_u]  # Quu
                    k_term = Qt[t, idx_u]  # Qu
                    K_term = Qtt[t, idx_u, idx_x]  # Qux
                else:
                    raise NotImplementedError('CONS_PER_STEP=TRUE, NOT IMPLEMENTED')
                    # inv_term = (1.0 / eta[t]) * Qtt[t, idx_u, idx_u] + prev_traj_distr.inv_pol_covar[t]
                    # k_term = (1.0 / eta[t]) * Qt[t, idx_u] - prev_traj_distr.inv_pol_covar[t].dot(prev_traj_distr.k[t])
                    # K_term = (1.0 / eta[t]) * Qtt[t, idx_u, idx_x] - \
                    #         prev_traj_distr.inv_pol_covar[t].dot(prev_traj_distr.K[t])

                # Compute Cholesky decomposition of Q function action component.
                try:
                    U = sp.linalg.cholesky(inv_term)
                    L = U.T
                except LinAlgError as e:
                    # Error thrown when Qtt[idx_u, idx_u] is not
                    # symmetric positive definite (SPD).
                    self.LOGGER.error('LinAlgError: %s', e)
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
                    new_k[t, :] = -sp.linalg.solve_triangular(
                        U, sp.linalg.solve_triangular(L, k_term, lower=True))
                    new_K[t, :, :] = -sp.linalg.solve_triangular(
                        U, sp.linalg.solve_triangular(L, K_term, lower=True))

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
                    Vxx[t, :, :] = Qtt[t, idx_x, idx_x] + \
                            Qtt[t, idx_x, idx_u].dot(traj_distr.K[t, :, :])
                    Vx[t, :] = Qt[t, idx_x] + \
                            Qtt[t, idx_x, idx_u].dot(traj_distr.k[t, :])
                Vxx[t, :, :] = 0.5 * (Vxx[t, :, :] + Vxx[t, :, :].T)

            if not self._hyperparams['update_in_bwd_pass']:
                traj_distr.K, traj_distr.k = new_K, new_k
                traj_distr.pol_covar = new_pS
                traj_distr.inv_pol_covar = new_ipS
                traj_distr.chol_pol_covar = new_cpS


            # Increment eta on non-SPD Q-function.
            if fail:
                if not self.cons_per_step:
                    if dual_to_check == 'eta':
                        old_eta = eta
                        eta = eta0 + del_
                        self.LOGGER.info('Non-SPD Q: Increasing eta: %f -> %f',
                                         old_eta, eta)
                        del_ *= 2  # Increase del_ exponentially on failure.
                    elif dual_to_check == 'nu':
                        old_eta = eta
                        eta = eta0 + del_
                        self.LOGGER.info('Non-SPD Q: Increasing eta: %f -> %f',
                                         old_eta, eta)
                        del_ *= 2  # Increase del_ exponentially on failure.
                        old_nu = nu
                        nu = nu0 - del_bad_
                        self.LOGGER.info('Non-SPD Q: Decreasing nu: %f -> %f',
                                         old_nu, nu)
                        del_bad_ *= 2  # Decrease del_ exponentially on failure.
                        old_omega = omega
                        omega = omega0 + del_good_
                        self.LOGGER.info('Non-SPD Q: Increasing omega: %f -> %f',
                                         old_omega, omega)
                        del_good_ *= 2  # Decrease del_ exponentially on failure.
                    elif dual_to_check == 'nu2':
                        old_eta = eta
                        eta = eta0 + del_
                        self.LOGGER.info('Non-SPD Q: Increasing eta: %f -> %f',
                                         old_eta, eta)
                        del_ *= 2  # Increase del_ exponentially on failure.
                        old_nu = nu
                        nu = nu0 - del_bad_
                        self.LOGGER.info('Non-SPD Q: Decreasing nu: %f -> %f',
                                         old_nu, nu)
                        del_bad_ *= 2  # Decrease del_ exponentially on failure.
                        #old_omega = omega
                        #omega = omega0 + del_good_
                        # self.LOGGER.info('Non-SPD Q: Increasing omega: %f -> %f',
                        #                  old_omega, omega)
                        #del_good_ *= 2  # Decrease del_ exponentially on failure.
                    elif dual_to_check == 'omega':
                        old_omega = omega
                        omega = omega0 + del_good_
                        self.LOGGER.info('Non-SPD Q: Increasing omega: %f -> %f',
                                         old_omega, omega)
                        del_good_ *= 2  # Increase del_ exponentially on failure.
                    else:
                        raise ValueError("Wrong dual_to_check option %s"
                                         % dual_to_check)

                else:
                    if dual_to_check == 'eta':
                        old_eta = eta[fail]
                        eta[fail] = eta0[fail] + del_[fail]
                        self.LOGGER.info('Non-SPD Q: Increasing eta[%d]: %f -> %f',
                                         fail, old_eta, eta[fail])
                        del_[fail] *= 2  # Increase del_ exponentially on failure.
                    elif dual_to_check == 'nu':
                        old_eta = eta[fail]
                        eta[fail] = eta0[fail] + del_[fail]
                        self.LOGGER.info('Increasing eta[%d]: %f -> %f',
                                         fail, old_eta, eta[fail])
                        del_[fail] *= 2  # Increase del_ exponentially on failure.
                        # old_nu = nu[fail]
                        # nu[fail] = nu0[fail] - del_bad_[fail]
                        # self.LOGGER.info('Decreasing nu[%d]: %f -> %f',
                        #                  fail, old_nu, nu[fail])
                        # del_bad_[fail] *= 2  # Increase del_ exponentially on failure.
                        old_omega = omega[fail]
                        omega[fail] = omega0[fail] + del_good_[fail]
                        self.LOGGER.info('Increasing omega[%d]: %f -> %f',
                                         fail, old_omega, omega[fail])
                        del_good_[fail] *= 2  # Increase del_ exponentially on failure.
                    elif dual_to_check == 'omega':
                        old_omega = omega[fail]
                        omega[fail] = omega0[fail] + del_good_[fail]
                        self.LOGGER.info('Increasing omega[%d]: %f -> %f',
                                         fail, old_omega, omega[fail])
                        del_good_[fail] *= 2  # Increase del_ exponentially on failure.
                    else:
                        raise ValueError("Wrong dual_to_check option %s"
                                         % dual_to_check)

                if self.cons_per_step:
                    fail_check = (eta[fail] > self._hyperparams['max_eta']
                                  or nu[fail] < self._hyperparams['min_nu']
                                  or omega[fail] > self._hyperparams['max_omega'])
                else:
                    fail_check = (eta > self._hyperparams['max_eta']
                                  or nu < self._hyperparams['min_nu']
                                  or omega > self._hyperparams['max_omega'])

                if fail_check:
                    if np.any(np.isnan(Fm)) or np.any(np.isnan(fv)):
                        raise ValueError('NaNs encountered in dynamics!')
                    if dual_to_check == 'eta':
                        raise ValueError("Failed to find PD solution even for "
                                         "very large eta "
                                         "(check that dynamics and cost are "
                                         "reasonably well conditioned)!")
                    elif dual_to_check == 'nu':
                        raise ValueError("Failed to find PD solution even for "
                                         "very small nu "
                                         "(check that dynamics and cost are "
                                         "reasonably well conditioned)!")
                    elif dual_to_check == 'nu2':
                        raise ValueError("Failed to find PD solution even for "
                                         "very small nu and very large eta "
                                         "(check that dynamics and cost are "
                                         "reasonably well conditioned)!")
                    elif dual_to_check == 'omega':
                        raise ValueError("Failed to find PD solution even for "
                                         "very large omega "
                                         "(check that dynamics and cost are "
                                         "reasonably well conditioned)!")
                    else:
                        raise ValueError("Wrong dual_to_check option %s"
                                         % dual_to_check)
        if print_DGD_log:
            backward_bar.end()
        return traj_distr, eta, omega, nu

    def _conv_prev_check(self, con, kl_step):
        """
        Function that checks whether ETA dual gradient descent has converged.
        """
        step_tol = self._hyperparams['step_tol']
        if self.cons_per_step:
            return all([abs(con[t]) < (step_tol*kl_step[t]) for t in range(con.size)])

        self.LOGGER.info("Conv kl_step:%s | "
                         "abs(con) < %.1f*kl_step | "
                         "abs(%f) < %f | %f%%"
                         % (abs(con) < step_tol * kl_step,
                            step_tol,
                            con, step_tol*kl_step, abs(con*100/kl_step)))
        return abs(con) < step_tol * kl_step

    def _conv_good_check(self, con_good, kl_step_good):
        """
        Function that checks whether OMEGA dual gradient descent has converged.
        """
        good_tol = self._hyperparams['good_tol']
        if self.cons_per_step:
            return all([abs(con_good[t]) < (good_tol*kl_step_good[t]) for t in range(con_good.size)])

        self.LOGGER.info("Conv kl_good:%s | "
                         "abs(con_good) < %.1f*kl_step_good | "
                         "abs(%f) < %f | %f%%"
                         % (abs(con_good) < good_tol * kl_step_good,
                            good_tol,
                            con_good,
                            good_tol*kl_step_good,
                            abs(con_good*100/kl_step_good)))
        return abs(con_good) < good_tol * kl_step_good

    def _conv_bad_check(self, con_bad, kl_step_bad):
        """
        Function that checks whether NU dual gradient descent has converged.
        """
        bad_tol = self._hyperparams['bad_tol']
        if self.cons_per_step:
            # return all([abs(con_bad[t]) < (bad_tol*kl_step_bad[t]) for t in range(con_bad.size)])
            return all([con_bad[t] <= 0 for t in range(con_bad.size)])

        self.LOGGER.info("Conv kl_bad:%s | con_bad <= 0 | %f < %f | %f%%"
                         % (con_bad < 0, con_bad, 0, con_bad*100/kl_step_bad))

        # return abs(con_bad) < bad_tol * kl_step_bad
        return con_bad <= 0

    def _gradient_descent_all(self, algorithm, m, eta, nu, omega):
        T = algorithm.T

        # Get current step_mult and traj_info
        step_mult = algorithm.cur[m].step_mult
        bad_step_mult = algorithm.cur[m].bad_step_mult
        good_step_mult = algorithm.cur[m].good_step_mult
        traj_info = algorithm.cur[m].traj_info

        # Get the trajectory distribution that is going to be used as constraint
        gps_algo = type(algorithm).__name__
        if gps_algo in ['DualGPS', 'MDGPS']:
            # For MDGPS, constrain to previous NN linearization
            prev_traj_distr = algorithm.cur[m].pol_info.traj_distr()
        else:
            # For BADMM/trajopt, constrain to previous LG controller
            prev_traj_distr = algorithm.cur[m].traj_distr

        # Good and Bad traj_dist
        bad_traj_distr = algorithm.bad_duality_info[m].traj_dist
        good_traj_distr = algorithm.good_duality_info[m].traj_dist

        # Set KL-divergence step size (epsilon) using step multiplier.
        kl_step = algorithm.base_kl_step * step_mult

        kl_step_good = algorithm.base_kl_good * good_step_mult
        kl_step_bad = algorithm.base_kl_bad * bad_step_mult

        if not self.cons_per_step:
            kl_step *= T
            kl_step_good *= T
            kl_step_bad *= T

        # We assume at min_eta, kl_div > kl_step, opposite for max_eta.
        if not self.cons_per_step:
            min_eta = self._hyperparams['min_eta']
            max_eta = self._hyperparams['max_eta']
            min_nu = self._hyperparams['min_nu']
            max_nu = self._hyperparams['max_nu']
            min_omega = self._hyperparams['min_omega']
            max_omega = self._hyperparams['max_omega']
            self.LOGGER.info('_'*60)
            self.LOGGER.info("Running DGD for traj[%d] | "
                             "eta: %4f, nu: %4f, omega: %4f",
                             m, eta, nu, omega)
            self.LOGGER.info('_'*60)
        else:
            min_eta = np.ones(T) * self._hyperparams['min_eta']
            max_eta = np.ones(T) * self._hyperparams['max_eta']
            min_nu = np.ones(T) * self._hyperparams['min_nu']
            max_nu = np.ones(T) * self._hyperparams['max_nu']
            min_omega = np.ones(T) * self._hyperparams['min_omega']
            max_omega = np.ones(T) * self._hyperparams['max_omega']
            self.LOGGER.info('_'*60)
            self.LOGGER.info("Running DGD for trajectory %d,"
                             "avg eta: %f, avg nu: %f, avg omega: %f",
                             m, np.mean(eta[:-1]), np.mean(nu[:-1]),
                             np.mean(omega[:-1]))
            self.LOGGER.info('_'*60)

        # Less iterations if cons_per_step=True
        max_itr = (DGD_MAX_LS_ITER if self.cons_per_step else
                   DGD_MAX_ITER)

        # Run ALL GD
        for itr in range(max_itr):
            self.LOGGER.info("-"*15)
            self.LOGGER.info("ALL GD | iter %d| "
                             "eta %.2e, nu %.2e, omega %.2e"
                             % (itr, eta, nu, omega))
            if not self.cons_per_step:
                self.LOGGER.info("ALL DGD iteration %d| "
                                 "ETA bracket: (%.2e , %.2e , %.2e)",
                                 itr, min_eta, eta, max_eta)
                self.LOGGER.info("ALL DGD iteration %d| "
                                 "NU bracket: (%.2e , %.2e , %.2e)",
                                 itr, min_nu, nu, max_nu)
                self.LOGGER.info("ALL DGD iteration %d| "
                                 "OMEGA bracket: (%.2e , %.2e , %.2e)",
                                 itr, min_omega, omega, max_omega)

            # Run fwd/bwd pass, note that eta may be updated.
            # Compute KL divergence constraint violation.
            if self.consider_good and self.consider_bad:
                self.LOGGER.info("Running ALL GD backward with "
                                 "dual_to_check NU")
                traj_distr, eta, omega, nu = \
                    self.backward(prev_traj_distr, traj_info,
                                  eta, omega, nu,
                                  # algorithm, m, dual_to_check='nu')
                                  algorithm, m, dual_to_check='eta')
            elif not self.consider_good and self.consider_bad:
                self.LOGGER.info("Running ALL GD backward with "
                                 "dual_to_check NU2")
                traj_distr, eta, omega, nu = \
                    self.backward(prev_traj_distr, traj_info,
                                  eta, omega, nu,
                                  algorithm, m, dual_to_check='nu2')
            elif self.consider_good and not self.consider_bad:
                self.LOGGER.info("Running ALL GD backward with "
                                 "dual_to_check OMEGA")
                traj_distr, eta, omega, nu = \
                    self.backward(prev_traj_distr, traj_info,
                                  eta, omega, nu,
                                  algorithm, m, dual_to_check='omega')
            else:
                self.LOGGER.info("Running ALL GD backward with "
                                 "dual_to_check ETA")
                traj_distr, eta, omega, nu = \
                    self.backward(prev_traj_distr, traj_info,
                                  eta, omega, nu,
                                  algorithm, m, dual_to_check='eta')

            new_mu, new_sigma = self.forward(traj_distr, traj_info)
            kl_div = self._traj_distr_kl_fcn(new_mu, new_sigma,
                                             traj_distr, prev_traj_distr,
                                             tot=(not self.cons_per_step))
            kl_div_bad = self._traj_distr_kl_fcn(new_mu, new_sigma,
                                                 traj_distr, bad_traj_distr,
                                                 tot=(not self.cons_per_step))
            kl_div_good = self._traj_distr_kl_fcn(new_mu, new_sigma,
                                                  traj_distr, good_traj_distr,
                                                  tot=(not self.cons_per_step))
            self.LOGGER.info("Prev_KL_div(eta): %f | "
                             "Bad_KL_div(nu): %f | "
                             "Good_KL_div(omega): %f"
                             % (kl_div, kl_div_bad, kl_div_good))

            con = kl_div - kl_step
            con_bad = kl_step_bad - kl_div_bad
            con_good = kl_div_good - kl_step_good

            # Convergence check - constraint satisfaction.
            eta_conv = self._conv_prev_check(con, kl_step)
            nu_conv = self._conv_bad_check(con_bad, kl_step_bad)
            omega_conv = self._conv_good_check(con_good, kl_step_good)

            # ALL has converged
            if eta_conv and nu_conv and omega_conv:
                if not self.cons_per_step:
                    self.LOGGER.info("KL_epsilon: %f <= %f, "
                                     "converged iteration %d",
                                     kl_div, kl_step, itr)
                    self.LOGGER.info("KL_nu: %f >= %f, "
                                     "converged iteration %d",
                                     kl_div_bad, kl_step_bad, itr)
                    self.LOGGER.info("KL_omega: %f <= %f, "
                                     "converged iteration %d",
                                     kl_div_good, kl_step_good, itr)
                else:
                    self.LOGGER.info("KL_epsilon: %f <= %f, "
                                     "converged iteration %d",
                                     np.mean(kl_div[:-1]),
                                     np.mean(kl_step[:-1]), itr)
                    self.LOGGER.info("KL_nu: %f >= %f, "
                                     "converged iteration %d",
                                     np.mean(kl_div_bad[:-1]),
                                     np.mean(kl_step_bad[:-1]), itr)
                    self.LOGGER.info("KL_omega: %f <= %f, "
                                     "converged iteration %d",
                                     np.mean(kl_div_good[:-1]),
                                     np.mean(kl_step_good[:-1]), itr)
                break

            # Check convergence between some limits
            if itr > 0:
                if (eta_conv
                    or abs(prev_eta - eta)/eta <= 0.05
                    or abs(eta) <= 0.0001):
                    break_1 = True
                else:
                    break_1 = False
                if (not self.consider_bad
                    or nu_conv
                    or abs(prev_nu - nu)/nu <= 0.05
                    or abs(nu) <= 0.0001):
                    break_2 = True
                else:
                    break_2 = False
                if (not self.consider_good
                    or omega_conv
                    or abs(prev_omega - omega)/omega <= 0.05
                    or abs(omega) <= 0.0001):
                    break_3 = True
                else:
                    break_3 = False
                if break_1 and break_2 and break_3:
                    self.LOGGER.info("Breaking DGD because it has converged or "
                                     "is stuck")
                    break

            if not self.cons_per_step:
                # Choose new eta (bisect bracket or multiply by constant)
                if not eta_conv:
                    self.LOGGER.info("")
                    if con < 0:  # Eta was too big.
                        max_eta = eta
                        geom = np.sqrt(min_eta*max_eta)  # Geometric mean.
                        new_eta = max(geom, 0.1*max_eta)
                        self.LOGGER.info("Modifying ETA | "
                                         "KL: %f <= %f, eta too big, "
                                         "new eta: %.3e",
                                         kl_div, kl_step, new_eta)
                    else:  # Eta was too small.
                        min_eta = eta
                        geom = np.sqrt(min_eta*max_eta)  # Geometric mean.
                        new_eta = min(geom, 10.0*min_eta)
                        self.LOGGER.info("Modifying ETA | "
                                         "KL: %f <= %f, eta too small, "
                                         "new eta: %.3e",
                                         kl_div, kl_step, new_eta)
                else:
                    self.LOGGER.info("NOT modifying ETA")
                    new_eta = eta

                # Choose new nu (bisect bracket or multiply by constant)
                self.LOGGER.info("consider_bad: %s" % self.consider_bad)
                if self.consider_bad and not nu_conv:
                    if con_bad < 0:  # Nu was too big.
                        max_nu = nu
                        geom = np.sqrt(min_nu*max_nu)  # Geometric mean.
                        #new_nu = max(geom, 0.1*max_nu)
                        new_nu = max(geom, 0.025*max_nu)
                        self.LOGGER.info("Modifying NU |"
                                         "KL: %f >= %f, nu too big, "
                                         "new nu: %.3e",
                                         kl_div_bad, kl_step_bad, new_nu)
                    else:  # Nu was too small.
                        min_nu = nu
                        geom = np.sqrt(min_nu*max_nu)  # Geometric mean.
                        #new_nu = min(geom, 10.0*min_nu)
                        new_nu = min(geom, 2.5*min_nu)
                        self.LOGGER.info("Modifying NU |"
                                         "KL: %f >= %f, nu too small, "
                                         "new nu: %.3e",
                                         kl_div_bad, kl_step_bad, new_nu)
                else:
                    self.LOGGER.info("NOT modifying NU")
                    new_nu = nu

                # Choose new omega (bisect bracket or multiply by constant)
                self.LOGGER.info("consider_good: %s" % self.consider_good)
                if self.consider_good and not omega_conv:
                    if con_good < 0:  # Nu was too big.
                        max_omega = omega
                        geom = np.sqrt(min_omega*max_omega)  # Geometric mean.
                        new_omega = max(geom, 0.1*max_omega)
                        self.LOGGER.info("Modifying OMEGA | "
                                         "KL: %f <= %f, omega too big, "
                                         "new omega: %.3e",
                                         kl_div_good, kl_step_good, new_omega)
                    else:  # Nu was too small.
                        min_omega = omega
                        geom = np.sqrt(min_omega*max_omega)  # Geometric mean.
                        new_omega = min(geom, 10.0*min_omega)
                        self.LOGGER.info("Modifying OMEGA | "
                                         "KL: %f <= %f, omega too small, "
                                         "new omega: %.3e",
                                         kl_div_good, kl_step_good, new_omega)
                else:
                    self.LOGGER.info("NOT modifying OMEGA")
                    new_omega = omega

                # Logarithmic mean: log_mean(x,y) = (y - x)/(log(y) - log(x))
                prev_eta = eta
                prev_nu = nu
                prev_omega = omega

                eta = new_eta
                omega = new_omega
                nu = new_nu

            else:
                raise NotImplementedError("Not implemented for cons_per_step")
                # for t in range(T):
                #     if con[t] < 0:  # Eta was too big.
                #         max_eta[t] = eta[t]
                #         geom = np.sqrt(min_eta[t]*max_eta[t])  # Geometric mean.
                #         eta[t] = max(geom, 0.1*max_eta[t])
                #     else:
                #         min_eta[t] = eta[t]
                #         geom = np.sqrt(min_eta[t]*max_eta[t])  # Geometric mean.
                #         eta[t] = min(geom, 10.0*min_eta[t])
                # if itr % 10 == 0:
                #     self.LOGGER.info("avg KL: %f <= %f, avg new eta: %.3e", np.mean(kl_div[:-1]), np.mean(kl_step[:-1]),
                #                  np.mean(eta[:-1]))

            self.LOGGER.info('eta_conv %s: kl_div < epsilon (%f < %f) | %f%%'
                             % (eta_conv, kl_div, kl_step,
                                abs(con*100/kl_step)))
            self.LOGGER.info('nu_conv %s: kl_div > xi (%f > %f) | %f%%'
                             % (nu_conv, kl_div_bad, kl_step_bad,
                                abs(con_bad*100/kl_step_bad)))
            self.LOGGER.info('omega_conv %s: kl_div < chi (%f < %f) | %f%%'
                             % (omega_conv, kl_div_good, kl_step_good,
                                abs(con_good*100/kl_step_good)))

        self.LOGGER.info('_'*40)
        self.LOGGER.info('FINAL VALUES at itr %d || '
                         'eta: %f | nu: %f | omega: %f'
                         % (itr, eta, nu, omega))
        self.LOGGER.info('eta_conv %s: kl_div < epsilon (%f < %f) | %f%%'
                         % (eta_conv, kl_div, kl_step, abs(con*100/kl_step)))
        self.LOGGER.info('nu_conv %s: kl_div > xi (%f > %f) | %f%%'
                         % (nu_conv, kl_div_bad, kl_step_bad,
                            abs(con_bad*100/kl_step_bad)))
        self.LOGGER.info('omega_conv %s: kl_div < chi (%f < %f) | %f%%'
                         % (omega_conv, kl_div_good, kl_step_good,
                            abs(con_good*100/kl_step_good)))

        if itr + 1 == max_itr:
            self.LOGGER.info("After %d iterations for ETA, NU, OMEGA,"
                             "the constraints have not been satisfied.", itr)

        if not self.consider_bad:
            nu_conv = True

        if not self.consider_good:
            omega_conv = True

        return traj_distr, (eta, nu, omega), (eta_conv, nu_conv, omega_conv)

    def lagrangian_function(self, duals, algorithm, m):
        input("LAGRANGIAN FUNCT")
        self.LOGGER.info(duals)
        return np.sum(duals)

    def lagrangian_gradient(self, duals, algorithm, m):
        input("LAGRANGIAN GRADIENT")
        self.LOGGER.info(duals)
        return np.sum(duals)
