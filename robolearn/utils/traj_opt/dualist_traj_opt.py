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
from scipy.optimize import minimize, minimize_scalar

from robolearn.utils.traj_opt.traj_opt import TrajOpt
from robolearn.utils.traj_opt.config import default_traj_opt_mdreps_hyperparams
from robolearn.utils.traj_opt.traj_opt_utils import traj_distr_kl, traj_distr_kl_alt
from robolearn.utils.print_utils import ProgressBar

print_DGD_log = False

MAX_ALL_DGD = 20
DGD_MAX_ITER = 50
DGD_MAX_LS_ITER = 20
DGD_MAX_GD_ITER = 50  #500 #200

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

        self.logger = logging.getLogger(__name__)

    # TODO - Add arg and return spec on this function.
    def update(self, m, algorithm):
        """
        Run dual gradient descent to optimize trajectories.
        It returns (optimized) new trajectory and eta.
        :param m: Condition number.
        :param algorithm: GPS algorithm to get info
        :param a: Linear Act ID (Multiple local policies).
        :return: traj_distr,
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

        """
        import matplotlib.pyplot as plt
        T = algorithm.T
        traj_info = algorithm.cur[m].traj_info

        sample_list = algorithm.cur[m].sample_list
        samples = sample_list.get_states()
        # samples = sample_list.get_actions()
        state_to_plot = 6
        mu_to_plot = 6

        for ss in range(samples.shape[0]):
            plt.plot(np.array(range(T)), samples[ss, :, state_to_plot],
                     color='brown')


        # INITIAL
        traj_distr = algorithm.cur[m].traj_distr
        mu0, sigma0 = self.forward(traj_distr, traj_info)
        plt.plot(mu0[:, mu_to_plot], color='black', label='initial')
        plt.fill_between(np.array(range(T)),
                         mu0[:, mu_to_plot] - 3*sigma0[:, mu_to_plot, mu_to_plot],
                         mu0[:, mu_to_plot] + 3*sigma0[:, mu_to_plot, mu_to_plot],
                         alpha=0.3, color='black')
        print("AYUDAMEEEEE1:", sigma0[:, state_to_plot, state_to_plot])
        print("AYUDAMEEEEE2:", sigma0[:, state_to_plot+1, state_to_plot+1])
        print("AYUDAMEEEEE3:", sigma0[:, state_to_plot+2, state_to_plot+3])


        print('%'*30)
        print('\n'*5)

        # Eta
        traj_distr1, duals, convs = \
            self._gradient_descent_all(algorithm, m, eta, 0, 0,
                                       opt_eta=True,
                                       opt_nu=False,
                                       opt_omega=False)
        # traj_distr1, duals, convs = \
        #     self._adam_all(algorithm, m, eta, 0, 0,
        #                    opt_eta=True, opt_nu=False, opt_omega=False, alpha=0.2)
        mu1, sigma1 = self.forward(traj_distr1, traj_info)
        plt.plot(mu1[:, mu_to_plot], color='blue', label='only_eta')
        # plt.fill_between(np.array(range(T)),
        #                  mu1[:, mu_to_plot] - 3*sigma1[:, mu_to_plot, mu_to_plot],
        #                  mu1[:, mu_to_plot] + 3*sigma1[:, mu_to_plot, mu_to_plot],
        #                  alpha=0.3, color='blue')

        # eta = duals[0]
        # nu = duals[1]
        # omega = duals[2]
        # eta_conv = convs[0]
        # nu_conv = convs[1]
        # omega_conv = convs[2]

        # print('%'*30)
        # print('\n'*8)
        # # DUAL
        # # traj_distr2, duals, convs = \
        # #     self._gradient_descent_all(algorithm, m, eta, nu, omega,
        # #                                opt_eta=True,
        # #                                opt_nu=True,
        # #                                opt_omega=True)
        # traj_distr2, duals, convs = \
        #     self._adam_all(algorithm, m, eta, nu, omega,
        #                    opt_eta=True, opt_nu=True, opt_omega=True, alpha=0.2)
        # mu2, sigma2 = self.forward(traj_distr2, traj_info)
        # plt.plot(mu2[:, mu_to_plot], color='purple', label='dual')
        # # plt.fill_between(np.array(range(T)),
        # #                  mu2[:, mu_to_plot] - 3*sigma2[:, mu_to_plot, mu_to_plot],
        # #                  mu2[:, mu_to_plot] + 3*sigma2[:, mu_to_plot, mu_to_plot],
        # #                  alpha=0.3, color='purple')

        print('%'*30)
        print('\n'*8)
        # Bad
        traj_distr3, duals, convs = \
            self._gradient_descent_all(algorithm, m, 0, nu, 0,
                                       opt_eta=False,
                                       opt_nu=True,
                                       opt_omega=False)
        # traj_distr3, duals, convs = \
        #     self._adam_all(algorithm, m, 0, nu, 0,
        #                    opt_eta=False, opt_nu=True, opt_omega=False, alpha=0.2)
        mu3, sigma3 = self.forward(traj_distr3, traj_info)
        plt.plot(mu3[:, mu_to_plot], color='red', label='bad')
        # plt.fill_between(np.array(range(T)),
        #                  mu3[:, mu_to_plot] - 3*sigma3[:, mu_to_plot, mu_to_plot],
        #                  mu3[:, mu_to_plot] + 3*sigma3[:, mu_to_plot, mu_to_plot],
        #                  alpha=0.3, color='red')

        # print('%'*30)
        # print('\n'*8)
        # # Good
        # # traj_distr4, duals, convs = \
        # #     self._gradient_descent_all(algorithm, m, 0, 0, omega,
        # #                                opt_eta=False,
        # #                                opt_nu=False,
        # #                                opt_omega=True)
        # traj_distr4, duals, convs = \
        #     self._adam_all(algorithm, m, 0, 0, omega,
        #                    opt_eta=False, opt_nu=False, opt_omega=True, alpha=0.2)
        # mu4, sigma4 = self.forward(traj_distr4, traj_info)
        # plt.plot(mu4[:, mu_to_plot], color='green', label='good')
        # # plt.fill_between(np.array(range(T)),
        # #                  mu4[:, mu_to_plot] - 3*sigma4[:, mu_to_plot, mu_to_plot],
        # #                  mu4[:, mu_to_plot] + 3*sigma4[:, mu_to_plot, mu_to_plot],
        # #                  alpha=0.3, color='green')

        # print('%'*30)
        # print('\n'*8)
        # # Bad+step
        # # traj_distr5, duals, convs = \
        # #     self._gradient_descent_all(algorithm, m, eta, nu, 0,
        # #                                opt_eta=True,
        # #                                opt_nu=True,
        # #                                opt_omega=False)
        traj_distr5, duals, convs = \
            self._adam_all(algorithm, m, eta, nu, 0,
                           opt_eta=True, opt_nu=True, opt_omega=False, alpha=0.2)
        mu5, sigma5 = self.forward(traj_distr5, traj_info)
        plt.plot(mu5[:, mu_to_plot], color='magenta', label='bad-2')
        # plt.fill_between(np.array(range(T)),
        #                  mu5[:, mu_to_plot] - 3*sigma5[:, mu_to_plot, mu_to_plot],
        #                  mu5[:, mu_to_plot] + 3*sigma5[:, mu_to_plot, mu_to_plot],
        #                  alpha=0.3, color='magenta')

        # print('%'*30)
        # print('\n'*8)
        # # Good+step
        # # traj_distr7, duals, convs = \
        # #     self._gradient_descent_all(algorithm, m, eta, 0, omega,
        # #                                opt_eta=True,
        # #                                opt_nu=False,
        # #                                opt_omega=True)
        # traj_distr7, duals, convs = \
        #     self._adam_all(algorithm, m, eta, 0, omega,
        #                    opt_eta=True, opt_nu=False, opt_omega=True, alpha=0.2)
        # mu6, sigma6 = self.forward(traj_distr7, traj_info)
        # plt.plot(mu6[:, mu_to_plot], color='grey', label='good-2')
        # # plt.fill_between(np.array(range(T)),
        # #                  mu6[:, mu_to_plot] - 3*sigma6[:, mu_to_plot, mu_to_plot],
        # #                  mu6[:, mu_to_plot] + 3*sigma6[:, mu_to_plot, mu_to_plot],
        # #                  alpha=0.3, color='grey')

        traj_distr8 = copy.deepcopy(traj_distr1)
        traj_distr8.K = np.mean( np.array([traj_distr1.K, traj_distr3.K]), axis=0)
        traj_distr8.k = np.mean( np.array([traj_distr1.k, traj_distr3.k]), axis=0)

        mu8, sigma8 = self.forward(traj_distr8, traj_info)
        plt.plot(mu8[:, mu_to_plot], color='orange', label='AVGbad-eta')
        # plt.fill_between(np.array(range(T)),
        #                  mu8[:, mu_to_plot] - 3*sigma8[:, mu_to_plot, mu_to_plot],
        #                  mu8[:, mu_to_plot] + 3*sigma8[:, mu_to_plot, mu_to_plot],
        #                  alpha=0.3, color='orange')

        plt.legend(loc='center left', shadow=False)

        plt.show(block=True)
        """

        """
        # Minimization
        min_eta = self._hyperparams['min_eta']
        max_eta = self._hyperparams['max_eta']
        min_nu = self._hyperparams['min_nu']
        max_nu = self._hyperparams['max_nu']
        min_omega = self._hyperparams['min_omega']
        max_omega = self._hyperparams['max_omega']

        x0 = np.array([eta, nu, omega])
        result = minimize(self.lagrangian_function, x0,
                          args=(algorithm, m, True, self.consider_bad, self.consider_good),
                          method='L-BFGS-B',
                          # jac=self.lagrangian_gradient,
                          bounds=[[min_eta, max_eta],
                                  [min_nu, max_nu],
                                  [min_omega, max_omega]],
                          tol=None, callback=None,
                          options={'disp': None, 'maxls': 20,
                                   'iprint': -1, 'gtol': 1e-05,
                                   'eps': 1e-08, 'maxiter': 15000,
                                   'ftol': 2.220446049250313e-09,
                                   'maxcor': 10, 'maxfun': 15000})
        eta = result.x[0]
        nu = result.x[1] if self.consider_bad else 0
        omega = result.x[2] if self.consider_bad else 0
        traj_distr, duals, convs = \
            self._gradient_descent_all(algorithm, m, eta, nu, omega,
                                       opt_eta=False,
                                       opt_nu=False,
                                       opt_omega=False)
        eta = duals[0]
        nu = duals[1]
        omega = duals[2]
        traj_distr, duals, convs = \
            self._adam_all(algorithm, m, eta, nu, omega,
                           opt_eta=True, opt_nu=False, opt_omega=False)
            # self._adam_all(algorithm, m, eta, nu, omega,
            #                opt_eta=True, opt_nu=self.consider_bad,
            #                opt_omega=self.consider_good)
        """

        """
        # Minimization
        min_eta = self._hyperparams['min_eta']
        max_eta = self._hyperparams['max_eta']
        min_nu = self._hyperparams['min_nu']
        max_nu = self._hyperparams['max_nu']
        min_omega = self._hyperparams['min_omega']
        max_omega = self._hyperparams['max_omega']

        x0 = np.array([eta, nu, omega])
        result = minimize(self.lagrangian_function, x0,
                          args=(algorithm, m, False, self.consider_bad, self.consider_good),
                          method='L-BFGS-B',
                          # jac=self.lagrangian_gradient,
                          bounds=[[min_eta, max_eta],
                                  [min_nu, max_nu],
                                  [min_omega, max_omega]],
                          tol=None, callback=None,
                          options={'disp': None, 'maxls': 20,
                                   'iprint': -1, 'gtol': 1e-05,
                                   'eps': 1e-08, 'maxiter': 15000,
                                   'ftol': 2.220446049250313e-09,
                                   'maxcor': 10, 'maxfun': 15000})

        # eta = result.x[0]
        nu = result.x[1] if self.consider_bad else 0
        omega = result.x[2] if self.consider_bad else 0
        traj_distr, duals, convs = \
            self._gradient_descent_all(algorithm, m, eta, nu, omega,
                                       opt_eta=False,
                                       opt_nu=False,
                                       opt_omega=False)
        eta = duals[0]
        nu = duals[1]
        omega = duals[2]

        traj_distr, duals, convs = \
            self._gradient_descent_all(algorithm, m, eta, nu, omega,
                                       opt_eta=True,
                                       opt_nu=False,
                                       opt_omega=False)

        eta_conv = convs[0]
        if not eta_conv:
            traj_distr, duals, convs = \
                self._adam_all(algorithm, m, eta, nu, omega,
                               opt_eta=True, opt_nu=False, opt_omega=False)
        """


        # Minimization
        min_eta = self._hyperparams['min_eta']
        max_eta = self._hyperparams['max_eta']
        min_nu = self._hyperparams['min_nu']
        max_nu = self._hyperparams['max_nu']
        min_omega = self._hyperparams['min_omega']
        max_omega = self._hyperparams['max_omega']

        x0 = np.array([eta, nu, omega])
        result = minimize(self.fcn_to_optimize, x0,
                          args=(algorithm, m, False, self.consider_bad, self.consider_good),
                          method='L-BFGS-B',
                          jac=self.grad_to_optimize,
                          bounds=[[min_eta, max_eta],
                                  [min_nu, max_nu],
                                  [min_omega, max_omega]],
                          tol=None, callback=None,
                          options={'disp': None, 'maxls': 20,
                                   'iprint': -1, 'gtol': 1e-05,
                                   'eps': 1e-08, 'maxiter': 15000,
                                   'ftol': 2.220446049250313e-09,
                                   'maxcor': 10, 'maxfun': 15000})

        # eta = result.x[0]
        nu = result.x[1] if self.consider_bad else 0
        omega = result.x[2] if self.consider_bad else 0
        traj_distr, duals, convs = \
            self._gradient_descent_all(algorithm, m, eta, nu, omega,
                                       opt_eta=False,
                                       opt_nu=False,
                                       opt_omega=False)
        eta = duals[0]
        nu = duals[1]
        omega = duals[2]

        print('\n\n')
        print('\n\n')
        print('fadsfdsfadsf')
        print('ADHASKHDKJHASJKHDJKHSAJKGDKASKJHDKJHASJKHDKJHASJKDAS')
        print('CONVS are:', convs)
        print('\n\n')
        print('\n\n')



        # self._adam_all(algorithm, m, eta, nu, omega,
        #                opt_eta=True, opt_nu=self.consider_bad,
        #                opt_omega=self.consider_good)



        # # Dual Gradient Descent
        # traj_distr, duals, convs = \
        #     self._gradient_descent_all(algorithm, m, eta, nu, omega,
        #                                opt_eta=True,
        #                                opt_nu=self.consider_bad,
        #                                opt_omega=self.consider_good)
        # eta = duals[0]
        # nu = duals[1]
        # omega = duals[2]
        # eta_conv = convs[0]
        # nu_conv = convs[1]
        # omega_conv = convs[2]
        #
        # if eta_conv and nu_conv and omega_conv:
        #     self.logger.info("ALL DGD has converged.")
        # else:
        #     self.logger.info("ALL DGD has NOT converged.")
        #     if not eta_conv:
        #         self.logger.warning("Final KL_step divergence after "
        #                             "DGD convergence is too high.")
        #
        #     if not nu_conv:
        #         self.logger.warning("Final KL_bad divergence after "
        #                             "DGD convergence is too low.")
        #
        #     if not omega_conv:
        #         self.logger.warning("Final KL_good divergence after "
        #                             "DGD convergence is too high.")


        # if not eta_conv:
        #     traj_distr, duals, convs = \
        #         self._adam_all(algorithm, m, eta, nu, omega,
        #                        opt_eta=True, opt_nu=False, opt_omega=False)


        # # # If it did not converge, use ADAM for some steps
        # # if not(eta_conv and nu_conv and omega_conv):
        # #     self.logger.warning('')
        # #     self.logger.warning("Refinement with Adam")
        # #     traj_distr, duals, convs = \
        # #         self._adam_all(algorithm, m, eta, nu, omega)
        # eta = duals[0]
        # nu = duals[1]
        # omega = duals[2]


        # TODO: DO WE NEED TO DO A BACKWARD PASS TO GET THE TRAJ_DISTR ?????
        # traj_distr, eta, nu, omega = \
        #     self.backward(prev_traj_distr, good_traj_distr, bad_traj_distr,
        #                   traj_info, eta, nu, omega, algorithm, m,
        #                   dual_to_check='omega')

        return traj_distr, eta, nu, omega

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

    def backward(self, prev_traj_distr, good_traj_distr, bad_traj_distr,
                 traj_info, eta, nu, omega, algorithm, m,
                 dual_to_check='eta'):
        """
        Perform LQR backward pass. This computes a new linear Gaussian policy
        object.
        Args:
            prev_traj_distr: A linear Gaussian policy object from
                previous iteration.
            traj_info: A TrajectoryInfo object.
            eta: Dual variable prev_traj, KL(p||q) <= epsilon.
            nu: Dual variable bad_traj, KL(p||b) >= xi.
            omega: Dual variable good_traj, KL(p||g) <= chi.
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
            fCm, fcv = compute_cost_fcn(m, eta, nu, omega,
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
                    K_term = Qtt[t, idx_u, idx_x]  # Qxu
                else:
                    # TODO: CHECK THAT THIS OP IS OK!!!
                    inv_term = (1.0 / (eta[t] + omega[t] - nu[t])) \
                               * Qtt[t, idx_u, idx_u] \
                               + (eta[t]/(eta[t] + omega[t] - nu[t])) \
                                 * prev_traj_distr.inv_pol_covar[t] \
                               + (omega[t]/(eta[t] + omega[t] - nu[t])) \
                                * good_traj_distr.inv_pol_covar[t] \
                               - (nu[t]/(eta[t] + omega[t] - nu[t])) \
                                * bad_traj_distr.inv_pol_covar[t]
                    k_term = (1.0 / (eta[t] + omega[t] - nu[t])) \
                             * Qt[t, idx_u] \
                             - (eta[t]/(eta[t] + omega[t] - nu[t])) \
                                * prev_traj_distr.inv_pol_covar[t].dot(prev_traj_distr.k[t]) \
                             - (omega[t]/(eta[t] + omega[t] - nu[t])) \
                               * good_traj_distr.inv_pol_covar[t].dot(good_traj_distr.k[t]) \
                             + (nu[t]/(eta[t] + omega[t] - nu[t])) \
                               * bad_traj_distr.inv_pol_covar[t].dot(bad_traj_distr.k[t])
                    K_term = (1.0 / (eta[t] + omega[t] - nu[t])) \
                             * Qtt[t, idx_u, idx_x] \
                             - (eta[t]/(eta[t] + omega[t] - nu[t])) \
                                * prev_traj_distr.inv_pol_covar[t].dot(prev_traj_distr.K[t]) \
                             - (omega[t]/(eta[t] + omega[t] - nu[t])) \
                               * good_traj_distr.inv_pol_covar[t].dot(good_traj_distr.K[t]) \
                             + (nu[t]/(eta[t] + omega[t] - nu[t])) \
                               * bad_traj_distr.inv_pol_covar[t].dot(bad_traj_distr.K[t])

                # Compute Cholesky decomposition of Q function action component.
                try:
                    U = sp.linalg.cholesky(inv_term)
                    L = U.T
                except LinAlgError as e:
                    # Error thrown when Qtt[idx_u, idx_u] is not
                    # symmetric positive definite (SPD).
                    self.logger.error('LinAlgError: %s', e)
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

            # Increment the dual_to_check on non-SPD Q-function.
            if fail:
                if not self.cons_per_step:
                    if dual_to_check == 'eta':
                        old_eta = eta
                        eta = eta0 + del_
                        self.logger.info('Non-SPD Q: Increasing eta: %f -> %f',
                                         old_eta, eta)
                        del_ *= 2  # Increase del_ exponentially on failure.
                    elif dual_to_check == 'nu':
                        old_eta = eta
                        eta = eta0 + del_
                        self.logger.info('Non-SPD Q: Increasing eta: %f -> %f',
                                         old_eta, eta)
                        del_ *= 2  # Increase del_ exponentially on failure.
                        old_nu = nu
                        nu = nu0 - del_bad_
                        self.logger.info('Non-SPD Q: Decreasing nu: %f -> %f',
                                         old_nu, nu)
                        del_bad_ *= 2  # Decrease del_ exponentially on failure.
                        old_omega = omega
                        omega = omega0 + del_good_
                        self.logger.info('Non-SPD Q: Increasing omega: %f -> %f',
                                         old_omega, omega)
                        del_good_ *= 2  # Decrease del_ exponentially on failure.
                    elif dual_to_check == 'nu2':
                        old_eta = eta
                        eta = eta0 + del_
                        self.logger.info('Non-SPD Q: Increasing eta: %f -> %f',
                                         old_eta, eta)
                        del_ *= 2  # Increase del_ exponentially on failure.
                        old_nu = nu
                        nu = nu0 - del_bad_
                        self.logger.info('Non-SPD Q: Decreasing nu: %f -> %f',
                                         old_nu, nu)
                        del_bad_ *= 2  # Decrease del_ exponentially on failure.
                        #old_omega = omega
                        #omega = omega0 + del_good_
                        # self.logger.info('Non-SPD Q: Increasing omega: %f -> %f',
                        #                  old_omega, omega)
                        #del_good_ *= 2  # Decrease del_ exponentially on failure.
                    elif dual_to_check == 'omega':
                        old_omega = omega
                        omega = omega0 + del_good_
                        self.logger.info('Non-SPD Q: Increasing omega: %f -> %f',
                                         old_omega, omega)
                        del_good_ *= 2  # Increase del_ exponentially on failure.
                    else:
                        raise ValueError("Wrong dual_to_check option %s"
                                         % dual_to_check)

                else:
                    if dual_to_check == 'eta':
                        old_eta = eta[fail]
                        eta[fail] = eta0[fail] + del_[fail]
                        self.logger.info('Non-SPD Q: Increasing eta[%d]: %f -> %f',
                                         fail, old_eta, eta[fail])
                        del_[fail] *= 2  # Increase del_ exponentially on failure.
                    elif dual_to_check == 'nu':
                        old_eta = eta[fail]
                        eta[fail] = eta0[fail] + del_[fail]
                        self.logger.info('Increasing eta[%d]: %f -> %f',
                                         fail, old_eta, eta[fail])
                        del_[fail] *= 2  # Increase del_ exponentially on failure.
                        # old_nu = nu[fail]
                        # nu[fail] = nu0[fail] - del_bad_[fail]
                        # self.logger.info('Decreasing nu[%d]: %f -> %f',
                        #                  fail, old_nu, nu[fail])
                        # del_bad_[fail] *= 2  # Increase del_ exponentially on failure.
                        old_omega = omega[fail]
                        omega[fail] = omega0[fail] + del_good_[fail]
                        self.logger.info('Increasing omega[%d]: %f -> %f',
                                         fail, old_omega, omega[fail])
                        del_good_[fail] *= 2  # Increase del_ exponentially on failure.
                    elif dual_to_check == 'omega':
                        old_omega = omega[fail]
                        omega[fail] = omega0[fail] + del_good_[fail]
                        self.logger.info('Increasing omega[%d]: %f -> %f',
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
        return traj_distr, eta, nu, omega

    def _conv_prev_check(self, con, kl_step):
        """
        Function that checks whether ETA dual gradient descent has converged.
        """
        step_tol = self._hyperparams['step_tol']
        if self.cons_per_step:
            return all([abs(con[t]) < (step_tol*kl_step[t])
                        for t in range(con.size)])

        self.logger.info("Conv kl_step:%s | "
                         "abs(con) < %.1f*kl_step | "
                         "abs(%f) < %f | %f%%"
                         % (abs(con) < step_tol * kl_step,
                            step_tol,
                            con, step_tol*kl_step, abs(con*100/kl_step)))
        return abs(con) < step_tol * kl_step

    def _conv_good_check(self, con_good, kl_good):
        """
        Function that checks whether OMEGA dual gradient descent has converged.
        """
        if not self.consider_good:
            self.logger.info("NOT CONSIDERING kl_good | "
                             "Setting conv_good to True)")
            return True

        good_tol = self._hyperparams['good_tol']
        if self.cons_per_step:
            return all([abs(con_good[t]) < (good_tol*kl_good[t])
                        for t in range(con_good.size)])

        self.logger.info("Conv kl_good:%s | "
                         "abs(con_good) < %.1f*kl_good | "
                         "abs(%f) < %f | %f%%"
                         % (abs(con_good) < good_tol * kl_good,
                            good_tol,
                            con_good,
                            good_tol*kl_good,
                            abs(con_good*100/kl_good)))
        return abs(con_good) < good_tol * kl_good

    def _conv_bad_check(self, con_bad, kl_bad):
        """
        Function that checks whether NU dual gradient descent has converged.
        """
        if not self.consider_bad:
            self.logger.info("NOT CONSIDERING kl_bad | "
                             "Setting conv_bad to True)")
            return True

        bad_tol = self._hyperparams['bad_tol']
        if self.cons_per_step:
            return all([abs(con_bad[t]) < (-bad_tol*kl_bad[t]) for t in range(con_bad.size)])
            # return all([con_bad[t] <= 0 for t in range(con_bad.size)])

        self.logger.info("Conv kl_bad:%s | abs(con_bad) <= %.1f*kl_bad | abs(%f) < %f | %f%%"
                         % (abs(con_bad) < bad_tol*kl_bad,
                            bad_tol,
                            con_bad,
                            bad_tol*kl_bad,
                            abs(con_bad)*100/kl_bad))

        # return con_bad < -bad_tol * kl_bad
        return abs(con_bad) < bad_tol * kl_bad
        # return con_bad <= 0

    def _gradient_descent_all(self, algorithm, m, eta, nu, omega,
                              dual_to_check='eta', opt_eta=True,
                              opt_nu=True, opt_omega=True):

        T = algorithm.T
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

        # Less iterations if cons_per_step=True
        max_itr = (DGD_MAX_LS_ITER if self.cons_per_step else
                   DGD_MAX_ITER)



        step_mult = algorithm.cur[m].step_mult
        bad_step_mult = algorithm.cur[m].bad_step_mult
        good_step_mult = algorithm.cur[m].good_step_mult

        # Set KL-divergences using their step multipliers.
        kl_step = algorithm.base_kl_step * step_mult
        kl_bad = algorithm.base_kl_bad * bad_step_mult
        kl_good = algorithm.base_kl_good * good_step_mult

        if not self.cons_per_step:
            kl_step *= T
            kl_bad *= T
            kl_good *= T
        else:
            if not isinstance(kl_step, (np.ndarray, list)):
                self.logger.warning('KL_step is not iterable. Converting it')
                kl_step = np.ones(T)*kl_step
            if not isinstance(kl_bad, (np.ndarray, list)):
                self.logger.warning('KL_bad is not iterable. Converting it')
                kl_bad = np.ones(T)*kl_bad
            if not isinstance(kl_good, (np.ndarray, list)):
                self.logger.warning('KL_good is not iterable. Converting it')
                kl_good = np.ones(T)*kl_good

        # We assume at min_eta, kl_div > kl_step, opposite for max_eta.
        if not self.cons_per_step:
            min_eta = self._hyperparams['min_eta']
            max_eta = self._hyperparams['max_eta']
            min_nu = self._hyperparams['min_nu']
            max_nu = self._hyperparams['max_nu']
            min_omega = self._hyperparams['min_omega']
            max_omega = self._hyperparams['max_omega']
        else:
            min_eta = np.ones(T) * self._hyperparams['min_eta']
            max_eta = np.ones(T) * self._hyperparams['max_eta']
            min_nu = np.ones(T) * self._hyperparams['min_nu']
            max_nu = np.ones(T) * self._hyperparams['max_nu']
            min_omega = np.ones(T) * self._hyperparams['min_omega']
            max_omega = np.ones(T) * self._hyperparams['max_omega']



        # We assume at min_eta, kl_div > kl_step, opposite for max_eta.
        if not self.cons_per_step:
            self.logger.info('_'*60)
            self.logger.info("Running DGD for traj[%d] | "
                             "eta: %4f, nu: %4f, omega: %4f",
                             m, eta, nu, omega)
            self.logger.info('_'*60)
        else:
            self.logger.info('_'*60)
            self.logger.info("Running DGD for trajectory %d,"
                             "avg eta: %f, avg nu: %f, avg omega: %f",
                             m, np.mean(eta[:-1]), np.mean(nu[:-1]),
                             np.mean(omega[:-1]))
            self.logger.info('_'*60)

        # Run ALL GD
        for itr in range(max_itr):
            self.logger.info("-"*15)
            if not self.cons_per_step:
                self.logger.info("ALL DGD | iter %d| Current dual values: "
                                 "eta %.2e, nu %.2e, omega %.2e"
                                 % (itr, eta, nu, omega))
            else:
                self.logger.info("ALL DGD | iter %d| Current dual values: "
                                 "avg_eta %.2r, avg_nu %.2r, avg_omega %.2r"
                                 % (itr, np.mean(eta[:-1]), np.mean(nu[:-1]),
                                    np.mean(omega[:-1])))

            if not self.cons_per_step:
                if opt_eta:
                    self.logger.info("ALL DGD iteration %d| "
                                     "ETA bracket: (%.2e , %.2e , %.2e)",
                                     itr, min_eta, eta, max_eta)
                if opt_nu:
                    self.logger.info("ALL DGD iteration %d| "
                                     "NU bracket: (%.2e , %.2e , %.2e)",
                                     itr, min_nu, nu, max_nu)
                if opt_omega:
                    self.logger.info("ALL DGD iteration %d| "
                                     "OMEGA bracket: (%.2e , %.2e , %.2e)",
                                     itr, min_omega, omega, max_omega)


            # Run Bwd pass to optimize the traj distribution
            traj_distr, eta, nu, omega = \
                self.backward(prev_traj_distr, good_traj_distr,
                              bad_traj_distr, traj_info,
                              eta, nu, omega,
                              algorithm, m, dual_to_check=dual_to_check)

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
            kl_div_bad = self._traj_distr_kl_fcn(mu_to_check, sigma_to_check,
                                                 traj_distr, bad_traj_distr,
                                                 tot=(not self.cons_per_step))
            kl_div_good = self._traj_distr_kl_fcn(mu_to_check, sigma_to_check,
                                                  traj_distr, good_traj_distr,
                                                  tot=(not self.cons_per_step))

            con = kl_div - kl_step  # KL - epsilon
            con_bad = kl_bad - kl_div_bad  # xi - KL
            con_good = kl_div_good - kl_good  # KL - chi

            # Convergence check - constraint satisfaction.
            eta_conv = self._conv_prev_check(con, kl_step)
            nu_conv = self._conv_bad_check(con_bad, kl_bad)
            omega_conv = self._conv_good_check(con_good, kl_good)

            # ALL has converged
            if (opt_eta and eta_conv) and (opt_nu and nu_conv) and (opt_omega and omega_conv):
                if not self.cons_per_step:
                    if opt_eta:
                        self.logger.info("KL_epsilon: %f <= %f, "
                                         "converged iteration %d",
                                         kl_div, kl_step, itr)
                    if opt_nu:
                        self.logger.info("KL_nu: %f >= %f, "
                                         "converged iteration %d",
                                         kl_div_bad, kl_bad, itr)
                    if opt_omega:
                        self.logger.info("KL_omega: %f <= %f, "
                                         "converged iteration %d",
                                         kl_div_good, kl_good, itr)
                else:
                    if opt_eta:
                        self.logger.info("KL_epsilon: %f <= %f, "
                                         "converged iteration %d",
                                         np.mean(kl_div[:-1]),
                                         np.mean(kl_step[:-1]), itr)
                    if opt_nu:
                        self.logger.info("KL_nu: %f >= %f, "
                                         "converged iteration %d",
                                         np.mean(kl_div_bad[:-1]),
                                         np.mean(kl_bad[:-1]), itr)
                    if opt_omega:
                        self.logger.info("KL_omega: %f <= %f, "
                                         "converged iteration %d",
                                         np.mean(kl_div_good[:-1]),
                                         np.mean(kl_good[:-1]), itr)
                break

            # Check convergence between some limits
            if itr > 0:
                if opt_eta:
                    if (eta_conv
                        or np.all(abs(prev_eta - eta)/eta <= 0.05)):
                        # or np.all(eta <= min_eta)):
                        break_1 = True
                    else:
                        break_1 = False
                else:
                    break_1 = True

                if opt_nu:
                    if (nu_conv
                        or np.all(abs(prev_nu - nu)/nu <= 0.05)):
                        # or np.all(nu <= min_nu)):
                        break_2 = True
                    else:
                        break_2 = False
                else:
                    break_2 = True

                if opt_omega:
                    if (omega_conv
                        or np.all(abs(prev_omega - omega)/omega <= 0.05)):
                        # or np.all(omega <= min_omega)):
                        break_3 = True
                    else:
                        break_3 = False
                else:
                    break_3 = True

                self.logger.info('Eta change: %r | '
                                 'Nu change: %r | '
                                 'Omega change: %r'
                                 % (np.mean(abs(prev_eta - eta)/(eta+1e-6)),
                                    np.mean(abs(prev_nu - nu)/(nu+1e-6)),
                                    np.mean(abs(prev_omega - omega)/(omega+1e-6))))
                self.logger.info('Eta conv: %r | '
                                 'Nu conv: %r | '
                                 'Omega conv: %r'
                                 % (eta_conv,
                                    nu_conv,
                                    omega_conv))

                if break_1 and break_2 and break_3:
                    self.logger.info("Breaking DGD because it has converged or "
                                     "is stuck")
                    break

            if not self.cons_per_step:
                # Choose new eta (bisect bracket or multiply by constant)
                if opt_eta and not eta_conv:
                    self.logger.info("")
                    if con < 0:  # Eta was too big.
                        max_eta = eta
                        geom = np.sqrt(min_eta*max_eta)  # Geometric mean.
                        new_eta = max(geom, 0.1*max_eta)
                        self.logger.info("Modifying ETA | "
                                         "KL: %f <= %f, eta too big, "
                                         "new eta: %.3e",
                                         kl_div, kl_step, new_eta)
                    else:  # Eta was too small.
                        min_eta = eta
                        geom = np.sqrt(min_eta*max_eta)  # Geometric mean.
                        new_eta = min(geom, 10.0*min_eta)
                        self.logger.info("Modifying ETA | "
                                         "KL: %f <= %f, eta too small, "
                                         "new eta: %.3e",
                                         kl_div, kl_step, new_eta)
                else:
                    self.logger.info("NOT modifying ETA")
                    new_eta = eta

                # Choose new nu (bisect bracket or multiply by constant)
                self.logger.info("consider_bad: %s" % self.consider_bad)
                print('YAPEEE', self.consider_bad, opt_nu, nu_conv)
                if self.consider_bad and opt_nu and not nu_conv:
                    if con_bad < 0:  # Nu was too big.
                        max_nu = nu
                        geom = np.sqrt(min_nu*max_nu)  # Geometric mean.
                        #new_nu = max(geom, 0.1*max_nu)
                        new_nu = max(geom, 0.025*max_nu)
                        self.logger.info("Modifying NU |"
                                         "KL: %f >= %f, nu too big, "
                                         "new nu: %.3e",
                                         kl_div_bad, kl_bad, new_nu)
                    else:  # Nu was too small.
                        min_nu = nu
                        geom = np.sqrt(min_nu*max_nu)  # Geometric mean.
                        #new_nu = min(geom, 10.0*min_nu)
                        new_nu = min(geom, 2.5*min_nu)
                        self.logger.info("Modifying NU |"
                                         "KL: %f >= %f, nu too small, "
                                         "new nu: %.3e",
                                         kl_div_bad, kl_bad, new_nu)
                else:
                    self.logger.info("NOT modifying NU")
                    new_nu = nu

                # Choose new omega (bisect bracket or multiply by constant)
                self.logger.info("consider_good: %s" % self.consider_good)
                if self.consider_good and opt_omega and not omega_conv:
                    if con_good < 0:  # Nu was too big.
                        max_omega = omega
                        geom = np.sqrt(min_omega*max_omega)  # Geometric mean.
                        new_omega = max(geom, 0.1*max_omega)
                        self.logger.info("Modifying OMEGA | "
                                         "KL: %f <= %f, omega too big, "
                                         "new omega: %.3e",
                                         kl_div_good, kl_good, new_omega)
                    else:  # Nu was too small.
                        min_omega = omega
                        geom = np.sqrt(min_omega*max_omega)  # Geometric mean.
                        new_omega = min(geom, 10.0*min_omega)
                        self.logger.info("Modifying OMEGA | "
                                         "KL: %f <= %f, omega too small, "
                                         "new omega: %.3e",
                                         kl_div_good, kl_good, new_omega)
                else:
                    self.logger.info("NOT modifying OMEGA")
                    new_omega = omega

            else:
                if opt_eta:
                    if not eta_conv:
                        new_eta = np.zeros_like(eta)
                        for t in range(T):
                            if con[t] < 0:  # Eta was too big.
                                max_eta[t] = eta[t]
                                geom = np.sqrt(min_eta[t]*max_eta[t])  # Geometric mean.
                                new_eta[t] = max(geom, 0.1*max_eta[t])
                            else:
                                min_eta[t] = eta[t]
                                geom = np.sqrt(min_eta[t]*max_eta[t])  # Geometric mean.
                                new_eta[t] = min(geom, 10.0*min_eta[t])

                        if itr % 10 == 0:
                            self.logger.info("Modifying ETA | "
                                             "avg KL: %f <= %f, avg new eta: %.3e",
                                             np.mean(kl_div[:-1]),
                                             np.mean(kl_step[:-1]),
                                                np.mean(eta[:-1]))
                    else:
                        self.logger.info("NOT modifying ETA")
                        new_eta = eta

                # Choose new nu (bisect bracket or multiply by constant)
                self.logger.info("consider_bad: %s" % self.consider_bad)
                if opt_nu and self.consider_bad:
                    if not nu_conv:
                        new_nu = np.zeros_like(nu)
                        for t in range(T):
                            if con_bad[t] < 0:  # Nu was too big.
                                max_nu[t] = nu[t]
                                geom = np.sqrt(min_nu[t]*max_nu[t])  # Geometric mean.
                                #new_nu[t] = max(geom, 0.1*max_nu[t])
                                new_nu[t] = max(geom, 0.025*max_nu[t])
                            else:  # Nu was too small.
                                min_nu[t] = nu[t]
                                geom = np.sqrt(min_nu[t]*max_nu[t])  # Geometric mean.
                                #new_nu = min(geom, 10.0*min_nu[t])
                                new_nu[t] = min(geom, 2.5*min_nu[t])

                        if itr % 10 == 0:
                            self.logger.info("Modifying NU | "
                                             "avg KL: %f <= %f, avg new nu: %.3e",
                                             np.mean(kl_div_bad[:-1]),
                                             np.mean(kl_bad[:-1]),
                                             np.mean(nu[:-1]))
                    else:
                        self.logger.info("NOT modifying NU")
                        new_nu = nu

                # Choose new omega (bisect bracket or multiply by constant)
                self.logger.info("consider_good: %s" % self.consider_good)
                if opt_omega and self.consider_good:
                    if not omega_conv:
                        new_eta = np.zeros_like(eta)
                        for t in range(T):
                            if con_good[t] < 0:  # Omega was too big.
                                max_omega[t] = omega[t]
                                geom = np.sqrt(min_omega[t]*max_omega[t])  # Geometric mean.
                                new_omega[t] = max(geom, 0.1*max_omega[t])
                            else:
                                min_omega[t] = omega[t]
                                geom = np.sqrt(min_omega[t]*max_omega[t])  # Geometric mean.
                                new_omega[t] = min(geom, 10.0*min_omega[t])

                        if itr % 10 == 0:
                            self.logger.info("Modifying OMEGA | "
                                             "avg KL: %f <= %f, avg new omega: %.3e",
                                             np.mean(kl_div_good[:-1]),
                                             np.mean(kl_good[:-1]),
                                             np.mean(omega[:-1]))
                    else:
                        self.logger.info("NOT modifying OMEGA")
                        new_omega = omega

            # Remember previous dual values to check if the dual is changing
            prev_eta = eta
            prev_nu = nu
            prev_omega = omega

            # Logarithmic mean: log_mean(x,y) = (y - x)/(log(y) - log(x))
            eta = new_eta
            omega = new_omega
            nu = new_nu

            if not self.cons_per_step:
                if opt_eta:
                    self.logger.info('eta_conv %s: kl_div < epsilon (%f < %f) | %f%%'
                                     % (eta_conv, kl_div, kl_step,
                                        abs(con*100/kl_step)))
                if opt_nu:
                    self.logger.info('nu_conv %s: kl_div > xi (%f > %f) | %f%%'
                                     % (nu_conv, kl_div_bad, kl_bad,
                                        abs(con_bad*100/kl_bad)))
                if opt_omega:
                    self.logger.info('omega_conv %s: kl_div < chi (%f < %f) | %f%%'
                                     % (omega_conv, kl_div_good, kl_good,
                                        abs(con_good*100/kl_good)))
            else:
                self.logger.info('eta_conv %s: avg_kl_div < epsilon '
                                 '(%r < %r) | %r%%'
                                 % (eta_conv, np.mean(kl_div[:-1]),
                                    np.mean(kl_step[:-1]),
                                    abs(np.mean(con[:-1])*100/np.mean(kl_step[:-1]))))
                self.logger.info('nu_conv %s: avg_kl_div > xi '
                                 '(%r > %r) | %r%%'
                                 % (nu_conv, np.mean(kl_div_bad[:-1]),
                                    np.mean(kl_bad[:-1]),
                                    abs(np.mean(con_bad[:-1])*100/np.mean(kl_bad[:-1]))))
                self.logger.info('omega_conv %s: avg_kl_div < chi '
                                 '(%r < %r) | %r%%'
                                 % (omega_conv, np.mean(kl_div_good[:-1]),
                                    np.mean(kl_good[:-1]),
                                    abs(np.mean(con_good[:-1])*100/np.mean(kl_good[:-1]))))

        if not self.cons_per_step:
            self.logger.info('_'*40)
            self.logger.info('FINAL VALUES at itr %d || '
                             'eta: %f | nu: %f | omega: %f'
                             % (itr, eta, nu, omega))

            self.logger.info('eta_conv %s: kl_div < epsilon (%f < %f) | %f%%'
                             % (eta_conv, kl_div, kl_step,
                                abs(con*100/kl_step)))
            self.logger.info('nu_conv %s: kl_div > xi (%f > %f) | %f%%'
                             % (nu_conv, kl_div_bad, kl_bad,
                                abs(con_bad*100/kl_bad)))
            self.logger.info('omega_conv %s: kl_div < chi (%f < %f) | %f%%'
                             % (omega_conv, kl_div_good, kl_good,
                                abs(con_good*100/kl_good)))

        if itr + 1 == max_itr:
            self.logger.info("After %d iterations for ETA, NU, OMEGA,"
                             "the constraints have not been satisfied.", itr)

        return traj_distr, (eta, nu, omega), (eta_conv, nu_conv, omega_conv)

    def _dual_gradient_projection_all(self, algorithm, m, eta, nu, omega,
                                      dual_to_check='eta',
                                      opt_eta=True, opt_nu=True, opt_omega=True,
                                      alpha=None):
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
        # Set Good KL-divergence (chi) using bad step multiplier.
        kl_good = algorithm.base_kl_good * good_step_mult
        # Set Bad KL-divergence (xi) using bad step multiplier.
        kl_bad = algorithm.base_kl_bad * bad_step_mult

        if not self.cons_per_step:
            kl_step *= T
            kl_good *= T
            kl_bad *= T
        else:
            if not isinstance(kl_step, (np.ndarray, list)):
                self.logger.warning('KL_step is not iterable. Converting it')
                kl_step = np.ones(T)*kl_step
            if not isinstance(kl_good, (np.ndarray, list)):
                self.logger.warning('KL_good is not iterable. Converting it')
                kl_good = np.ones(T)*kl_good
            if not isinstance(kl_bad, (np.ndarray, list)):
                self.logger.warning('KL_bad is not iterable. Converting it')
                kl_bad = np.ones(T)*kl_bad

        # We assume at min_eta, kl_div > kl_step, opposite for max_eta.
        if not self.cons_per_step:
            min_eta = self._hyperparams['min_eta']
            max_eta = self._hyperparams['max_eta']
            min_nu = self._hyperparams['min_nu']
            max_nu = self._hyperparams['max_nu']
            min_omega = self._hyperparams['min_omega']
            max_omega = self._hyperparams['max_omega']
            self.logger.info('_'*60)
            self.logger.info("Running DAdam for traj[%d] | "
                             "eta: %4f, nu: %4f, omega: %4f",
                             m, eta, nu, omega)
            self.logger.info('_'*60)
        else:
            min_eta = np.ones(T) * self._hyperparams['min_eta']
            max_eta = np.ones(T) * self._hyperparams['max_eta']
            min_nu = np.ones(T) * self._hyperparams['min_nu']
            max_nu = np.ones(T) * self._hyperparams['max_nu']
            min_omega = np.ones(T) * self._hyperparams['min_omega']
            max_omega = np.ones(T) * self._hyperparams['max_omega']
            self.logger.info('_'*60)
            self.logger.info("Running DAdam for trajectory %d,"
                             "avg eta: %f, avg nu: %f, avg omega: %f",
                             m, np.mean(eta[:-1]), np.mean(nu[:-1]),
                             np.mean(omega[:-1]))
            self.logger.info('_'*60)

        # m_b and v_b per dual variable
        if self.cons_per_step:
            m_b, v_b = np.zeros((3, T-1)), np.zeros((3, T-1))
        else:
            m_b, v_b = np.zeros(3), np.zeros(3)

        for itr in range(DGD_MAX_GD_ITER):
            self.logger.info("-"*15)
            if not self.cons_per_step:
                self.logger.info("ALL Adam iter %d | Current dual values: "
                                 "eta %.2e, nu %.2e, omega %.2e"
                                 % (itr, eta, nu, omega))
            else:
                self.logger.info("ALL Adam iter %d | Current dual values: "
                                 "avg_eta %.2r, avg_nu %.2r, avg_omega %.2r"
                                 % (itr, np.mean(eta[:-1]), np.mean(nu[:-1]),
                                    np.mean(omega[:-1])))

            # TODO: ALWAYS DUAL_TO_CHECK ETA?????
            traj_distr, eta, nu, omega = \
                self.backward(prev_traj_distr, good_traj_distr,
                              bad_traj_distr, traj_info,
                              eta, nu, omega,
                              # algorithm, m, dual_to_check='nu')
                              algorithm, m, dual_to_check=dual_to_check)

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
            kl_div_bad = self._traj_distr_kl_fcn(mu_to_check, sigma_to_check,
                                                 traj_distr, bad_traj_distr,
                                                 tot=(not self.cons_per_step))
            kl_div_good = self._traj_distr_kl_fcn(mu_to_check, sigma_to_check,
                                                  traj_distr, good_traj_distr,
                                                  tot=(not self.cons_per_step))

            print('step', kl_div, kl_step)
            print('bad', kl_div_bad, kl_bad)
            print('good', kl_div_good, kl_good)

            con = kl_div - kl_step  # KL - epsilon
            con_bad = kl_bad - kl_div_bad  # xi - KL
            con_good = kl_div_good - kl_good  # KL - chi

            # Convergence check - constraint satisfaction.
            eta_conv = self._conv_prev_check(con, kl_step)
            nu_conv = self._conv_bad_check(con_bad, kl_bad)
            omega_conv = self._conv_good_check(con_good, kl_good)

            if self.cons_per_step:
                raise NotImplementedError('NOT IMPLEMENTED cons_per_step')
                min_duals = np.vstack((min_eta[:-1], min_nu[:-1], min_omega[:-1]))
                max_duals = np.vstack((max_eta[:-1], max_nu[:-1], max_omega[:-1]))
                duals = np.vstack((eta[:-1], nu[:-1], omega[:-1]))

                if alpha is None:
                    alpha = ALPHA

                grads = np.vstack((con, con_bad, con_good))
                m_b = (BETA1 * m_b + (1-BETA1) * grads[:, :-1])  # Biased first moment estimate
                v_b = (BETA2 * v_b + (1-BETA2) * np.square(grads[:, :-1]))  # Biased second raw moment estimate

                m_u = m_b / (1 - BETA1 ** (itr+1))  # Bias-corrected first moment estimate
                v_u = v_b / (1 - BETA2 ** (itr+1))  # Bias-corrected second raw moment estimate
                adam_update = duals + alpha * m_u / (np.sqrt(v_u) + EPS)

                adam_update = np.minimum(np.maximum(adam_update, min_duals),
                                         max_duals)
                eta[:-1] = adam_update[0, :]
                nu[:-1] = adam_update[1, :]
                omega[:-1] = adam_update[2, :]

            else:
                grads = np.array([con, con_bad, con_good])
                min_duals = np.array([min_eta, min_nu, min_omega])
                max_duals = np.array([max_eta, max_nu, max_omega])
                duals = np.array([eta, nu, omega])

                grads *= np.array([opt_eta, opt_nu, opt_omega])

                if alpha is None:
                    alpha = ALPHA

                divisor = np.linalg.norm(grads)


                m_b = (BETA1 * m_b + (1-BETA1) * grads)
                v_b = (BETA2 * v_b + (1-BETA2) * np.square(grads))

                m_u = m_b / (1 - BETA1 ** (itr+1))
                v_u = v_b / (1 - BETA2 ** (itr+1))

                adam_update = duals + alpha * m_u / (np.sqrt(v_u) + EPS)

                adam_update = np.minimum(np.maximum(adam_update, min_duals),
                                         max_duals)

                # print('grads:%r' % grads)
                # print('m_b:%r' % m_b)
                # print('m_u:%r' % m_u)
                self.logger.info('duals_change:%r'
                                 % (alpha * m_u / (np.sqrt(v_u) + EPS)))
                print('prev_eta:%f -- new_eta:%f' % (eta, adam_update[0]))
                print('prev_nu:%f -- new_nu:%f' % (nu, adam_update[1]))
                print('prev_omega:%f -- new_omega:%f' % (omega, adam_update[2]))
                eta = adam_update[0]
                nu = adam_update[1]
                omega = adam_update[2]

            if not self.cons_per_step:
                self.logger.info('eta_conv %s: kl_div < epsilon (%f < %f) | %f%%'
                                 % (eta_conv, kl_div, kl_step,
                                    abs(con*100/kl_step)))
                self.logger.info('nu_conv %s: kl_div > xi (%f > %f) | %f%%'
                                 % (nu_conv, kl_div_bad, kl_bad,
                                    abs(con_bad*100/kl_bad)))
                self.logger.info('omega_conv %s: kl_div < chi (%f < %f) | %f%%'
                                 % (omega_conv, kl_div_good, kl_good,
                                    abs(con_good*100/kl_good)))
            else:
                self.logger.info('eta_conv %s: avg_kl_div < epsilon '
                                 '(%r < %r) | %r%%'
                                 % (eta_conv, np.mean(kl_div[:-1]),
                                    np.mean(kl_step[:-1]),
                                    abs(np.mean(con[:-1])*100/np.mean(kl_step[:-1]))))
                self.logger.info('nu_conv %s: avg_kl_div > xi '
                                 '(%r > %r) | %r%%'
                                 % (nu_conv, np.mean(kl_div_bad[:-1]),
                                    np.mean(kl_bad[:-1]),
                                    abs(np.mean(con_bad[:-1])*100/np.mean(kl_bad[:-1]))))
                self.logger.info('omega_conv %s: avg_kl_div < chi '
                                 '(%r < %r) | %r%%'
                                 % (omega_conv, np.mean(kl_div_good[:-1]),
                                    np.mean(kl_good[:-1]),
                                    abs(np.mean(con_good[:-1])*100/np.mean(kl_good[:-1]))))

            if eta_conv and nu_conv and omega_conv:
                self.logger.info("It has converged with Adam")
                break

        if not self.cons_per_step:
            self.logger.info('_'*40)
            self.logger.info('FINAL VALUES at itr %d || '
                             'eta: %f | nu: %f | omega: %f'
                             % (itr, eta, nu, omega))

            self.logger.info('eta_conv %s: kl_div < epsilon (%f < %f) | %f%%'
                             % (eta_conv, kl_div, kl_step,
                                abs(con*100/kl_step)))
            self.logger.info('nu_conv %s: kl_div > xi (%f > %f) | %f%%'
                             % (nu_conv, kl_div_bad, kl_bad,
                                abs(con_bad*100/kl_bad)))
            self.logger.info('omega_conv %s: kl_div < chi (%f < %f) | %f%%'
                             % (omega_conv, kl_div_good, kl_good,
                                abs(con_good*100/kl_good)))

        return traj_distr, (eta, nu, omega), (eta_conv, nu_conv, omega_conv)

    def _adam_all(self, algorithm, m, eta, nu, omega, dual_to_check='eta',
                  opt_eta=True, opt_nu=True, opt_omega=True, alpha=None):
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
        # Set Good KL-divergence (chi) using bad step multiplier.
        kl_good = algorithm.base_kl_good * good_step_mult
        # Set Bad KL-divergence (xi) using bad step multiplier.
        kl_bad = algorithm.base_kl_bad * bad_step_mult

        if not self.cons_per_step:
            kl_step *= T
            kl_good *= T
            kl_bad *= T
        else:
            if not isinstance(kl_step, (np.ndarray, list)):
                self.logger.warning('KL_step is not iterable. Converting it')
                kl_step = np.ones(T)*kl_step
            if not isinstance(kl_good, (np.ndarray, list)):
                self.logger.warning('KL_good is not iterable. Converting it')
                kl_good = np.ones(T)*kl_good
            if not isinstance(kl_bad, (np.ndarray, list)):
                self.logger.warning('KL_bad is not iterable. Converting it')
                kl_bad = np.ones(T)*kl_bad

        # We assume at min_eta, kl_div > kl_step, opposite for max_eta.
        if not self.cons_per_step:
            min_eta = self._hyperparams['min_eta']
            max_eta = self._hyperparams['max_eta']
            min_nu = self._hyperparams['min_nu']
            max_nu = self._hyperparams['max_nu']
            min_omega = self._hyperparams['min_omega']
            max_omega = self._hyperparams['max_omega']
            self.logger.info('_'*60)
            self.logger.info("Running DAdam for traj[%d] | "
                             "eta: %4f, nu: %4f, omega: %4f",
                             m, eta, nu, omega)
            self.logger.info('_'*60)
        else:
            min_eta = np.ones(T) * self._hyperparams['min_eta']
            max_eta = np.ones(T) * self._hyperparams['max_eta']
            min_nu = np.ones(T) * self._hyperparams['min_nu']
            max_nu = np.ones(T) * self._hyperparams['max_nu']
            min_omega = np.ones(T) * self._hyperparams['min_omega']
            max_omega = np.ones(T) * self._hyperparams['max_omega']
            self.logger.info('_'*60)
            self.logger.info("Running DAdam for trajectory %d,"
                             "avg eta: %f, avg nu: %f, avg omega: %f",
                             m, np.mean(eta[:-1]), np.mean(nu[:-1]),
                             np.mean(omega[:-1]))
            self.logger.info('_'*60)

        # m_b and v_b per dual variable
        if self.cons_per_step:
            m_b, v_b = np.zeros((3, T-1)), np.zeros((3, T-1))
        else:
            m_b, v_b = np.zeros(3), np.zeros(3)

        for itr in range(DGD_MAX_GD_ITER):
            self.logger.info("-"*15)
            if not self.cons_per_step:
                self.logger.info("ALL Adam iter %d | Current dual values: "
                                 "eta %.2e, nu %.2e, omega %.2e"
                                 % (itr, eta, nu, omega))
            else:
                self.logger.info("ALL Adam iter %d | Current dual values: "
                                 "avg_eta %.2r, avg_nu %.2r, avg_omega %.2r"
                                 % (itr, np.mean(eta[:-1]), np.mean(nu[:-1]),
                                    np.mean(omega[:-1])))

            # TODO: ALWAYS DUAL_TO_CHECK ETA?????
            traj_distr, eta, nu, omega = \
                self.backward(prev_traj_distr, good_traj_distr,
                              bad_traj_distr, traj_info,
                              eta, nu, omega,
                              # algorithm, m, dual_to_check='nu')
                              algorithm, m, dual_to_check=dual_to_check)

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
            kl_div_bad = self._traj_distr_kl_fcn(mu_to_check, sigma_to_check,
                                                 traj_distr, bad_traj_distr,
                                                 tot=(not self.cons_per_step))
            kl_div_good = self._traj_distr_kl_fcn(mu_to_check, sigma_to_check,
                                                  traj_distr, good_traj_distr,
                                                  tot=(not self.cons_per_step))

            print('step', kl_div, kl_step)
            print('bad', kl_div_bad, kl_bad)
            print('good', kl_div_good, kl_good)

            con = kl_div - kl_step  # KL - epsilon
            con_bad = kl_bad - kl_div_bad  # xi - KL
            con_good = kl_div_good - kl_good  # KL - chi

            # Convergence check - constraint satisfaction.
            eta_conv = self._conv_prev_check(con, kl_step)
            nu_conv = self._conv_bad_check(con_bad, kl_bad)
            omega_conv = self._conv_good_check(con_good, kl_good)

            if self.cons_per_step:
                min_duals = np.vstack((min_eta[:-1], min_nu[:-1], min_omega[:-1]))
                max_duals = np.vstack((max_eta[:-1], max_nu[:-1], max_omega[:-1]))
                duals = np.vstack((eta[:-1], nu[:-1], omega[:-1]))

                if alpha is None:
                    alpha = ALPHA

                grads = np.vstack((con, con_bad, con_good))
                m_b = (BETA1 * m_b + (1-BETA1) * grads[:, :-1])  # Biased first moment estimate
                v_b = (BETA2 * v_b + (1-BETA2) * np.square(grads[:, :-1]))  # Biased second raw moment estimate

                m_u = m_b / (1 - BETA1 ** (itr+1))  # Bias-corrected first moment estimate
                v_u = v_b / (1 - BETA2 ** (itr+1))  # Bias-corrected second raw moment estimate
                adam_update = duals + alpha * m_u / (np.sqrt(v_u) + EPS)

                adam_update = np.minimum(np.maximum(adam_update, min_duals),
                                         max_duals)
                eta[:-1] = adam_update[0, :]
                nu[:-1] = adam_update[1, :]
                omega[:-1] = adam_update[2, :]

            else:
                grads = np.array([con, con_bad, con_good])
                min_duals = np.array([min_eta, min_nu, min_omega])
                max_duals = np.array([max_eta, max_nu, max_omega])
                duals = np.array([eta, nu, omega])

                grads *= np.array([opt_eta, opt_nu, opt_omega])

                if alpha is None:
                    alpha = ALPHA

                m_b = (BETA1 * m_b + (1-BETA1) * grads)
                v_b = (BETA2 * v_b + (1-BETA2) * np.square(grads))

                m_u = m_b / (1 - BETA1 ** (itr+1))
                v_u = v_b / (1 - BETA2 ** (itr+1))

                adam_update = duals + alpha * m_u / (np.sqrt(v_u) + EPS)

                adam_update = np.minimum(np.maximum(adam_update, min_duals),
                                         max_duals)

                # print('grads:%r' % grads)
                # print('m_b:%r' % m_b)
                # print('m_u:%r' % m_u)
                self.logger.info('duals_change:%r'
                                 % (alpha * m_u / (np.sqrt(v_u) + EPS)))
                print('prev_eta:%f -- new_eta:%f' % (eta, adam_update[0]))
                print('prev_nu:%f -- new_nu:%f' % (nu, adam_update[1]))
                print('prev_omega:%f -- new_omega:%f' % (omega, adam_update[2]))
                eta = adam_update[0]
                nu = adam_update[1]
                omega = adam_update[2]

            if not self.cons_per_step:
                self.logger.info('eta_conv %s: kl_div < epsilon (%f < %f) | %f%%'
                                 % (eta_conv, kl_div, kl_step,
                                    abs(con*100/kl_step)))
                self.logger.info('nu_conv %s: kl_div > xi (%f > %f) | %f%%'
                                 % (nu_conv, kl_div_bad, kl_bad,
                                    abs(con_bad*100/kl_bad)))
                self.logger.info('omega_conv %s: kl_div < chi (%f < %f) | %f%%'
                                 % (omega_conv, kl_div_good, kl_good,
                                    abs(con_good*100/kl_good)))
            else:
                self.logger.info('eta_conv %s: avg_kl_div < epsilon '
                                 '(%r < %r) | %r%%'
                                 % (eta_conv, np.mean(kl_div[:-1]),
                                    np.mean(kl_step[:-1]),
                                    abs(np.mean(con[:-1])*100/np.mean(kl_step[:-1]))))
                self.logger.info('nu_conv %s: avg_kl_div > xi '
                                 '(%r > %r) | %r%%'
                                 % (nu_conv, np.mean(kl_div_bad[:-1]),
                                    np.mean(kl_bad[:-1]),
                                    abs(np.mean(con_bad[:-1])*100/np.mean(kl_bad[:-1]))))
                self.logger.info('omega_conv %s: avg_kl_div < chi '
                                 '(%r < %r) | %r%%'
                                 % (omega_conv, np.mean(kl_div_good[:-1]),
                                    np.mean(kl_good[:-1]),
                                    abs(np.mean(con_good[:-1])*100/np.mean(kl_good[:-1]))))

            if eta_conv and nu_conv and omega_conv:
                self.logger.info("It has converged with Adam")
                break

        if not self.cons_per_step:
            self.logger.info('_'*40)
            self.logger.info('FINAL VALUES at itr %d || '
                             'eta: %f | nu: %f | omega: %f'
                             % (itr, eta, nu, omega))

            self.logger.info('eta_conv %s: kl_div < epsilon (%f < %f) | %f%%'
                             % (eta_conv, kl_div, kl_step,
                                abs(con*100/kl_step)))
            self.logger.info('nu_conv %s: kl_div > xi (%f > %f) | %f%%'
                             % (nu_conv, kl_div_bad, kl_bad,
                                abs(con_bad*100/kl_bad)))
            self.logger.info('omega_conv %s: kl_div < chi (%f < %f) | %f%%'
                             % (omega_conv, kl_div_good, kl_good,
                                abs(con_good*100/kl_good)))

        return traj_distr, (eta, nu, omega), (eta_conv, nu_conv, omega_conv)

    def lagrangian_function(self, duals, algorithm, m,
                            opt_eta=True, opt_nu=True, opt_omega=True):
        self.logger.info(duals)
        eta = duals[0]
        nu = duals[1]
        omega = duals[2]

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
        # Set Good KL-divergence (chi) using bad step multiplier.
        kl_good = algorithm.base_kl_good * good_step_mult
        # Set Bad KL-divergence (xi) using bad step multiplier.
        kl_bad = algorithm.base_kl_bad * bad_step_mult

        traj_distr, eta, nu, omega = \
            self.backward(prev_traj_distr, good_traj_distr,
                          bad_traj_distr, traj_info,
                          eta, nu, omega,
                          # algorithm, m, dual_to_check='nu')
                          algorithm, m, dual_to_check='eta')

        traj_cost = self.estimate_cost(traj_distr, traj_info).sum()

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
        kl_div_bad = self._traj_distr_kl_fcn(mu_to_check, sigma_to_check,
                                             traj_distr, bad_traj_distr,
                                             tot=(not self.cons_per_step))
        kl_div_good = self._traj_distr_kl_fcn(mu_to_check, sigma_to_check,
                                              traj_distr, good_traj_distr,
                                              tot=(not self.cons_per_step))

        con = kl_div - kl_step  # KL - epsilon
        con_bad = kl_bad - kl_div_bad  # xi - KL
        con_good = kl_div_good - kl_good  # KL - chi

        # self.logger.info('LAG duals: %f, %f, %f' % (eta, nu, omega))
        # self.logger.info('LAG gradients: %f, %f, %f' % (con, con_bad, con_good))

        if not opt_eta:
            con = 0

        if not opt_nu:
            con_bad = 0

        if not opt_omega:
            con_good = 0

        # total_cost = traj_cost + eta*con + nu*con_bad + omega*con_good
        # total_cost = con - con_bad + con_good
        # total_cost = abs(con) + abs(con_bad) + abs(con_good)
        total_cost = (con)**2 + (con_bad)**2 + (con_good)**2

        print('desired:', kl_step, kl_bad, kl_good)
        print('current:', kl_div, kl_div_bad, kl_div_good)
        print('TOTAL_COST:', total_cost, '|', con, con_bad, con_good)

        return total_cost

    def lagrangian_gradient(self, duals, algorithm, m,
                            opt_eta=True, opt_nu=True, opt_omega=True):
        self.logger.info(duals)
        eta = duals[0]
        nu = duals[1]
        omega = duals[2]

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
        # Set Good KL-divergence (chi) using bad step multiplier.
        kl_good = algorithm.base_kl_good * good_step_mult
        # Set Bad KL-divergence (xi) using bad step multiplier.
        kl_bad = algorithm.base_kl_bad * bad_step_mult

        traj_distr, eta, nu, omega = \
            self.backward(prev_traj_distr, good_traj_distr,
                          bad_traj_distr, traj_info,
                          eta, nu, omega,
                          # algorithm, m, dual_to_check='nu')
                          algorithm, m, dual_to_check='eta')

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
        kl_div_bad = self._traj_distr_kl_fcn(mu_to_check, sigma_to_check,
                                             traj_distr, bad_traj_distr,
                                             tot=(not self.cons_per_step))
        kl_div_good = self._traj_distr_kl_fcn(mu_to_check, sigma_to_check,
                                              traj_distr, good_traj_distr,
                                              tot=(not self.cons_per_step))

        con = kl_div - kl_step  # KL - epsilon
        con_bad = kl_bad - kl_div_bad  # xi - KL
        con_good = kl_div_good - kl_good  # KL - chi
        # self.logger.info('duals: %f, %f, %f' % (eta, nu, omega))
        # self.logger.info('gradients: %f, %f, %f' % (con, con_bad, con_good))

        if not opt_eta:
            con = 0

        if not opt_nu:
            con_bad = 0

        if not opt_omega:
            con_good = 0

        # self.logger.info('final gradients: %f, %f, %f' % (con, con_bad, con_good))

        # return np.array([con, con_bad, con_good])
        # return np.array([con, -con_bad, con_good])
        return np.array([con/abs(con+1e-10), con_bad/abs(con_bad+1e-10),
                         con_good/abs(con_good+1e-10)])

    def fcn_to_optimize(self, duals, algorithm, m,
                        opt_eta=True, opt_nu=True, opt_omega=True):
        self.logger.info(duals)
        eta = duals[0]
        nu = duals[1]
        omega = duals[2]

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
        # Set Good KL-divergence (chi) using bad step multiplier.
        kl_good = algorithm.base_kl_good * good_step_mult
        # Set Bad KL-divergence (xi) using bad step multiplier.
        kl_bad = algorithm.base_kl_bad * bad_step_mult

        traj_distr, eta, nu, omega = \
            self.backward(prev_traj_distr, good_traj_distr,
                          bad_traj_distr, traj_info,
                          eta, nu, omega,
                          # algorithm, m, dual_to_check='nu')
                          algorithm, m, dual_to_check='eta')

        traj_cost = self.estimate_cost(traj_distr, traj_info).sum()

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
        kl_div_bad = self._traj_distr_kl_fcn(mu_to_check, sigma_to_check,
                                             traj_distr, bad_traj_distr,
                                             tot=(not self.cons_per_step))
        kl_div_good = self._traj_distr_kl_fcn(mu_to_check, sigma_to_check,
                                              traj_distr, good_traj_distr,
                                              tot=(not self.cons_per_step))

        con = kl_div - kl_step  # KL - epsilon
        con_bad = kl_bad - kl_div_bad  # xi - KL
        con_good = kl_div_good - kl_good  # KL - chi

        # self.logger.info('LAG duals: %f, %f, %f' % (eta, nu, omega))
        # self.logger.info('LAG gradients: %f, %f, %f' % (con, con_bad, con_good))

        if not opt_eta:
            con = 0

        if not opt_nu:
            con_bad = 0

        if not opt_omega:
            con_good = 0

        # total_cost = traj_cost + eta*con + nu*con_bad + omega*con_good
        # total_cost = con - con_bad + con_good
        # total_cost = abs(con) + abs(con_bad) + abs(con_good)
        total_cost = - (traj_cost + eta*con + nu*con_bad + omega*con_good)

        print('desired:', kl_step, kl_bad, kl_good)
        print('current:', kl_div, kl_div_bad, kl_div_good)
        print('TOTAL_COST:', total_cost, '|', con, con_bad, con_good)

        return total_cost

    def grad_to_optimize(self, duals, algorithm, m,
                         opt_eta=True, opt_nu=True, opt_omega=True):
        self.logger.info(duals)
        eta = duals[0]
        nu = duals[1]
        omega = duals[2]

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
        # Set Good KL-divergence (chi) using bad step multiplier.
        kl_good = algorithm.base_kl_good * good_step_mult
        # Set Bad KL-divergence (xi) using bad step multiplier.
        kl_bad = algorithm.base_kl_bad * bad_step_mult

        traj_distr, eta, nu, omega = \
            self.backward(prev_traj_distr, good_traj_distr,
                          bad_traj_distr, traj_info,
                          eta, nu, omega,
                          # algorithm, m, dual_to_check='nu')
                          algorithm, m, dual_to_check='eta')

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
        kl_div_bad = self._traj_distr_kl_fcn(mu_to_check, sigma_to_check,
                                             traj_distr, bad_traj_distr,
                                             tot=(not self.cons_per_step))
        kl_div_good = self._traj_distr_kl_fcn(mu_to_check, sigma_to_check,
                                              traj_distr, good_traj_distr,
                                              tot=(not self.cons_per_step))

        con = kl_div - kl_step  # KL - epsilon
        con_bad = kl_bad - kl_div_bad  # xi - KL
        con_good = kl_div_good - kl_good  # KL - chi
        # self.logger.info('duals: %f, %f, %f' % (eta, nu, omega))
        # self.logger.info('gradients: %f, %f, %f' % (con, con_bad, con_good))

        if not opt_eta:
            con = 0

        if not opt_nu:
            con_bad = 0

        if not opt_omega:
            con_good = 0

        # self.logger.info('final gradients: %f, %f, %f' % (con, con_bad, con_good))

        # return np.array([con, con_bad, con_good])
        return np.array([-con, -con_bad, -con_good])
        # return np.array([con, -con_bad, con_good])
        # return np.array([con/abs(con+1e-10), con_bad/abs(con_bad+1e-10),
        #                  con_good/abs(con_good+1e-10)])

