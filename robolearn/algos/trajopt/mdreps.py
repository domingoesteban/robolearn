""" This file defines the mDREPS-based trajectory optimization method. """

import sys
import numpy as np
import scipy as sp

from robolearn.algos.gps.gps import GPS
from robolearn.algos.trajopt.trajopt_config import default_mdreps_hyperparams
from robolearn.algos.gps.gps_utils import IterationData, TrajectoryInfo, extract_condition, DualityInfo
from robolearn.utils.sample import Sample
from robolearn.utils.sample_list import SampleList
from robolearn.policies.lin_gauss_policy import LinearGaussianPolicy
from robolearn.algos.gps.gps_utils import gauss_fit_joint_prior


import logging
LOGGER = logging.getLogger(__name__)
# Logging into console AND file
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
LOGGER.addHandler(ch)


class MDREPS(GPS):
    """ Sample-based trajectory optimization. """
    def __init__(self, agent, env, **kwargs):
        super(MDREPS, self).__init__(agent, env, **kwargs)
        gps_algo_hyperparams = default_mdreps_hyperparams.copy()
        gps_algo_hyperparams.update(self._hyperparams['gps_algo_hyperparams'])
        self._hyperparams.update(gps_algo_hyperparams)

        self.gps_algo = 'mdreps'

        # TrajectoryInfo for good and bad trajectories
        if self._hyperparams['fit_dynamics']:
            # Add dynamics if the algorithm requires fit_dynamics (Same type for all the conditions)
            dynamics = self._hyperparams['dynamics']

        self.good_trajectories_info = [None for _ in range(self.M)]
        self.bad_trajectories_info = [None for _ in range(self.M)]
        self.good_duality_info = [DualityInfo() for _ in range(self.M)]  # [Sample_list, cs_each, cs_traj, traj_dist]
        self.bad_duality_info = [DualityInfo() for _ in range(self.M)]

        for m in range(self.M):
            self.good_trajectories_info[m] = TrajectoryInfo()
            self.bad_trajectories_info[m] = TrajectoryInfo()

            if self._hyperparams['fit_dynamics']:
                self.good_trajectories_info[m].dynamics = dynamics['type'](dynamics)
                self.bad_trajectories_info[m].dynamics = dynamics['type'](dynamics)

            # TODO: Use demonstration trajectories
            # # Get the initial trajectory distribution hyperparams
            # init_traj_distr = extract_condition(self._hyperparams['init_traj_distr'], self._train_cond_idx[m])
            # Instantiate Trajectory Distribution: init_lqr or init_pd
            # self.good_duality_infor = init_traj_distr['type'](init_traj_distr)
            # self.bad_duality_infor = init_traj_distr['type'](init_traj_distr)

        self.base_kl_good = self._hyperparams['base_kl_good']
        self.base_kl_bad = self._hyperparams['base_kl_bad']

    def iteration(self, sample_lists):
        """
        Run iteration of mDREPS.
        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]

        # Update dynamics model using all samples.
        print('')
        print('->Updating dynamics linearization...')
        self._update_dynamics()

        print('')
        print('->Updating KL step size...')
        self._update_step_size()  # KL Divergence step size.

        print('')
        print('->Getting good and bad trajectories...')
        self._get_good_trajectories()
        self._get_bad_trajectories()

        print('')
        print('->Updating data of good and bad trajectories...')
        print('-->Update dynamics...')
        self._update_good_bad_dynamics()
        print('-->Update costs...')
        self._eval_good_bad_costs()
        print('-->Update traj dist...')
        self._fit_good_bad_traj_dist()

        # Run inner loop to compute new policies.
        print('->Updating trajectories...')
        for ii in range(self._hyperparams['inner_iterations']):
            print('-->Inner iteration %d/%d' % (ii+1, self._hyperparams['inner_iterations']))
            self._update_trajectories()

        self._advance_iteration_variables()

    def _update_step_size(self):
        """ Evaluate costs on samples, and adjust the step size. """
        # Evaluate cost function for all conditions and samples.
        for m in range(self.M):
            self._eval_cost(m)

        # Adjust step size relative to the previous iteration.
        for m in range(self.M):
            if self.iteration_count >= 1 and self.prev[m].sample_list:
                self._stepadjust(m)

    def _stepadjust(self, m):
        """
        Calculate new step sizes.
        Args:
            m: Condition
        """
        # Compute values under Laplace approximation. This is the policy
        # that the previous samples were actually drawn from under the
        # dynamics that were estimated from the previous samples.
        previous_laplace_obj = self.traj_opt.estimate_cost(
            self.prev[m].traj_distr, self.prev[m].traj_info
        )
        # This is the policy that we just used under the dynamics that
        # were estimated from the previous samples (so this is the cost
        # we thought we would have).
        new_predicted_laplace_obj = self.traj_opt.estimate_cost(
            self.cur[m].traj_distr, self.prev[m].traj_info
        )

        # This is the actual cost we have under the current trajectory
        # based on the latest samples.
        new_actual_laplace_obj = self.traj_opt.estimate_cost(
            self.cur[m].traj_distr, self.cur[m].traj_info
        )

        # Measure the entropy of the current trajectory (for printout).
        ent = self._measure_ent(m)

        # Compute actual objective values based on the samples.
        previous_mc_obj = np.mean(np.sum(self.prev[m].cs, axis=1), axis=0)
        new_mc_obj = np.mean(np.sum(self.cur[m].cs, axis=1), axis=0)

        LOGGER.debug('Trajectory step: ent: %f cost: %f -> %f',
                     ent, previous_mc_obj, new_mc_obj)

        # Compute predicted and actual improvement.
        predicted_impr = np.sum(previous_laplace_obj) - \
                np.sum(new_predicted_laplace_obj)
        actual_impr = np.sum(previous_laplace_obj) - \
                np.sum(new_actual_laplace_obj)

        # Print improvement details.
        LOGGER.debug('Previous cost: Laplace: %f MC: %f',
                     np.sum(previous_laplace_obj), previous_mc_obj)
        LOGGER.debug('Predicted new cost: Laplace: %f MC: %f',
                     np.sum(new_predicted_laplace_obj), new_mc_obj)
        LOGGER.debug('Actual new cost: Laplace: %f MC: %f',
                     np.sum(new_actual_laplace_obj), new_mc_obj)
        LOGGER.debug('Predicted/actual improvement: %f / %f',
                     predicted_impr, actual_impr)

        self._set_new_mult(predicted_impr, actual_impr, m)

    def compute_costs(self, m, eta, omega, nu, augment=True):
        """
        Compute cost estimates used in the LQR backward pass.
        :param m: Condition
        :param eta: Dual variable corresponding to KL divergence with previous policy.
        :param omega: Dual variable(s) corresponding to KL divergence with good trajectories.
        :param nu: Dual variable(s) corresponding to KL divergence with bad trajectories.
        :param augment: True if we want a KL constraint for all time-steps. False otherwise.
        :return: Cm and cv
        """
        traj_info = self.cur[m].traj_info
        traj_distr = self.cur[m].traj_distr
        good_distr = self.good_duality_info[m].traj_dist
        bad_distr = self.bad_duality_info[m].traj_dist
        if not augment:  # Whether to augment cost with term to penalize KL
            return traj_info.Cm, traj_info.cv

        multiplier = self._hyperparams['max_ent_traj']  # Weight of maximum entropy term in trajectory optimization

        # TVLGC terms from previous traj
        K, ipc, k = traj_distr.K, traj_distr.inv_pol_covar, traj_distr.k
        # TVLGC terms from good traj
        K_good, ipc_good, k_good = good_distr.K, good_distr.inv_pol_covar, good_distr.k
        # TVLGC terms from bad traj
        K_bad, ipc_bad, k_bad = bad_distr.K, bad_distr.inv_pol_covar, bad_distr.k

        # omega = 0
        # nu = 0

        # Surrogate cost
        fCm = traj_info.Cm / (eta + omega - nu + multiplier)
        fcv = traj_info.cv / (eta + omega - nu + multiplier)

        # Add in the KL divergence with previous policy.
        for t in range(self.T - 1, -1, -1):
            fCm[t, :, :] += eta / (eta + omega - nu + multiplier) * np.vstack([
                np.hstack([
                    K[t, :, :].T.dot(ipc[t, :, :]).dot(K[t, :, :]),
                    -K[t, :, :].T.dot(ipc[t, :, :])
                ]),
                np.hstack([
                    -ipc[t, :, :].dot(K[t, :, :]), ipc[t, :, :]
                ])
            ])
            fcv[t, :] += eta / (eta + omega - nu + multiplier) * np.hstack([
                K[t, :, :].T.dot(ipc[t, :, :]).dot(k[t, :]),
                -ipc[t, :, :].dot(k[t, :])
            ])

        # Add in the KL divergence with good trajectories.
        for t in range(self.T - 1, -1, -1):
            fCm[t, :, :] += omega / (eta + omega - nu + multiplier) * np.vstack([
                np.hstack([
                    K_good[t, :, :].T.dot(ipc_good[t, :, :]).dot(K_good[t, :, :]),
                    -K_good[t, :, :].T.dot(ipc_good[t, :, :])
                ]),
                np.hstack([
                    -ipc_good[t, :, :].dot(K_good[t, :, :]), ipc_good[t, :, :]
                ])
            ])
            fcv[t, :] += omega / (eta + omega - nu + multiplier) * np.hstack([
                K_good[t, :, :].T.dot(ipc_good[t, :, :]).dot(k_good[t, :]),
                -ipc_good[t, :, :].dot(k_good[t, :])
            ])

        # Subtract in the KL divergence with bad trajectories.
        print("TODO: We are adding the bad trajs to cost, not substracting!!")
        for t in range(self.T - 1, -1, -1):
            fCm[t, :, :] += nu / (eta + omega - nu + multiplier) * np.vstack([
                np.hstack([
                    K_bad[t, :, :].T.dot(ipc_bad[t, :, :]).dot(K_bad[t, :, :]),
                    -K_bad[t, :, :].T.dot(ipc_bad[t, :, :])
                ]),
                np.hstack([
                    -ipc_bad[t, :, :].dot(K_bad[t, :, :]), ipc_bad[t, :, :]
                ])
            ])
            fcv[t, :] += nu / (eta + omega - nu + multiplier) * np.hstack([
                K_bad[t, :, :].T.dot(ipc_bad[t, :, :]).dot(k_bad[t, :]),
                -ipc_bad[t, :, :].dot(k_bad[t, :])
            ])

        return fCm, fcv

    def _get_good_trajectories(self):
        for cond in range(self.M):
            # Sample costs estimate.
            cs = self.cur[cond].cs
            sample_list = self.cur[cond].sample_list

            # Get index of sample with best Return
            #best_index = np.argmin(np.sum(cs, axis=1))
            n_good = self._hyperparams['n_good_samples']
            best_indeces = np.argpartition(np.sum(cs, axis=1), n_good)[:n_good]

            # Get current best trajectory
            if self.good_duality_info[cond].sample_list is None:
                self.good_duality_info[cond].sample_list = SampleList([sample_list[good_index] for good_index in best_indeces])
                self.good_duality_info[cond].samples_cost = cs[best_indeces, :]
            else:
                # Update only if it is better than previous traj_dist
                for good_index in best_indeces:
                    least_worse_index = np.argpartition(np.sum(self.good_duality_info[cond].samples_cost, axis=1), -1)[-1:]
                    if np.sum(self.good_duality_info[cond].samples_cost[least_worse_index, :]) > np.sum(cs[good_index, :]):
                        self.good_duality_info[cond].sample_list.set_sample(least_worse_index, sample_list[good_index])
                        self.good_duality_info[cond].samples_cost[least_worse_index, :] = cs[good_index, :]

    def _get_bad_trajectories(self):
        for cond in range(self.M):
            # Sample costs estimate.
            cs = self.cur[cond].cs
            sample_list = self.cur[cond].sample_list

            # Get index of sample with worst Return
            #worst_index = np.argmax(np.sum(cs, axis=1))
            n_bad = self._hyperparams['n_bad_samples']
            worst_indeces = np.argpartition(np.sum(cs, axis=1), -n_bad)[-n_bad:]

            # Get current best trajectory
            if self.bad_duality_info[cond].sample_list is None:
                self.bad_duality_info[cond].sample_list = SampleList([sample_list[bad_index] for bad_index in worst_indeces])
                self.bad_duality_info[cond].samples_cost = cs[worst_indeces, :]
            else:
                # Update only if it is better than before
                for bad_index in worst_indeces:
                    least_worst_index = np.argpartition(np.sum(self.bad_duality_info[cond].samples_cost, axis=1), 1)[:1]
                    if np.sum(self.bad_duality_info[cond].samples_cost[least_worst_index, :]) < np.sum(cs[bad_index, :]):
                        print("replacing %f > %f" % (np.sum(self.bad_duality_info[cond].samples_cost[least_worst_index, :]),
                                                     np.sum(cs[bad_index, :])))
                        self.bad_duality_info[cond].sample_list.set_sample(least_worst_index, sample_list[bad_index])
                        self.bad_duality_info[cond].samples_cost[least_worst_index, :] = cs[bad_index, :]

    def _update_good_bad_dynamics(self):
        """
        Instantiate dynamics objects and update prior. Fit dynamics to sample(s).
        """
        for m in range(self.M):
            good_data = self.good_duality_info[m].sample_list
            bad_data = self.bad_duality_info[m].sample_list
            X_good = good_data.get_states()
            U_good = good_data.get_actions()
            X_bad = bad_data.get_states()
            U_bad = bad_data.get_actions()

            # Update prior and fit dynamics.
            self.good_trajectories_info[m].dynamics.update_prior(good_data)
            self.good_trajectories_info[m].dynamics.fit(X_good, U_good)
            self.bad_trajectories_info[m].dynamics.update_prior(bad_data)
            self.bad_trajectories_info[m].dynamics.fit(X_bad, U_bad)

            # Fit x0mu/x0sigma.
            x0_good = X_good[:, 0, :]
            x0mu_good = np.mean(x0_good, axis=0)  # TODO: SAME X0 FOR ALL??
            self.good_trajectories_info[m].x0mu = x0mu_good
            self.good_trajectories_info[m].x0sigma = np.diag(np.maximum(np.var(x0_good, axis=0),
                                                                        self._hyperparams['initial_state_var']))
            x0_bad = X_bad[:, 0, :]
            x0mu_bad = np.mean(x0_bad, axis=0)  # TODO: SAME X0 FOR ALL??
            self.bad_trajectories_info[m].x0mu = x0mu_bad
            self.bad_trajectories_info[m].x0sigma = np.diag(np.maximum(np.var(x0_bad, axis=0),
                                                                        self._hyperparams['initial_state_var']))

            prior_good = self.good_trajectories_info[m].dynamics.get_prior()
            if prior_good:
                mu0, Phi, priorm, n0 = prior_good.initial_state()
                N = len(good_data)
                self.good_trajectories_info[m].x0sigma += Phi + (N*priorm) / (N+priorm) * np.outer(x0mu_good-mu0, x0mu_good-mu0) / (N+n0)

            prior_bad = self.good_trajectories_info[m].dynamics.get_prior()
            if prior_bad:
                mu0, Phi, priorm, n0 = prior_bad.initial_state()
                N = len(bad_data)
                self.bad_trajectories_info[m].x0sigma += Phi + (N*priorm) / (N+priorm) * np.outer(x0mu_bad-mu0, x0mu_bad-mu0) / (N+n0)

    def _eval_good_bad_costs(self):
        """
        Evaluate costs for all current samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        for cond in range(self.M):
            cs, cc, cv, Cm = self._eval_dual_costs(cond, self.good_duality_info[cond].sample_list)
            self.good_duality_info[cond].traj_cost = cs
            self.good_trajectories_info[cond].cc = cc
            self.good_trajectories_info[cond].cv = cv
            self.good_trajectories_info[cond].Cm = Cm

            cs, cc, cv, Cm = self._eval_dual_costs(cond, self.bad_duality_info[cond].sample_list)
            self.bad_duality_info[cond].traj_cost = cs
            self.bad_trajectories_info[cond].cc = cc
            self.bad_trajectories_info[cond].cv = cv
            self.bad_trajectories_info[cond].Cm = Cm

    def _eval_dual_costs(self, cond, dual_sample_list):
        # Constants.
        T, dX, dU = self.T, self.dX, self.dU
        N = len(dual_sample_list)

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX+dU))
        Cm = np.zeros((N, T, dX+dU, dX+dU))
        for n in range(N):
            sample = dual_sample_list[n]
            # Get costs.
            l, lx, lu, lxx, luu, lux, _ = self.cost_function[cond].eval(sample)
            cc[n, :] = l
            cs[n, :] = l

            # Assemble matrix and vector.
            cv[n, :, :] = np.c_[lx, lu]
            Cm[n, :, :, :] = np.concatenate(
                (np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]),
                axis=1
            )

            # Adjust for expanding cost around a sample.
            X = sample.get_states()
            U = sample.get_acts()
            yhat = np.c_[X, U]
            rdiff = -yhat
            rdiff_expand = np.expand_dims(rdiff, axis=2)
            cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)
            cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) + 0.5 * np.sum(rdiff * cv_update, axis=1)
            cv[n, :, :] += cv_update

        # Fill in cost estimate.
        cc = np.mean(cc, 0)  # Constant term (scalar).
        cv = np.mean(cv, 0)  # Linear term (vector).
        Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        return cs, cc, cv, Cm

    def _fit_good_bad_traj_dist(self):
        for cond in range(self.M):
            self.good_duality_info[cond].traj_dist = self.fit_traj_dist(self.good_duality_info[cond].sample_list)
            self.bad_duality_info[cond].traj_dist = self.fit_traj_dist(self.bad_duality_info[cond].sample_list)

    @staticmethod
    def fit_traj_dist(sample_list):
        samples = sample_list

        X = samples.get_states()
        obs = samples.get_obs()
        U = samples.get_actions()

        N, T, dX = X.shape
        dU = U.shape[2]
        if N == 1:
            raise ValueError("Cannot fit traj_dist on 1 sample")

        # import matplotlib.pyplot as plt
        # plt.plot(U[1, :])
        # plt.show(block=False)
        # raw_input(U.shape)

        pol_mu = U
        pol_sig = np.zeros((N, T, dU, dU))

        for t in range(T):
            # Using only diagonal covariances
            pol_sig[:, t, :, :] = np.tile(np.diag(np.diag(np.cov(U[:, t, :].T))), (N, 1, 1))

        # Collapse policy covariances. (This is only correct because the policy doesn't depend on state).
        pol_sig = np.mean(pol_sig, axis=0)

        # Allocate.
        pol_K = np.zeros([T, dU, dX])
        pol_k = np.zeros([T, dU])
        pol_S = np.zeros([T, dU, dU])
        chol_pol_S = np.zeros([T, dU, dU])
        inv_pol_S = np.zeros([T, dU, dU])

        # Update policy prior.
        def eval_prior(Ts, Ps):
            strength = 1e-4
            dX, dU = Ts.shape[-1], Ps.shape[-1]
            prior_fd = np.zeros((dU, dX))
            prior_cond = 1e-5 * np.eye(dU)
            sig = np.eye(dX)
            Phi = strength * np.vstack([np.hstack([sig, sig.dot(prior_fd.T)]),
                                        np.hstack([prior_fd.dot(sig), prior_fd.dot(sig).dot(prior_fd.T) + prior_cond])])
            return np.zeros(dX+dU), Phi, 0, strength


        # Fit linearization with least squares regression
        dwts = (1.0 / N) * np.ones(N)
        for t in range(T):
            Ts = X[:, t, :]
            Ps = pol_mu[:, t, :]
            Ys = np.concatenate([Ts, Ps], axis=1)
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = eval_prior(Ts, Ps)
            sig_reg = np.zeros((dX+dU, dX+dU))
            # Slightly regularize on first timestep.
            if t == 0:
                sig_reg[:dX, :dX] = 1e-8
            pol_K[t, :, :], pol_k[t, :], pol_S[t, :, :] = gauss_fit_joint_prior(Ys, mu0, Phi, mm, n0, dwts, dX, dU, sig_reg)
        pol_S += pol_sig  # Add policy covariances mean

        for t in range(T):
            chol_pol_S[t, :, :] = sp.linalg.cholesky(pol_S[t, :, :])
            inv_pol_S[t, :, :] = np.linalg.inv(pol_S[t, :, :])

        return LinearGaussianPolicy(pol_K, pol_k, pol_S, chol_pol_S, inv_pol_S)

    # Replacing self._update_trajectories in gps.py
    def _update_trajectories(self):
        """
        Compute new linear Gaussian controllers.
        """
        print('-->Updating trajectories (local policies)...')
        if self.new_traj_distr is None:
            self.new_traj_distr = [self.cur[cond].traj_distr for cond in range(self.M)]
        for cond in range(self.M):
            self.new_traj_distr[cond], self.cur[cond].eta, self.cur[cond].omega, self.cur[cond].nu = self.traj_opt.update(cond, self)
