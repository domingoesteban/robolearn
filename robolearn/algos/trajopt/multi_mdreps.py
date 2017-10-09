""" This file defines the mDREPS-based trajectory optimization method. """

import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import copy

from robolearn.algos.gps.gps import GPS
from robolearn.algos.trajopt.trajopt_config import default_mdreps_hyperparams
from robolearn.algos.gps.gps_utils import IterationData, TrajectoryInfo, extract_condition, DualityInfo
from robolearn.algos.gps.gps_utils import PolicyInfo
from robolearn.utils.sample.sample import Sample
from robolearn.utils.sample.sample_list import SampleList
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

        # Use inititial dual variables
        for m in range(self.M):
            self.cur[m].eta = self._hyperparams['init_eta']
            self.cur[m].nu = self._hyperparams['init_nu']
            self.cur[m].omega = self._hyperparams['init_omega']

        self.base_kl_good = self._hyperparams['base_kl_good']
        self.base_kl_bad = self._hyperparams['base_kl_bad']

        self.use_global_policy = self._hyperparams['use_global_policy']

        if self.use_global_policy:
            policy_prior = self._hyperparams['policy_prior']
            for m in range(self.M):
                # Same policy prior type for all conditions
                self.cur[m].pol_info = PolicyInfo(self._hyperparams)
                self.cur[m].pol_info.policy_prior = policy_prior['type'](policy_prior)

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


        # Evaluate cost function for all conditions and samples.
        print('')
        print('->Evaluating samples costs...')
        for m in range(self.M):
            self._eval_cost(m)

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.use_global_policy and self.iteration_count == 0:
            self.new_traj_distr = [self.cur[cond].traj_distr for cond in range(self.M)]
            print("\n"*2)
            print('->S-step for init_traj_distribution (iter=0)...')
            self.update_policy()

        # Update global policy linearizations.
        print('')
        if self.use_global_policy:
            print('->Updating global policy linearization...')
            for m in range(self.M):
                self.update_policy_fit(m)

        print('')
        if self.use_global_policy and self.iteration_count > 0:
            print('->Updating KL step size with GLOBAL policy...')
            self._update_step_size_global_policy()
        else:
            print('->Updating KL step size with previous LOCAL policy...')
            self._update_step_size()  # KL Divergence step size.

        print('')
        print('->Getting good and bad trajectories...')
        self._get_good_trajectories(option=self._hyperparams['good_traj_selection_type'])
        self._get_bad_trajectories(option=self._hyperparams['bad_traj_selection_type'])

        print('')
        print('->Updating data of good and bad trajectories...')
        print('-->Update dynamics...')
        self._update_good_bad_dynamics()
        print('-->Update costs...')
        self._eval_good_bad_costs()
        print('-->Update traj dist...')
        self._fit_good_bad_traj_dist()

        # Run inner loop to compute new policies.
        print('')
        print('->Updating trajectories...')
        for ii in range(self._hyperparams['inner_iterations']):
            print('-->Inner iteration %d/%d' % (ii+1, self._hyperparams['inner_iterations']))
            self._update_trajectories()

        if self.use_global_policy:
            print('')
            print('->| S-step |<-')
            self.update_policy()

        self.advance_duality_iteration_variables()

    def _update_step_size(self):
        """ Evaluate costs on samples, and adjust the step size. """
        # Evaluate cost function for all conditions and samples.
        #for m in range(self.M):
        #    self._eval_cost(m)

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

    def _update_step_size_global_policy(self):
        """
        Calculate new step sizes. This version uses the same step size for all conditions.
        """
        # Compute previous cost and previous expected cost.
        prev_M = len(self.prev)  # May be different in future.
        prev_laplace = np.empty(prev_M)
        prev_mc = np.empty(prev_M)
        prev_predicted = np.empty(prev_M)
        for m in range(prev_M):
            prev_nn = self.prev[m].pol_info.traj_distr()
            prev_lg = self.prev[m].new_traj_distr

            # Compute values under Laplace approximation. This is the policy that the previous samples were actually
            # drawn from under the dynamics that were estimated from the previous samples.
            prev_laplace[m] = self.traj_opt.estimate_cost(prev_nn, self.prev[m].traj_info).sum()
            # This is the actual cost that we experienced.
            prev_mc[m] = self.prev[m].cs.mean(axis=0).sum()
            # This is the policy that we just used under the dynamics that were estimated from the prev samples (so
            # this is the cost we thought we would have).
            prev_predicted[m] = self.traj_opt.estimate_cost(prev_lg, self.prev[m].traj_info).sum()

        # Compute current cost.
        cur_laplace = np.empty(self.M)
        cur_mc = np.empty(self.M)
        for m in range(self.M):
            cur_nn = self.cur[m].pol_info.traj_distr()
            # This is the actual cost we have under the current trajectory based on the latest samples.
            cur_laplace[m] = self.traj_opt.estimate_cost(cur_nn, self.cur[m].traj_info).sum()
            cur_mc[m] = self.cur[m].cs.mean(axis=0).sum()

        # Compute predicted and actual improvement.
        prev_laplace = prev_laplace.mean()
        prev_mc = prev_mc.mean()
        prev_predicted = prev_predicted.mean()
        cur_laplace = cur_laplace.mean()
        cur_mc = cur_mc.mean()
        if self._hyperparams['step_rule'] == 'laplace':
            predicted_impr = prev_laplace - prev_predicted
            actual_impr = prev_laplace - cur_laplace
        elif self._hyperparams['step_rule'] == 'mc':
            predicted_impr = prev_mc - prev_predicted
            actual_impr = prev_mc - cur_mc
        LOGGER.debug('Previous cost: Laplace: %f, MC: %f',
                     prev_laplace, prev_mc)
        LOGGER.debug('Predicted cost: Laplace: %f', prev_predicted)
        LOGGER.debug('Actual cost: Laplace: %f, MC: %f',
                     cur_laplace, cur_mc)

        for m in range(self.M):
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

        # print("TODO: SETTING DUMMY IPC GOOD AND BAD")
        # for t in range(self.T):
        #     ipc_good[t, :, :] = np.eye(7)
        #     ipc_bad[t, :, :] = np.eye(7)

        # omega = 0
        # nu = 0

        # Surrogate cost
        fCm = traj_info.Cm / (eta + omega - nu + multiplier)
        fcv = traj_info.cv / (eta + omega - nu + multiplier)

        # idx_u = slice(self.dX, self.dX+self.dU)
        # print("")
        # print(fCm[-1, idx_u, idx_u])
        # print("Original fCm/eta[%d] PD?: %s" % (-1, np.all(np.linalg.eigvals(fCm[-1, idx_u, idx_u]) > 0)))

        # Add in the KL divergence with previous policy.
        for t in range(self.T - 1, -1, -1):
            fCm[t, :, :] += eta / (eta + omega - nu + multiplier) * np.vstack([
                np.hstack([  # dX x (dX + dU)
                    K[t, :, :].T.dot(ipc[t, :, :]).dot(K[t, :, :]),
                    -K[t, :, :].T.dot(ipc[t, :, :])
                ]),
                np.hstack([  # dU x (dX + dU)
                    -ipc[t, :, :].dot(K[t, :, :]), ipc[t, :, :]
                ])
            ])
            fcv[t, :] += eta / (eta + omega - nu + multiplier) * np.hstack([
                K[t, :, :].T.dot(ipc[t, :, :]).dot(k[t, :]),  # dX
                -ipc[t, :, :].dot(k[t, :])  # dU
            ])

        # print("")
        # print(ipc[-1, :, :])
        # print("-")
        # print(fCm[-1, idx_u, idx_u])
        # print("Surrogate fCm/eta[%d] PD?: %s" % (-1, np.all(np.linalg.eigvals(fCm[-1, idx_u, idx_u]) > 0)))
        # raw_input("avercomo")


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

        # print("")
        # print(ipc_good[-1, :, :])
        # print("-")
        # print(fCm[-1, idx_u, idx_u])
        # print("Surrogate fCm/omega[%d] PD?: %s" % (-1, np.all(np.linalg.eigvals(fCm[-1, idx_u, idx_u]) > 0)))
        # raw_input("avercomo")

        # Subtract in the KL divergence with bad trajectories.
        for t in range(self.T - 1, -1, -1):
            fCm[t, :, :] -= nu / (eta + omega - nu + multiplier) * np.vstack([
                np.hstack([
                    K_bad[t, :, :].T.dot(ipc_bad[t, :, :]).dot(K_bad[t, :, :]),
                    -K_bad[t, :, :].T.dot(ipc_bad[t, :, :])
                ]),
                np.hstack([
                    -ipc_bad[t, :, :].dot(K_bad[t, :, :]), ipc_bad[t, :, :]
                ])
            ])
            fcv[t, :] -= nu / (eta + omega - nu + multiplier) * np.hstack([
                K_bad[t, :, :].T.dot(ipc_bad[t, :, :]).dot(k_bad[t, :]),
                -ipc_bad[t, :, :].dot(k_bad[t, :])
            ])

        # print("")
        # print(ipc_bad[-1, :, :])
        # print("-")
        # print(fCm[-1, idx_u, idx_u])
        # print("Surrogate fCm/nu[%d] PD?: %s" % (-1, np.all(np.linalg.eigvals(fCm[-1, idx_u, idx_u]) > 0)))
        # raw_input("avercomo")

        return fCm, fcv

    def _get_good_trajectories(self, option='only_traj'):
        """
        
        :param option: 'only_traj' or 'all'
        :return: 
        """
        for cond in range(self.M):
            # Sample costs estimate.
            cs = self.cur[cond].cs
            sample_list = self.cur[cond].sample_list

            # Get index of sample with best Return
            #best_index = np.argmin(np.sum(cs, axis=1))
            n_good = self._hyperparams['n_good_samples']
            if n_good == cs.shape[0]:
                best_indeces = range(n_good)
            else:
                best_indeces = np.argpartition(np.sum(cs, axis=1), n_good)[:n_good]

            # Get current best trajectory
            if self.good_duality_info[cond].sample_list is None:
                for gg, good_index in enumerate(best_indeces):
                    print("Defining GOOD trajectory sample %d | cur_cost=%f" % (gg, np.sum(cs[good_index, :])))
                self.good_duality_info[cond].sample_list = SampleList([sample_list[good_index] for good_index in best_indeces])
                self.good_duality_info[cond].samples_cost = cs[best_indeces, :]
            else:
                # Update only if it is better than previous traj_dist
                if option == 'only_traj':
                    # If there is a better trajectory, replace only that trajectory to previous ones
                    for good_index in best_indeces:
                        least_best_index = np.argpartition(np.sum(self.good_duality_info[cond].samples_cost, axis=1), -1)[-1:]
                        if np.sum(self.good_duality_info[cond].samples_cost[least_best_index, :]) > np.sum(cs[good_index, :]):
                            print("Updating GOOD trajectory sample %d | cur_cost=%f > new_cost=%f"
                                  % (least_best_index,
                                     np.sum(self.good_duality_info[cond].samples_cost[least_best_index, :]),
                                     np.sum(cs[good_index, :])))
                            self.good_duality_info[cond].sample_list.set_sample(least_best_index, sample_list[good_index])
                            self.good_duality_info[cond].samples_cost[least_best_index, :] = cs[good_index, :]
                elif option == 'always':
                    for gg, good_index in enumerate(best_indeces):
                        print("Updating GOOD trajectory sample %d | cur_cost=%f > new_cost=%f"
                              % (gg, np.sum(self.good_duality_info[cond].samples_cost[gg, :]),
                                 np.sum(cs[good_index, :])))
                        self.good_duality_info[cond].sample_list.set_sample(gg, sample_list[good_index])
                        self.good_duality_info[cond].samples_cost[gg, :] = cs[good_index, :]
                else:
                    raise ValueError("Wrong get_good_grajectories option: %s" % option)

    def _get_bad_trajectories(self, option='only_traj'):
        for cond in range(self.M):
            # Sample costs estimate.
            cs = self.cur[cond].cs
            sample_list = self.cur[cond].sample_list

            # Get index of sample with worst Return
            #worst_index = np.argmax(np.sum(cs, axis=1))
            n_bad = self._hyperparams['n_bad_samples']
            if n_bad == cs.shape[0]:
                worst_indeces = range(n_bad)
            else:
                worst_indeces = np.argpartition(np.sum(cs, axis=1), -n_bad)[-n_bad:]

            # Get current best trajectory
            if self.bad_duality_info[cond].sample_list is None:
                for bb, bad_index in enumerate(worst_indeces):
                    print("Defining BAD trajectory sample %d | cur_cost=%f" % (bb, np.sum(cs[bad_index, :])))
                self.bad_duality_info[cond].sample_list = SampleList([sample_list[bad_index] for bad_index in worst_indeces])
                self.bad_duality_info[cond].samples_cost = cs[worst_indeces, :]
            else:
                # Update only if it is better than before
                if option == 'only_traj':
                    for bad_index in worst_indeces:
                        least_worst_index = np.argpartition(np.sum(self.bad_duality_info[cond].samples_cost, axis=1), 1)[:1]
                        if np.sum(self.bad_duality_info[cond].samples_cost[least_worst_index, :]) < np.sum(cs[bad_index, :]):
                            print("Updating BAD trajectory sample %d | cur_cost=%f < new_cost=%f"
                                  % (least_worst_index,
                                     np.sum(self.bad_duality_info[cond].samples_cost[least_worst_index, :]),
                                     np.sum(cs[bad_index, :])))
                            self.bad_duality_info[cond].sample_list.set_sample(least_worst_index, sample_list[bad_index])
                            self.bad_duality_info[cond].samples_cost[least_worst_index, :] = cs[bad_index, :]
                elif option == 'always':
                    for bb, bad_index in enumerate(worst_indeces):
                        print("Updating BAD trajectory sample %d | cur_cost=%f < new_cost=%f"
                              % (bb, np.sum(self.bad_duality_info[cond].samples_cost[bb, :]),
                                 np.sum(cs[bad_index, :])))
                        self.bad_duality_info[cond].sample_list.set_sample(bb, sample_list[bad_index])
                        self.bad_duality_info[cond].samples_cost[bb, :] = cs[bad_index, :]
                else:
                    raise ValueError("Wrong get_bad_grajectories option: %s" % option)

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
        min_good_var = self._hyperparams['min_good_var']
        min_bad_var = self._hyperparams['min_good_var']

        for cond in range(self.M):
            self.good_duality_info[cond].traj_dist = self.fit_traj_dist(self.good_duality_info[cond].sample_list, min_good_var)
            self.bad_duality_info[cond].traj_dist = self.fit_traj_dist(self.bad_duality_info[cond].sample_list, min_bad_var)

    @staticmethod
    def fit_traj_dist(sample_list, min_variance):
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

        print("TODO: WE ARE GIVING MIN GOOD/BAD VARIANCE")
        for t in range(T):
            # Using only diagonal covariances
            # pol_sig[:, t, :, :] = np.tile(np.diag(np.diag(np.cov(U[:, t, :].T))), (N, 1, 1))
            current_diag = np.diag(np.cov(U[:, t, :].T))
            new_diag = np.max(np.vstack((current_diag, min_variance)), axis=0)
            pol_sig[:, t, :, :] = np.tile(np.diag(new_diag), (N, 1, 1))

        # Collapse policy covariances. (This is only correct because the policy doesn't depend on state).
        pol_sig = np.mean(pol_sig, axis=0)

        # print(pol_sig)
        # raw_input('perhkhjk')

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

        # plt.subplots()
        # plt.plot(pol_S[:, 0, 0], label='0')
        # plt.plot(pol_S[:, 1, 1], label='1')
        # plt.plot(pol_S[:, 2, 2], label='2')
        # plt.plot(pol_S[:, 3, 3], label='3')
        # plt.plot(pol_S[:, 4, 4], label='4')
        # plt.plot(pol_S[:, 5, 5], label='5')
        # plt.legend()
        # plt.show(block=False)
        # raw_input('ploteandooo')

        for t in range(T):
            chol_pol_S[t, :, :] = sp.linalg.cholesky(pol_S[t, :, :])
            inv_pol_S[t, :, :] = np.linalg.inv(pol_S[t, :, :])

        # plt.subplots()
        # plt.plot(inv_pol_S[:, 0, 0], label='0')
        # plt.plot(inv_pol_S[:, 1, 1], label='1')
        # plt.plot(inv_pol_S[:, 2, 2], label='2')
        # plt.plot(inv_pol_S[:, 3, 3], label='3')
        # plt.plot(inv_pol_S[:, 4, 4], label='4')
        # plt.plot(inv_pol_S[:, 5, 5], label='5')
        # plt.legend()
        # plt.show(block=False)
        # raw_input('ploteandooo')

        # max_inv_pol = np.zeros(7)
        # max_inv_pol[0] = np.argmax(inv_pol_S[:, 0, 0])
        # max_inv_pol[1] = np.argmax(inv_pol_S[:, 1, 1])
        # max_inv_pol[2] = np.argmax(inv_pol_S[:, 2, 2])
        # max_inv_pol[3] = np.argmax(inv_pol_S[:, 3, 3])
        # max_inv_pol[4] = np.argmax(inv_pol_S[:, 4, 4])
        # max_inv_pol[5] = np.argmax(inv_pol_S[:, 5, 5])
        # max_inv_pol[6] = np.argmax(inv_pol_S[:, 6, 6])

        # print("%f, %d" % (np.max(inv_pol_S[:, 0, 0]), np.argmax(inv_pol_S[:, 0, 0])))
        # print("%f, %d" % (np.max(inv_pol_S[:, 1, 1]), np.argmax(inv_pol_S[:, 1, 1])))
        # print("%f, %d" % (np.max(inv_pol_S[:, 2, 2]), np.argmax(inv_pol_S[:, 2, 2])))
        # print("%f, %d" % (np.max(inv_pol_S[:, 3, 3]), np.argmax(inv_pol_S[:, 3, 3])))
        # print("%f, %d" % (np.max(inv_pol_S[:, 4, 4]), np.argmax(inv_pol_S[:, 4, 4])))
        # print("%f, %d" % (np.max(inv_pol_S[:, 5, 5]), np.argmax(inv_pol_S[:, 5, 5])))
        # print("%f, %d" % (np.max(inv_pol_S[:, 6, 6]), np.argmax(inv_pol_S[:, 6, 6])))

        # print("COVARIANCES:")
        # print("%f, %d" % (pol_S[int(max_inv_pol[0]), 0, 0], max_inv_pol[0]))
        # print("%f, %d" % (pol_S[int(max_inv_pol[1]), 1, 1], max_inv_pol[1]))
        # print("%f, %d" % (pol_S[int(max_inv_pol[2]), 2, 2], max_inv_pol[2]))
        # print("%f, %d" % (pol_S[int(max_inv_pol[3]), 3, 3], max_inv_pol[3]))
        # print("%f, %d" % (pol_S[int(max_inv_pol[4]), 4, 4], max_inv_pol[4]))
        # print("%f, %d" % (pol_S[int(max_inv_pol[5]), 5, 5], max_inv_pol[5]))
        # print("%f, %d" % (pol_S[int(max_inv_pol[6]), 6, 6], max_inv_pol[6]))

        # raw_input("AAAA")

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

    def log_duality_vars(self, itr):
            print("Logging God/Bad duality data")
            self.data_logger.pickle(
                ('good_trajectories_info_itr_%02d.pkl' % itr),
                copy.copy(self.good_trajectories_info)
            )
            self.data_logger.pickle(
                ('bad_trajectories_info_itr_%02d.pkl' % itr),
                copy.copy(self.bad_trajectories_info)
            )
            self.data_logger.pickle(
                ('good_duality_info_itr_%02d.pkl' % itr),
                copy.copy(self.good_duality_info)
            )
            self.data_logger.pickle(
                ('bad_duality_info_itr_%02d.pkl' % itr),
                copy.copy(self.bad_duality_info)
            )

    def load_duality_vars(self, itr):
        print("Loading Duality data")
        good_trajectories_file = 'good_trajectories_info_itr_%02d.pkl' % itr
        self.good_trajectories_info = self.data_logger.unpickle(good_trajectories_file)
        bad_trajectories_file = 'bad_trajectories_info_itr_%02d.pkl' % itr
        self.bad_trajectories_info = self.data_logger.unpickle(bad_trajectories_file)
        good_duality_file = 'good_duality_info_itr_%02d.pkl' % itr
        self.good_duality_info = self.data_logger.unpickle(good_duality_file)
        bad_duality_file = 'bad_duality_info_itr_%02d.pkl' % itr
        self.bad_duality_info = self.data_logger.unpickle(bad_duality_file)

    def advance_duality_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', reinitialize 'cur' variables, and advance iteration counter.
        :return: None
        """
        self._advance_iteration_variables()
        for m in range(self.M):
            self.cur[m].nu = self.prev[m].nu
            self.cur[m].omega = self.prev[m].omega

            if self.use_global_policy:
                self.cur[m].traj_info.last_kl_step = self.prev[m].traj_info.last_kl_step
                self.cur[m].pol_info = copy.deepcopy(self.prev[m].pol_info)

    def update_policy(self):
        """
        Computes(updates) a new global policy.
        :return: 
        """
        print('-->Updating Global policy...')
        dU, dO, T = self.dU, self.dO, self.T
        # Compute target mean, cov(precision), and weight for each sample; and concatenate them.
        obs_data, tgt_mu = np.zeros((0, T, dO)), np.zeros((0, T, dU))
        tgt_prc, tgt_wt = np.zeros((0, T, dU, dU)), np.zeros((0, T))
        for m in range(self.M):
            samples = self.cur[m].sample_list
            X = samples.get_states()
            N = len(samples)
            traj, pol_info = self.new_traj_distr[m], self.cur[m].pol_info
            mu = np.zeros((N, T, dU))
            prc = np.zeros((N, T, dU, dU))
            wt = np.zeros((N, T))
            # Get time-indexed actions.
            for t in range(T):
                # Compute actions along this trajectory.
                prc[:, t, :, :] = np.tile(traj.inv_pol_covar[t, :, :], [N, 1, 1])
                for i in range(N):
                    mu[i, t, :] = (traj.K[t, :, :].dot(X[i, t, :]) + traj.k[t, :])

                wt[:, t].fill(pol_info.pol_wt[t])

            tgt_mu = np.concatenate((tgt_mu, mu))
            tgt_prc = np.concatenate((tgt_prc, prc))
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs_data = np.concatenate((obs_data, samples.get_obs()))

        self.policy_opt.update(obs_data, tgt_mu, tgt_prc, tgt_wt)

    def update_policy_fit(self, cond):
        """
        Re-estimate the local policy values in the neighborhood of the trajectory.
        :param cond: Condition
        :return: None
        """
        dX, dU, T = self.dX, self.dU, self.T
        # Choose samples to use.
        samples = self.cur[cond].sample_list
        N = len(samples)
        pol_info = self.cur[cond].pol_info
        X = samples.get_states().copy()
        obs = samples.get_obs().copy()
        pol_mu, pol_sig = self.policy_opt.prob(obs)[:2]
        pol_info.pol_mu, pol_info.pol_sig = pol_mu, pol_sig

        # Update policy prior.
        policy_prior = pol_info.policy_prior
        samples = SampleList(self.cur[cond].sample_list)
        mode = self._hyperparams['policy_sample_mode']
        policy_prior.update(samples, self.policy_opt, mode)

        # Fit linearization and store in pol_info.
        pol_info.pol_K, pol_info.pol_k, pol_info.pol_S = policy_prior.fit(X, pol_mu, pol_sig)
        for t in range(T):
            pol_info.chol_pol_S[t, :, :] = sp.linalg.cholesky(pol_info.pol_S[t, :, :])
