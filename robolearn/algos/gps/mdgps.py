"""
# ################### #
# ################### #
# ###### MDGPS ###### #
# ################### #
# ################### #
MDGPS algorithm. 
Author: C.Finn et al
Reference:
W. Montgomery, S. Levine
Guided Policy Search as Approximate Mirror Descent. 2016. https://arxiv.org/abs/1607.04614
"""

import sys
import numpy as np
import scipy as sp
import copy

from robolearn.algos.gps.gps import GPS
from robolearn.algos.gps.gps_config import default_mdgps_hyperparams
from robolearn.algos.gps.gps_utils import PolicyInfo

from robolearn.utils.sample.sample import Sample
from robolearn.utils.sample.sample_list import SampleList

import logging
LOGGER = logging.getLogger(__name__)
# Logging into console AND file
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
LOGGER.addHandler(ch)


class MDGPS(GPS):
    def __init__(self, agent, env, **kwargs):
        super(MDGPS, self).__init__(agent, env, **kwargs)

        gps_algo_hyperparams = default_mdgps_hyperparams.copy()
        gps_algo_hyperparams.update(self._hyperparams['gps_algo_hyperparams'])
        self._hyperparams.update(gps_algo_hyperparams)

        self.gps_algo = 'mdgps'

        # Policy Prior #
        # ------------ #
        policy_prior = self._hyperparams['policy_prior']
        for m in range(self.M):
            # Same policy prior type for all conditions
            self.cur[m].pol_info = PolicyInfo(self._hyperparams)
            self.cur[m].pol_info.policy_prior = policy_prior['type'](policy_prior)

    def iteration(self, sample_lists):
        """
        Run iteration of MDGPS-based guided policy search.
        :param sample_lists: List of SampleList objects for each condition.
        :return: None
        """
        # Store the samples and evaluate the costs.
        print("\n"*2)
        print('->Evaluating samples costs...')
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
            self._eval_cost(m)

        # Update dynamics linearizations (linear-Gaussian dynamics).
        print("\n"*2)
        print('->Updating dynamics linearization...')
        self._update_dynamics()

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.iteration_count == 0:
            self.new_traj_distr = [self.cur[cond].traj_distr for cond in range(self.M)]
            print("\n"*2)
            print('->S-step for init_traj_distribution (iter=0)...')
            self.update_policy()

        # Update global policy linearizations.
        print("\n"*2)
        print('->Updating global policy linearization...')
        for m in range(self.M):
            self.update_policy_fit(m)

        # C-step
        if self.iteration_count > 0:
            print("\n"*2)
            print('-->Adjust step size (epsilon) multiplier...')
            self.stepadjust()
        print("\n"*2)
        print('->| C-step |<-')
        self._update_trajectories()

        # S-step
        print("\n"*2)
        print('->| S-step |<-')
        self.update_policy()

        # Prepare for next iteration
        self.advance_iteration_variables()

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

    def advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', reinitialize 'cur' variables, and advance iteration counter.
        :return: None
        """
        self._advance_iteration_variables()
        for m in range(self.M):
            self.cur[m].traj_info.last_kl_step = self.prev[m].traj_info.last_kl_step
            self.cur[m].pol_info = copy.deepcopy(self.prev[m].pol_info)

    def stepadjust(self):
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

    def compute_costs(self, m, eta, augment=True):
        """ Compute cost estimates used in the LQR backward pass. """

        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr
        if not augment:  # Whether to augment cost with term to penalize KL
            return traj_info.Cm, traj_info.cv

        pol_info = self.cur[m].pol_info
        multiplier = self._hyperparams['max_ent_traj']
        T, dU, dX = traj_distr.T, traj_distr.dU, traj_distr.dX
        Cm, cv = np.copy(traj_info.Cm), np.copy(traj_info.cv)

        PKLm = np.zeros((T, dX+dU, dX+dU))
        PKLv = np.zeros((T, dX+dU))
        fCm, fcv = np.zeros(Cm.shape), np.zeros(cv.shape)
        for t in range(T):
            # Policy KL-divergence terms.
            inv_pol_S = np.linalg.solve(pol_info.chol_pol_S[t, :, :],
                                        np.linalg.solve(pol_info.chol_pol_S[t, :, :].T, np.eye(dU)))
            KB, kB = pol_info.pol_K[t, :, :], pol_info.pol_k[t, :]
            PKLm[t, :, :] = np.vstack([np.hstack([KB.T.dot(inv_pol_S).dot(KB), -KB.T.dot(inv_pol_S)]),
                                       np.hstack([-inv_pol_S.dot(KB), inv_pol_S])])
            PKLv[t, :] = np.concatenate([KB.T.dot(inv_pol_S).dot(kB),
                                         -inv_pol_S.dot(kB)])
            fCm[t, :, :] = (Cm[t, :, :] + PKLm[t, :, :] * eta) / (eta + multiplier)
            fcv[t, :] = (cv[t, :] + PKLv[t, :] * eta) / (eta + multiplier)

        return fCm, fcv
