""" This file defines the mDREPS-based trajectory optimization method. """

import sys
import numpy as np

from robolearn.algos.gps.gps import GPS
from robolearn.algos.trajopt.trajopt_config import default_mdreps_hyperparams


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

        self.good_trajectories = [[None, None] for _ in range(self.M)]
        self.bad_trajectories = [[None, None] for _ in range(self.M)]

    def iteration(self, sample_lists):
        """
        Run iteration of mDREPS.
        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]

        # Update dynamics model using all samples.
        print('->Updating dynamics linearization...')
        self._update_dynamics()

        print('->Updating KL step size...')
        self._update_step_size()  # KL Divergence step size.

        print('->Getting good and bad trajectories...')
        for m in range(self.M):
            self._get_good_trajectories(m)  # KL Divergence step size.
            self._get_bad_trajectories(m)  # KL Divergence step size.


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
        good_distr = self.good_trajectories[m]
        bad_distr = self.bad_trajectories[m]
        if not augment:  # Whether to augment cost with term to penalize KL
            return traj_info.Cm, traj_info.cv

        multiplier = self._hyperparams['max_ent_traj']  # Weight of maximum entropy term in trajectory optimization

        # TVLGC terms from previous traj
        K, ipc, k = traj_distr.K, traj_distr.inv_pol_covar, traj_distr.k
        # TVLGC terms from good traj
        K_good, ipc_good, k_good = good_distr.K, good_distr.inv_pol_covar, good_distr.k
        # TVLGC terms from bad traj
        K_bad, ipc_bad, k_bad = bad_distr.K, bad_distr.inv_pol_covar, bad_distr.k

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
            fcv[t, :] += eta / (eta + multiplier) * np.hstack([
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
            fcv[t, :] += omega / (eta + multiplier) * np.hstack([
                K_good[t, :, :].T.dot(ipc_good[t, :, :]).dot(k_good[t, :]),
                -ipc_good[t, :, :].dot(k_good[t, :])
            ])

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
            fcv[t, :] -= nu / (eta + multiplier) * np.hstack([
                K_bad[t, :, :].T.dot(ipc_bad[t, :, :]).dot(k_bad[t, :]),
                -ipc_bad[t, :, :].dot(k_bad[t, :])
            ])

        return fCm, fcv

    def _get_good_trajectories(self, cond):
        # Sample costs estimate.
        cc = self.cur[cond].traj_info.cc
        cv = self.cur[cond].traj_info.cv
        Cm = self.cur[cond].traj_info.Cm
        cs = self.cur[cond].cs

        print(self.cur[cond].cs.shape)
        print(self.cur[cond].cs)
        # Get Return of each sample
        best_index = np.argmin(np.sum(cs, axis=1))
        best_return = np.sum(cs[best_index])
        print(np.sum(self.cur[cond], axis=1))
        print(best_index)
        print(type(self.cur[cond].sample_list[best_index]))
        raw_input('chiquitin')

        # Get current best trajectory
        if self.good_trajectories[cond][0] is None:
            self.good_trajectories[cond][0] = self.cur[cond].sample_list[best_index]
            self.good_trajectories[cond][1] = best_return
        else:
            # Update only if it is better than before
            if self.good_trajectories[cond][1] > best_return:
                self.good_trajectories[cond][0] = self.cur[cond].sample_list[best_index]
                self.good_trajectories[cond][1] = best_return




    def _get_bad_trajectories(self, cond):
        # Sample costs estimate.
        cc = self.cur[cond].traj_info.cc
        cv = self.cur[cond].traj_info.cv
        Cm = self.cur[cond].traj_info.Cm
        cs = self.cur[cond].cs

        # Get Return of each sample
        worst_index = np.argmin(np.sum(cs, axis=1))
        worst_return = np.sum(cs[worst_index])

        # Get current worst trajectory
        # Get current worst trajectory
        if self.bad_trajectories[cond][0] is None:
            self.bad_trajectories[cond][0] = self.cur[cond].sample_list[best_index]
            self.bad_trajectories[cond][1] = worst_return
        else:
            # Update only if it is better than before
            if self.bad_trajectories[cond][1] > worst_return:
                self.bad_trajectories[cond][0] = self.cur[cond].sample_list[best_index]
                self.bad_trajectories[cond][1] = worst_return
