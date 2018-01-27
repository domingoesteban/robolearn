import numpy as np
from robolearn.algos.gps.gps_utils import PolicyInfo, extract_condition
from robolearn.algos.gps.gps_utils import TrajectoryInfo, DualityInfo
from robolearn.utils.traj_opt.traj_opt_utils import traj_distr_kl, traj_distr_kl_alt
from robolearn.utils.plot_utils import lqr_forward
from robolearn.utils.policy_utils import fit_linear_gaussian_policy

from robolearn.utils.sample.sample import Sample
from robolearn.utils.sample.sample_list import SampleList


class Dualism(object):
    def __init__(self):
        # Duality data with: [sample_list, samples_cost, cs_traj, traj_dist, pol_info]
        self.good_duality_info = [DualityInfo() for _ in range(self.M)]
        self.bad_duality_info = [DualityInfo() for _ in range(self.M)]

        # TrajectoryInfo for good and bad trajectories
        self.good_trajs_info = [None for _ in range(self.M)]
        self.bad_trajs_info = [None for _ in range(self.M)]
        for m in range(self.M):
            self.good_trajs_info[m] = TrajectoryInfo()
            self.bad_trajs_info[m] = TrajectoryInfo()

            if self._hyperparams['fit_dynamics']:
                dynamics = self._hyperparams['dynamics']
                self.good_trajs_info[m].dynamics = dynamics['type'](dynamics)
                self.bad_trajs_info[m].dynamics = dynamics['type'](dynamics)

            # TODO: Use demonstration trajectories
            # # Get the initial trajectory distribution hyperparams
            # init_traj_distr = extract_condition(self._hyperparams['init_traj_distr'], self._train_cond_idx[m])
            # Instantiate Trajectory Distribution: init_lqr or init_pd
            # self.good_duality_infor = init_traj_distr['type'](init_traj_distr)
            # self.bad_duality_infor = init_traj_distr['type'](init_traj_distr)

            # TODO: Using same init traj
            # Get the initial trajectory distribution hyperparams
            # init_traj_distr = self._hyperparams['init_traj_distr']
            init_traj_distr = extract_condition(self._hyperparams['init_traj_distr'],
                                                self._train_cond_idx[m])
            # Instantiate Trajectory Distribution: init_lqr or init_pd
            self.good_duality_info[m].traj_dist = init_traj_distr['type'](init_traj_distr)
            self.bad_duality_info[m].traj_dist = init_traj_distr['type'](init_traj_distr)

            # Same policy prior than GlobalPol for good/bad
            self._hyperparams['algo_hyperparams']['T'] = self.T
            self._hyperparams['algo_hyperparams']['dU'] = self.dU
            self._hyperparams['algo_hyperparams']['dX'] = self.dX
            policy_prior = self._hyperparams['algo_hyperparams']['policy_prior']
            self.good_duality_info[m].pol_info = \
                PolicyInfo(self._hyperparams['algo_hyperparams'])
            self.good_duality_info[m].pol_info.policy_prior = \
                policy_prior['type'](policy_prior)
            self.bad_duality_info[m].pol_info = \
                PolicyInfo(self._hyperparams['algo_hyperparams'])
            self.bad_duality_info[m].pol_info.policy_prior = \
                policy_prior['type'](policy_prior)

    def _get_bad_samples(self, option='only_traj'):
        """
        Get bad trajectory samples.
        :param option(str):
                 'only_traj': update bad_duality_info sample list only when the
                              trajectory sample is worse than any previous
                              sample.
                 'always': Update bad_duality_info sample list with the worst
                           trajectory samples in the current iteration.

        :return:
        """

        LOGGER = self.logger

        for cond in range(self.M):
            # Sample costs estimate.
            cs = self.cur[cond].cs
            sample_list = self.cur[cond].sample_list

            # Get index of sample with worst Return
            #worst_index = np.argmax(np.sum(cs, axis=1))
            n_bad = self._hyperparams['algo_hyperparams']['n_bad_samples']
            if n_bad == cs.shape[0]:
                worst_indeces = range(n_bad)
            else:
                worst_indeces = np.argpartition(np.sum(cs, axis=1), -n_bad)[-n_bad:]

            # Get current worst sample
            if self.bad_duality_info[cond].sample_list is None:
                for bb, bad_index in enumerate(worst_indeces):
                    LOGGER.info("DualGPS: Defining BAD trajectory sample %d | "
                                "cur_cost=%f from sample %d"
                                % (bb, np.sum(cs[bad_index, :]), bad_index))
                self.bad_duality_info[cond].sample_list = SampleList([sample_list[bad_index] for bad_index in worst_indeces])
                self.bad_duality_info[cond].samples_cost = cs[worst_indeces, :]
            else:
                # Update only if it is better than before
                if option == 'only_traj':
                    for bad_index in worst_indeces:
                        least_worst_index = np.argpartition(np.sum(self.bad_duality_info[cond].samples_cost, axis=1), 1)[:1]
                        if np.sum(self.bad_duality_info[cond].samples_cost[least_worst_index, :]) < np.sum(cs[bad_index, :]):
                            LOGGER.info("DualGPS: Updating BAD trajectory sample %d | cur_cost=%f < new_cost=%f"
                                        % (least_worst_index,
                                           np.sum(self.bad_duality_info[cond].samples_cost[least_worst_index, :]),
                                           np.sum(cs[bad_index, :])))
                            self.bad_duality_info[cond].sample_list.set_sample(least_worst_index, sample_list[bad_index])
                            self.bad_duality_info[cond].samples_cost[least_worst_index, :] = cs[bad_index, :]
                elif option == 'always':
                    for bb, bad_index in enumerate(worst_indeces):
                        print("Worst bad index is %d | and replaces %d" % (bad_index, bb))
                        LOGGER.info("DualGPS: Updating BAD trajectory sample %d | cur_cost=%f < new_cost=%f"
                                    % (bb, np.sum(self.bad_duality_info[cond].samples_cost[bb, :]),
                                       np.sum(cs[bad_index, :])))
                        self.bad_duality_info[cond].sample_list.set_sample(bb, sample_list[bad_index])
                        self.bad_duality_info[cond].samples_cost[bb, :] = cs[bad_index, :]
                else:
                    raise ValueError("DualGPS: Wrong get_bad_samples option: %s"
                                     % option)

    def _get_good_samples(self, option='only_traj'):
        """
        Get good trajectory samples.

        Args:
            option (str): 'only_traj': update good_duality_info sample list only when the trajectory sample is better
                                       than any previous sample.
                          'always': update good_duality_info sample list with the best trajectory samples in the current
                                    iteration.

        Returns:
            None

        """
        LOGGER = self.logger

        for cond in range(self.M):
            # Sample costs estimate.
            cs = self.cur[cond].cs
            sample_list = self.cur[cond].sample_list

            # Get index of sample with best Return
            #best_index = np.argmin(np.sum(cs, axis=1))
            n_good = self._hyperparams['algo_hyperparams']['n_good_samples']
            if n_good == cs.shape[0]:
                best_indeces = range(n_good)
            else:
                best_indeces = np.argpartition(np.sum(cs, axis=1), n_good)[:n_good]

            # Get current best trajectory
            if self.good_duality_info[cond].sample_list is None:
                for gg, good_index in enumerate(best_indeces):
                    LOGGER.info("DualGPS: Defining GOOD trajectory sample %d | cur_cost=%f from sample %d" % (gg,
                                                                                                              np.sum(cs[good_index, :]), good_index))
                self.good_duality_info[cond].sample_list = SampleList([sample_list[good_index] for good_index in best_indeces])
                self.good_duality_info[cond].samples_cost = cs[best_indeces, :]
            else:
                # Update only if it is better than previous traj_dist
                if option == 'only_traj':
                    # If there is a better trajectory, replace only that trajectory to previous ones
                    for good_index in best_indeces:
                        least_best_index = np.argpartition(np.sum(self.good_duality_info[cond].samples_cost, axis=1), -1)[-1:]
                        if np.sum(self.good_duality_info[cond].samples_cost[least_best_index, :]) > np.sum(cs[good_index, :]):
                            LOGGER.info("DualGPS: Updating GOOD trajectory sample %d | cur_cost=%f > new_cost=%f"
                                        % (least_best_index,
                                           np.sum(self.good_duality_info[cond].samples_cost[least_best_index, :]),
                                           np.sum(cs[good_index, :])))
                            self.good_duality_info[cond].sample_list.set_sample(least_best_index, sample_list[good_index])
                            self.good_duality_info[cond].samples_cost[least_best_index, :] = cs[good_index, :]
                elif option == 'always':
                    for gg, good_index in enumerate(best_indeces):
                        print("Best good index is %d | and replaces %d" % (good_index, gg))
                        LOGGER.info("DualGPS: Updating GOOD trajectory sample %d | cur_cost=%f > new_cost=%f"
                                    % (gg, np.sum(self.good_duality_info[cond].samples_cost[gg, :]),
                                       np.sum(cs[good_index, :])))
                        self.good_duality_info[cond].sample_list.set_sample(gg, sample_list[good_index])
                        self.good_duality_info[cond].samples_cost[gg, :] = cs[good_index, :]
                else:
                    raise ValueError("DualGPS: Wrong get_good_samples option: %s"
                                     % option)

    def _eval_good_bad_samples_costs(self):
        """
        Evaluate costs for all current samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        for cond in range(self.M):
            cost_fcn = self.cost_function[cond]
            good_sample_list = self.good_duality_info[cond].sample_list
            bad_sample_list = self.bad_duality_info[cond].sample_list

            good_true_cost, good_cost_estimate, _ = \
                self._eval_sample_list_cost(good_sample_list, cost_fcn)
            bad_true_cost, bad_cost_estimate, _ = \
                self._eval_sample_list_cost(bad_sample_list, cost_fcn)

            # True value of cost.
            self.good_duality_info[cond].traj_cost = good_true_cost
            self.bad_duality_info[cond].traj_cost = bad_true_cost

            # Cost estimate.
            self.good_trajs_info[cond].Cm = good_cost_estimate[0]  # Quadratic term (matrix).
            self.good_trajs_info[cond].cv = good_cost_estimate[1]  # Linear term (vector).
            self.good_trajs_info[cond].cc = good_cost_estimate[2]  # Constant term (scalar).

            self.bad_trajs_info[cond].Cm = bad_cost_estimate[0]  # Quadratic term (matrix).
            self.bad_trajs_info[cond].cv = bad_cost_estimate[1]  # Linear term (vector).
            self.bad_trajs_info[cond].cc = bad_cost_estimate[2]  # Constant term (scalar).

    def _check_kl_div_good_bad(self):
        for cond in range(self.M):
            good_distr = self.good_duality_info[cond].traj_dist
            bad_distr = self.bad_duality_info[cond].traj_dist
            mu_good, sigma_good = lqr_forward(good_distr,
                                              self.good_trajs_info[cond])
            mu_bad, sigma_bad = lqr_forward(bad_distr,
                                            self.bad_trajs_info[cond])
            kl_div_good_bad = traj_distr_kl_alt(mu_good, sigma_good,
                                                good_distr, bad_distr, tot=True)
            #print("G/B KL_div: %f " % kl_div_good_bad)
            self.logger.info('--->Divergence btw good/bad trajs is: %f'
                             % kl_div_good_bad)

    def _update_good_bad_dynamics(self, option='duality'):
        """
        Instantiate dynamics objects and update prior. Fit dynamics to sample(s).
        """
        for m in range(self.M):
            if option == 'duality':
                good_data = self.good_duality_info[m].sample_list
                bad_data = self.bad_duality_info[m].sample_list
            else:
                good_data = self.cur[m].sample_list
                bad_data = self.cur[m].sample_list

            X_good = good_data.get_states()
            U_good = good_data.get_actions()
            X_bad = bad_data.get_states()
            U_bad = bad_data.get_actions()

            # Update prior and fit dynamics.
            self.good_trajs_info[m].dynamics.update_prior(good_data)
            self.good_trajs_info[m].dynamics.fit(X_good, U_good)
            self.bad_trajs_info[m].dynamics.update_prior(bad_data)
            self.bad_trajs_info[m].dynamics.fit(X_bad, U_bad)

            # Fit x0mu/x0sigma.
            x0_good = X_good[:, 0, :]
            x0mu_good = np.mean(x0_good, axis=0)  # TODO: SAME X0 FOR ALL??
            self.good_trajs_info[m].x0mu = x0mu_good
            self.good_trajs_info[m].x0sigma = np.diag(np.maximum(np.var(x0_good, axis=0),
                                                                 self._hyperparams['initial_state_var']))
            x0_bad = X_bad[:, 0, :]
            x0mu_bad = np.mean(x0_bad, axis=0)  # TODO: SAME X0 FOR ALL??
            self.bad_trajs_info[m].x0mu = x0mu_bad
            self.bad_trajs_info[m].x0sigma = np.diag(np.maximum(np.var(x0_bad, axis=0),
                                                                self._hyperparams['initial_state_var']))

            prior_good = self.good_trajs_info[m].dynamics.get_prior()
            if prior_good:
                mu0, Phi, priorm, n0 = prior_good.initial_state()
                N = len(good_data)
                self.good_trajs_info[m].x0sigma += Phi + (N*priorm) / (N+priorm) * np.outer(x0mu_good-mu0, x0mu_good-mu0) / (N+n0)

            prior_bad = self.good_trajs_info[m].dynamics.get_prior()
            if prior_bad:
                mu0, Phi, priorm, n0 = prior_bad.initial_state()
                N = len(bad_data)
                self.bad_trajs_info[m].x0sigma += Phi + (N*priorm) / (N+priorm) * np.outer(x0mu_bad-mu0, x0mu_bad-mu0) / (N+n0)

    def _update_good_bad_fit(self):
        min_good_var = self._hyperparams['algo_hyperparams']['min_good_var']
        min_bad_var = self._hyperparams['algo_hyperparams']['min_bad_var']

        fit_traj_dist_fcn = fit_linear_gaussian_policy

        for cond in range(self.M):
            self.good_duality_info[cond].traj_dist = \
                fit_traj_dist_fcn(self.good_duality_info[cond].sample_list,
                                  min_good_var)
            self.bad_duality_info[cond].traj_dist = \
                fit_traj_dist_fcn(self.bad_duality_info[cond].sample_list,
                                  min_bad_var)

