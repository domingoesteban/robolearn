import numpy as np
from robolearn.v010.algos.gps.gps_utils import PolicyInfo, extract_condition
from robolearn.v010.algos.gps.gps_utils import TrajectoryInfo, DualityInfo
from robolearn.v010.utils.experience_buffer import ExperienceBuffer
from robolearn.v010.utils.experience_buffer import get_bigger_idx, get_smaller_idx
from robolearn.v010.utils.traj_opt.traj_opt_utils import traj_distr_kl, traj_distr_kl_alt
from robolearn.v010.utils.policy_utils import fit_linear_gaussian_policy, lqr_forward

from robolearn.v010.utils.sample.sample import Sample
from robolearn.v010.utils.sample.sample_list import SampleList


class Dualism(object):
    def __init__(self):
        # Duality data with: [sample_list, samples_cost, cs_traj, traj_dist, pol_info]
        self._bad_duality_info = [DualityInfo() for _ in range(self.M)]
        self._good_duality_info = [DualityInfo() for _ in range(self.M)]

        # Good/Bad Experience Buffer
        for ii in range(self.M):
            buffer_size = self._hyperparams['algo_hyperparams']['n_bad_buffer']
            selection_type = \
                self._hyperparams['algo_hyperparams']['bad_traj_selection_type']
            self._bad_duality_info[ii].experience_buffer = \
                ExperienceBuffer(buffer_size, 'bad', selection_type)

            buffer_size = self._hyperparams['algo_hyperparams']['n_good_buffer']
            selection_type = \
                self._hyperparams['algo_hyperparams']['good_traj_selection_type']
            self._good_duality_info[ii].experience_buffer = \
                ExperienceBuffer(buffer_size, 'good', selection_type)

        # TrajectoryInfo for good and bad trajectories
        self.bad_trajs_info = [None for _ in range(self.M)]
        self.good_trajs_info = [None for _ in range(self.M)]
        for m in range(self.M):
            self.bad_trajs_info[m] = TrajectoryInfo()
            self.good_trajs_info[m] = TrajectoryInfo()

            if self._hyperparams['fit_dynamics']:
                dynamics = self._hyperparams['dynamics']
                self.bad_trajs_info[m].dynamics = dynamics['type'](dynamics)
                self.good_trajs_info[m].dynamics = dynamics['type'](dynamics)

            # TODO: Use demonstration trajectories
            # # Get the initial trajectory distribution hyperparams
            # init_traj_distr = extract_condition(self._hyperparams['init_traj_distr'], self._train_cond_idx[m])
            # Instantiate Trajectory Distribution: init_lqr or init_pd
            # self.good_duality_infor = init_traj_distr['type'](init_traj_distr)
            # self.bad_duality_infor = init_traj_distr['type'](init_traj_distr)

            # Get the initial trajectory distribution hyperparams
            init_traj_distr = extract_condition(self._hyperparams['init_traj_distr'],
                                                self._train_cond_idx[m])
            # Instantiate Trajectory Distribution: init_lqr or init_pd
            self._bad_duality_info[m].traj_dist = init_traj_distr['type'](init_traj_distr)
            self._good_duality_info[m].traj_dist = init_traj_distr['type'](init_traj_distr)

            # Same policy prior than GlobalPol for good/bad
            self._hyperparams['algo_hyperparams']['T'] = self.T
            self._hyperparams['algo_hyperparams']['dU'] = self.dU
            self._hyperparams['algo_hyperparams']['dX'] = self.dX
            policy_prior = self._hyperparams['algo_hyperparams']['policy_prior']
            self._bad_duality_info[m].pol_info = \
                PolicyInfo(self._hyperparams['algo_hyperparams'])
            self._bad_duality_info[m].pol_info.policy_prior = \
                policy_prior['type'](policy_prior)
            self._good_duality_info[m].pol_info = \
                PolicyInfo(self._hyperparams['algo_hyperparams'])
            self._good_duality_info[m].pol_info.policy_prior = \
                policy_prior['type'](policy_prior)

    def _update_bad_samples(self):
        """Update the Bad samples list and samples cost

        Returns: None

        """

        logger = self.logger

        for cond in range(self.M):
            # Sample costs estimate.
            if self._hyperparams['algo_hyperparams']['bad_costs']:
                cs = np.zeros_like(self.cur[cond].cs)
                for bc in self._hyperparams['algo_hyperparams']['bad_costs']:
                    for ss in range(cs.shape[0]):  # Over samples
                        cs[ss, :] += self.cur[cond].cost_compo[ss][bc]
                # If this specific cost is zero, then use the total cost
                if np.sum(cs) == 0:
                    cs = self.cur[cond].cs
            else:
                cs = self.cur[cond].cs

            sample_list = self.cur[cond].sample_list

            # Get index of sample with worst Return
            #worst_index = np.argmax(np.sum(cs, axis=1))
            n_bad = self._hyperparams['algo_hyperparams']['n_bad_samples']
            if n_bad == cs.shape[0]:
                worst_indeces = range(n_bad)
            else:
                worst_indeces = get_bigger_idx(np.sum(cs, axis=1), n_bad)

            # TODO: Maybe it is better to put this step directly in exp_buffer
            samples_to_add = [sample_list[bad_index] for bad_index in worst_indeces]
            costs_to_add = [cs[bad_index] for bad_index in worst_indeces]

            # Get the experience buffer
            exp_buffer = self._bad_duality_info[cond].experience_buffer

            # Add to buffer
            exp_buffer.add(samples_to_add, costs_to_add)

            # TODO: CHeck if it is better to fit to the whole buffer
            # Get the desired number of elements to fit the traj
            trajs, costs = exp_buffer.get_trajs_and_costs(n_bad)

            # TODO: Find a better way than create always SampleList
            self._bad_duality_info[cond].sample_list = SampleList(trajs)
            self._bad_duality_info[cond].samples_cost = costs

            # print(sorted(np.sum(cs, axis=1)))
            # print(np.sum(costs_to_add, axis=1))
            # print(np.sum(SampleList(trajs).get_states()))
            # print(np.sum(costs, axis=1))
            # print('buffer:', np.sum([cc for cc in exp_buffer._costs], axis=1))
            # print('%%')

    def _update_good_samples(self):
        """Update the Good samples list and samples cost

        Returns: None

        """
        logger = self.logger

        for cond in range(self.M):
            # Sample costs estimate.
            if self._hyperparams['algo_hyperparams']['bad_costs']:
                cs = np.zeros_like(self.cur[cond].cs)
                for bc in self._hyperparams['algo_hyperparams']['bad_costs']:
                    for ss in range(cs.shape[0]):  # Over samples
                        cs[ss, :] += self.cur[cond].cost_compo[ss][bc]
                # If this specific cost is zero, then use the total cost
                if np.sum(cs) == 0:
                    cs = self.cur[cond].cs
            else:
                cs = self.cur[cond].cs

            sample_list = self.cur[cond].sample_list

            # Get index of sample with best Return
            n_good = self._hyperparams['algo_hyperparams']['n_good_samples']
            if n_good == cs.shape[0]:
                best_indeces = range(n_good)
            else:
                best_indeces = get_smaller_idx(np.sum(cs, axis=1), n_good)

            # TODO: Maybe it is better to put this step directly in exp_buffer
            samples_to_add = [sample_list[good_index] for good_index in best_indeces]
            costs_to_add = [cs[good_index] for good_index in best_indeces]

            # Get the experience buffer
            exp_buffer = self._good_duality_info[cond].experience_buffer

            # Add to buffer
            exp_buffer.add(samples_to_add, costs_to_add)

            # TODO: CHeck if it is better to fit to the whole buffer
            # Get the desired number of elements to fit the traj
            trajs, costs = exp_buffer.get_trajs_and_costs(n_good)

            # TODO: Find a better way than create always SampleList
            self._good_duality_info[cond].sample_list = SampleList(trajs)
            self._good_duality_info[cond].samples_cost = costs

    def _eval_good_bad_samples_costs(self):
        """
        Evaluate costs for all current samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        for cond in range(self.M):
            cost_fcn = self.cost_function[cond]
            good_sample_list = self._good_duality_info[cond].sample_list
            bad_sample_list = self._bad_duality_info[cond].sample_list

            good_true_cost, good_cost_estimate, _ = \
                self._eval_sample_list_cost(good_sample_list, cost_fcn)
            bad_true_cost, bad_cost_estimate, _ = \
                self._eval_sample_list_cost(bad_sample_list, cost_fcn)

            # True value of cost (cs average).
            self._good_duality_info[cond].traj_cost = np.mean(good_true_cost,
                                                              axis=0)
            self._bad_duality_info[cond].traj_cost = np.mean(bad_true_cost,
                                                             axis=0)

            # Reward estimate.
            self.good_trajs_info[cond].Cm = good_cost_estimate[0]  # Quadratic term (matrix).
            self.good_trajs_info[cond].cv = good_cost_estimate[1]  # Linear term (vector).
            self.good_trajs_info[cond].cc = good_cost_estimate[2]  # Constant term (scalar).

            self.bad_trajs_info[cond].Cm = bad_cost_estimate[0]  # Quadratic term (matrix).
            self.bad_trajs_info[cond].cv = bad_cost_estimate[1]  # Linear term (vector).
            self.bad_trajs_info[cond].cc = bad_cost_estimate[2]  # Constant term (scalar).

    def _check_kl_div_good_bad(self):
        for cond in range(self.M):
            good_distr = self._good_duality_info[cond].traj_dist
            bad_distr = self._bad_duality_info[cond].traj_dist
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
                good_data = self._good_duality_info[m].sample_list
                bad_data = self._bad_duality_info[m].sample_list
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
            self._bad_duality_info[cond].traj_dist = \
                fit_traj_dist_fcn(self._bad_duality_info[cond].sample_list,
                                  min_bad_var)
            self._good_duality_info[cond].traj_dist = \
                fit_traj_dist_fcn(self._good_duality_info[cond].sample_list,
                                  min_good_var)
            self.cur[cond].good_traj_distr = self._good_duality_info[cond].traj_dist
            self.cur[cond].bad_traj_distr = self._bad_duality_info[cond].traj_dist

    def _update_good_bad_size(self):
        for m in range(self.M):
            if self.iteration_count >= 1 and self.prev[m].sample_list:
                # Good
                # self.cur[m].good_step_mult = 2*self.cur[m].step_mult
                good_mult = \
                    self._hyperparams['algo_hyperparams']['good_fix_rel_multi']*self.cur[m].step_mult
                new_good = max(
                    min(good_mult,
                        self._hyperparams['algo_hyperparams']['max_good_mult']),
                    self._hyperparams['algo_hyperparams']['min_good_mult']
                )
                self.cur[m].good_step_mult = new_good

                #
                traj_info = self.cur[m].traj_info
                current_distr = self.cur[m].traj_distr
                bad_distr = self.cur[m].bad_traj_distr
                prev_traj_info = self.prev[m].traj_info
                prev_distr = self.prev[m].traj_distr
                prev_bad_distr = self.prev[m].bad_traj_distr

                actual_laplace = \
                    self.traj_opt.estimate_cost(current_distr, traj_info)
                self.logger.info('actual_laplace: %r' % actual_laplace.sum())

                prev_laplace = \
                    self.traj_opt.estimate_cost(prev_distr, traj_info)
                self.logger.info('prev_laplace: %r' % prev_laplace.sum())

                bad_laplace = \
                    self.traj_opt.estimate_cost(bad_distr, traj_info)

                self.logger.info('actual_bad: %r' % bad_laplace.sum())

                prev_bad_laplace = \
                    self.traj_opt.estimate_cost(prev_bad_distr, traj_info)

                self.logger.info('prev_bad: %r' % prev_bad_laplace.sum())


                # THIS WAS BEFORE 08/02
                rel_difference = (1 + (bad_laplace.sum() - actual_laplace.sum())/actual_laplace.sum())
                self.logger.info('Actual/Bad REL difference %r' % rel_difference)

                mu_to_check, sigma_to_check = \
                    self.traj_opt.forward(current_distr, traj_info)

                kl_div_bad = traj_distr_kl_alt(mu_to_check, sigma_to_check,
                                               current_distr, bad_distr,
                                               tot=True)
                print('Current bad_div:', kl_div_bad)

                prev_mu_to_check, prev_sigma_to_check = \
                    self.traj_opt.forward(prev_distr, traj_info)

                prev_kl_div_bad = \
                    traj_distr_kl_alt(prev_mu_to_check, prev_sigma_to_check,
                                      prev_distr, prev_bad_distr, tot=True)

                rel_kl = max(0,
                             1 + (prev_kl_div_bad - kl_div_bad)/prev_kl_div_bad)

                print('#$'*30)
                print('MULTIPLY REL_DIFFERENCE EEEEEEE!!!!!!')
                min_rel_diff = self._hyperparams['algo_hyperparams']['min_bad_rel_diff']
                max_rel_diff = self._hyperparams['algo_hyperparams']['max_bad_rel_diff']
                mult_rel_diff = self._hyperparams['algo_hyperparams']['mult_bad_rel_diff']

                rel_difference = min(max(rel_difference, min_rel_diff),
                                        max_rel_diff)
                rel_difference = mult_rel_diff*rel_difference

                self.logger.info('ACTUAL/BAD MULT %r, %r, %r'
                                 % (min_rel_diff, max_rel_diff, mult_rel_diff))
                self.logger.info('Actual/Bad difference %r' % rel_difference)

                # bad_mult = rel_difference*self.cur[m].step_mult

                print('BAD REL_DIFFERENCE IS WRT PREV_BAD_MULT!!!!!!')
                bad_mult = rel_difference*1

                new_bad = max(
                    min(bad_mult,
                        self._hyperparams['algo_hyperparams']['max_bad_mult']),
                    self._hyperparams['algo_hyperparams']['min_bad_mult']
                )
                self.cur[m].bad_step_mult = new_bad


                # if not hasattr(self, 'bad_discount'):
                #     self.bad_discount = self.kl_bad/self.max_iterations
                # self.cur[m].bad_step_mult = new_bad

    def _take_dualist_sample(self, bad_or_good, itr, train_or_test='train'):
        """
        Collect a sample from the environment.
        :param traj_or_pol: Use trajectory distributions or current policy.
                            'traj' or 'pol'
        :param itr: Current TrajOpt iteration
        :return:
        """
        # If 'pol' sampling, do it with zero noise
        zero_noise = np.zeros((self.T, self.dU))

        self.logger.info("Sampling with dualism:'%s'" % bad_or_good)

        if train_or_test == 'train':
            conditions = self._train_cond_idx
        elif train_or_test == 'test':
            conditions = self._test_cond_idx
        else:
            raise ValueError("Wrong train_or_test option %s" % train_or_test)

        on_policy = False
        total_samples = 1
        save = False  # Add sample to agent sample list TODO: CHECK THIS

        # A list of SampleList for each condition
        sample_lists = list()

        for cc, cond in enumerate(conditions):
            samples = list()

            # On-policy or Off-policy
            if on_policy and (self.iteration_count > 0 or
                              ('sample_pol_first_itr' in self._hyperparams
                               and self._hyperparams['sample_pol_first_itr'])):
                policy = None
                self.logger.info("On-policy sampling: %s!"
                                 % type(self.agent.policy).__name__)
            else:
                policy = self.cur[cond].traj_distr
                self.logger.info("Off-policy sampling: %s!"
                                 % type(policy).__name__)
            if bad_or_good == 'bad':
                policy = self.cur[cond].bad_traj_distr
            elif bad_or_good == 'good':
                policy = self.cur[cond].good_traj_distr
            else:
                raise ValueError("Wrong bad_or_good option %s" % bad_or_good)
            self.logger.info("Off-policy sampling with dualism: %s (%s)!"
                             % (bad_or_good, type(policy).__name__))

            for i in range(total_samples):
                noise = zero_noise

                self.env.reset(condition=cond)
                sample_text = "'%s' sampling | itr:%d/%d, cond:%d/%d, s:%d/%d" \
                              % (bad_or_good, itr+1, self.max_iterations,
                                 cond+1, len(conditions),
                                 i+1, total_samples)
                self.logger.info(sample_text)
                sample = self.agent.sample(self.env, cond, self.T,
                                           self.dt, noise, policy=policy,
                                           save=save,
                                           real_time=self._hyperparams['sample_real_time'])
                samples.append(sample)

            sample_lists.append(SampleList(samples))

        return sample_lists
