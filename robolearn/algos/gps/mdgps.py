import os
import numpy as np
import scipy as sp
import datetime
import copy
from robolearn.algos.algorithm import Algorithm

from robolearn.utils.sample.sample import Sample
from robolearn.utils.sample.sample_list import SampleList

from robolearn.algos.gps.gps_config import DEFAULT_MDGPS_HYPERPARAMS
from robolearn.algos.gps.gps_utils import PolicyInfo
from robolearn.algos.gps.gps_utils import IterationData, extract_condition
from robolearn.algos.gps.gps_utils import TrajectoryInfo
from robolearn.agents.agent_utils import generate_noise
from robolearn.utils.data_logger import DataLogger


class MDGPS(Algorithm):
    def __init__(self, agent, env, **kwargs):
        super(MDGPS, self).__init__(DEFAULT_MDGPS_HYPERPARAMS, kwargs)
        self.agent = agent
        self.env = env

        # Get dimensions from the environment
        self.dU = self._hyperparams['dU'] = env.get_action_dim()
        self.dX = self._hyperparams['dX'] = env.get_state_dim()
        self.dO = self._hyperparams['dO'] = env.get_obs_dim()

        # Get time values from 'hyperparams'
        self.T = self._hyperparams['T']
        self.dt = self._hyperparams['dt']

        # Get/Define train and test conditions idxs
        if 'train_conditions' in self._hyperparams \
                and self._hyperparams['train_conditions'] is not None:
            self._train_cond_idx = self._hyperparams['train_conditions']
            self._test_cond_idx = self._hyperparams['test_conditions']
        else:
            self._train_cond_idx = self._test_cond_idx = list(range(self.M))
            self._hyperparams['train_conditions'] = self._train_cond_idx
            self._hyperparams['test_conditions'] = self._test_cond_idx

        # Number of initial conditions
        self.M = len(self._train_cond_idx)

        # Log and Data files
        if 'data_files_dir' in self._hyperparams:
            self._data_files_dir = self._hyperparams['data_files_dir']
        else:
            self._data_files_dir = 'GPS_' + \
                                   str(datetime.datetime.now().
                                       strftime("%Y-%m-%d_%H:%M:%S"))
        self.data_logger = DataLogger(self._data_files_dir)
        self.logger = self._setup_logger('log_dualgps', self._data_files_dir,
                                         '/log_dualgps.log', also_screen=True)

        # Get max number of iterations and define counter
        self.max_iterations = self._hyperparams['iterations']
        self.iteration_count = 0

        # Noise to be used with trajectory distributions
        self.noise_data = np.zeros((self.max_iterations, self.M,
                                    self._hyperparams['num_samples'],
                                    self.T, self.dU))

        if 'seed' in self._hyperparams:
            seed = self._hyperparams['seed']
        else:
            seed = 0
        np.random.seed(seed)

        if self._hyperparams['noisy_samples']:
            for ii in range(self.max_iterations):
                for cond in range(self.M):
                    for n in range(self._hyperparams['num_samples']):
                        self.noise_data[ii, cond, n, :, :] = \
                            generate_noise(self.T, self.dU, self._hyperparams)

        # IterationData objects for each condition.
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]

        # Initial trajectory hyperparams
        init_traj_distr = self._hyperparams['init_traj_distr']
        init_traj_distr['x0'] = env.get_conditions()
        init_traj_distr['dX'] = self.dX
        init_traj_distr['dU'] = self.dU
        init_traj_distr['dt'] = self.dt
        init_traj_distr['T'] = self.T

        # Add same dynamics for all the condition if the algorithm requires it
        if self._hyperparams['fit_dynamics']:
            dynamics = self._hyperparams['dynamics']

        # Trajectory Info
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()

            if self._hyperparams['fit_dynamics']:
                self.cur[m].traj_info.dynamics = dynamics['type'](dynamics)

            # Get the initial trajectory distribution hyperparams
            init_traj_distr = extract_condition(
                self._hyperparams['init_traj_distr'], self._train_cond_idx[m])

            # Instantiate Trajectory Distribution: init_lqr or init_pd
            self.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr)

        # Last trajectory distribution optimized in C-step
        self.new_traj_distr = None

        # Cost function #
        # ------------- #
        if self._hyperparams['cost'] is None:
            raise AttributeError("Cost function has not been defined")
        total_conditions = self._train_cond_idx + self._test_cond_idx
        if isinstance(type(self._hyperparams['cost']), list):
            # One cost function type for each condition
            self.cost_function = \
                [self._hyperparams['cost'][i]['type'](self._hyperparams['cost'][i])
                 for i in total_conditions]
        else:
            # Same cost function type for all conditions
            self.cost_function = \
                [self._hyperparams['cost']['type'](self._hyperparams['cost'])
                 for _ in total_conditions]

        # KL base values #
        # -------------- #
        self.base_kl_step = self._hyperparams['algo_hyperparams']['kl_step']
        # Set initial dual variables
        for m in range(self.M):
            self.cur[m].eta = self._hyperparams['algo_hyperparams']['init_eta']

        # Traj Opt (Local policy opt) method #
        # ---------------------------------- #
        # Options: LQR, PI2, DualistTrajOpt
        self.traj_opt = \
            self._hyperparams['traj_opt']['type'](self._hyperparams['traj_opt'])
        self.traj_opt.set_logger(self.logger)

        # Global Policy #
        # ------------- #
        self.policy_opt = self.agent.policy_opt
        self._policy_samples = [None for _ in self._test_cond_idx]

        # MDGPS data #
        # ---------- #
        self._hyperparams['algo_hyperparams']['T'] = self.T
        self._hyperparams['algo_hyperparams']['dU'] = self.dU
        self._hyperparams['algo_hyperparams']['dX'] = self.dX
        policy_prior = self._hyperparams['algo_hyperparams']['policy_prior']
        for m in range(self.M):
            # Same policy prior type for all conditions
            self.cur[m].pol_info = PolicyInfo(self._hyperparams['algo_hyperparams'])
            self.cur[m].pol_info.policy_prior = policy_prior['type'](policy_prior)

    def _iteration(self, itr):
        logger = self.logger

        # Sample from environment using current trajectory distributions
        logger.info('')
        logger.info('%s: itr:%02d | Sampling from local trajectories...'
                    % (type(self).__name__, itr+1))
        traj_or_pol = 'traj'
        self._take_sample(traj_or_pol, itr, 'train')

        # Get last samples from agent
        n_samples = self._hyperparams['num_samples']
        traj_sample_lists = [self.agent.get_samples(cond, -n_samples)
                             for cond in self._train_cond_idx]

        # TODO: Check if it is better to 'remember' these samples
        # Clear agent sample
        self.agent.clear_samples()

        # Update dynamics model using all samples.
        logger.info('')
        logger.info('MDGPS: itr:%02d | '
                    'Updating dynamics linearization...' % (itr+1))
        self._update_dynamic_model()

        logger.info('')
        logger.info('MDGPS: itr:%02d | '
                    'Evaluating samples costs...' % (itr+1))
        for m in range(self.M):
            self._eval_iter_samples_cost(m)

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.iteration_count == 0:
            self.new_traj_distr = [self.cur[cond].traj_distr
                                   for cond in range(self.M)]
            logger.info("\n"*2)
            logger.info('%s: itr:%02d | '
                        'S-step for init_traj_distribution (iter=0)...'
                        % (type(self).__name__, itr+1))
            self._update_policy()

        # Update global policy linearizations.
        logger.info('')
        logger.info('%s: itr:%02d | '
                    'Updating global policy linearization...'
                    % (type(self).__name__, itr+1))
        for m in range(self.M):
            self._update_policy_fit(m)

        # Update KL step
        logger.info('')
        if self.iteration_count > 0:
            logger.info('%s: itr:%02d | '
                        'Updating KL step size with GLOBAL policy...'
                        % (type(self).__name__, itr+1))
            self._update_step_size()

        # C-step
        logger.info('')
        logger.info('%s: itr:%02d | '
                    'Updating trajectories...'
                    % (type(self).__name__, itr+1))
        for ii in range(self._hyperparams['algo_hyperparams']
                        ['inner_iterations']):
            logger.info('-%s: itr:%02d | Inner iteration %d/%d'
                        % (type(self).__name__, itr+1, ii+1,
                           self._hyperparams['algo_hyperparams']
                           ['inner_iterations']))
            self._update_trajectories()

        # S-step
        logger.info('')
        logger.info('%s:itr:%02d | ->| S-step |<-'
                    % (type(self).__name__, itr+1))
        self._update_policy()

        # Test policy after iteration
        if self._hyperparams['test_after_iter']:
            logger.info('')
            logger.info('%s: itr:%02d | '
                        'Testing global policy...'
                        % (type(self).__name__, itr+1))
            traj_or_pol = 'pol'
            self._take_sample(traj_or_pol, itr, 'test')

            #
            # pol_sample_lists = list()
            # for m in range(self.M):
            #     pol_sample_lists.append(self.cur[m].pol_info.policy_samples)

            pol_sample_lists_costs = [None for _ in self._test_cond_idx]
            pol_sample_lists_cost_compositions = [None for _ in self._test_cond_idx]
            for cc, cond in enumerate(self._test_cond_idx):
                sample_list = self._policy_samples[cc]
                cost_fcn = self.cost_function[cond]
                costs = self._eval_sample_list_cost(sample_list, cost_fcn)
                pol_sample_lists_costs[cc] = costs[0]
                pol_sample_lists_cost_compositions[cc] = costs[2]

            for m, cond in enumerate(self._test_cond_idx):
                print('&'*10)
                print('Average Cost')
                print('Condition:%02d' % cond)
                print('Avg cost: %f'
                      % pol_sample_lists_costs[m].sum(axis=1).mean())
                print('&'*10)

        else:
            pol_sample_lists_costs = None
            pol_sample_lists_cost_compositions = None


        # Log data
        self._log_iter_data(itr, traj_sample_lists, self._policy_samples,
                            pol_sample_lists_costs,
                            pol_sample_lists_cost_compositions)

        # Prepare everything for next iteration
        self._advance_iteration_variables()

    def compute_traj_cost(self, cond, eta, augment=True):
        """
        Compute cost estimates used in the LQR backward pass.

        :param cond: Number of condition
        :param eta: Dual variable corresponding to KL divergence with
                    previous policy.
        :param augment: True if we want a KL constraint for all time-steps.
                        False otherwise. True for MDGPS
        :return: Cm and cv
        """
        traj_info = self.cur[cond].traj_info
        traj_distr = self.cur[cond].traj_distr  # We do not use it

        if not augment:  # Whether to augment cost with term to penalize KL
            return traj_info.Cm, traj_info.cv

        T = self.T
        dX = self.dX
        dU = self.dU

        Cm, cv = np.copy(traj_info.Cm), np.copy(traj_info.cv)

        # Pol_info
        pol_info = self.cur[cond].pol_info

        # Weight of maximum entropy term in trajectory optimization
        multiplier = self._hyperparams['max_ent_traj']

        # Surrogate cost
        PKLm = np.zeros((T, dX+dU, dX+dU))
        PKLv = np.zeros((T, dX+dU))

        self.logger.warning('WARN: adding a beta to divisor in '
                            'compute_traj_cost')
        divisor = (eta + multiplier)
        fCm = Cm / divisor
        fcv = cv / divisor

        # Add in the KL divergence with previous policy.
        for t in range(self.T):
            # Policy KL-divergence terms.
            inv_pol_S = np.linalg.solve(
                pol_info.chol_pol_S[t, :, :],
                np.linalg.solve(pol_info.chol_pol_S[t, :, :].T, np.eye(dU))
            )
            KB = pol_info.pol_K[t, :, :]
            kB = pol_info.pol_k[t, :]

            PKLm[t, :, :] = np.vstack([
                np.hstack([KB.T.dot(inv_pol_S).dot(KB), -KB.T.dot(inv_pol_S)]),
                np.hstack([-inv_pol_S.dot(KB), inv_pol_S])
            ])
            PKLv[t, :] = np.concatenate([
                KB.T.dot(inv_pol_S).dot(kB), -inv_pol_S.dot(kB)
            ])
            fCm[t, :, :] += PKLm[t, :, :] * eta / divisor
            fcv[t, :] += PKLv[t, :] * eta / divisor

        return fCm, fcv

    def _take_sample(self, traj_or_pol, itr, train_or_test='train'):
        """
        Collect a sample from the environment.
        :param traj_or_pol: Use trajectory distributions or current policy.
                            'traj' or 'pol'
        :param itr: Current TrajOpt iteration
        :return:
        """
        # If 'pol' sampling, do it with zero noise
        zero_noise = np.zeros((self.T, self.dU))

        self.logger.info("Sampling with mode:'%s'" % traj_or_pol)

        if train_or_test == 'train':
            conditions = self._train_cond_idx
        elif train_or_test == 'test':
            conditions = self._test_cond_idx
        else:
            raise ValueError("Wrong train_or_test option %s" % train_or_test)

        if traj_or_pol == 'traj':
            on_policy = self._hyperparams['sample_on_policy']
            total_samples = self._hyperparams['num_samples']
            save = True  # Add sample to agent sample list

        elif traj_or_pol == 'pol':
            on_policy = True
            total_samples = self._hyperparams['test_samples']
            save = False  # Add sample to agent sample list TODO: CHECK THIS
        else:
            raise ValueError("Wrong traj_or_pol option %s" % traj_or_pol)

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

            for i in range(total_samples):
                if traj_or_pol == 'traj':
                    noise = self.noise_data[itr, cond, i, :, :]
                else:
                    noise = zero_noise

                self.env.reset(condition=cond)
                sample_text = "'%s' sampling | itr:%d/%d, cond:%d/%d, s:%d/%d" \
                              % (traj_or_pol, itr+1, self.max_iterations,
                                 cond+1, len(conditions),
                                 i+1, total_samples)
                self.logger.info(sample_text)
                sample = self.agent.sample(self.env, cond, self.T,
                                           self.dt, noise, policy=policy,
                                           save=save)
                samples.append(sample)

            sample_lists.append(SampleList(samples))

            # Save them also in pol_info/policy_samples
            if on_policy:
                if train_or_test == 'train':
                    self.cur[cond].pol_info.policy_samples = sample_lists[cond]
                else:
                    self._policy_samples[cc] = sample_lists[cc]

        return sample_lists

    def _eval_iter_samples_cost(self, cond):
        """
        Evaluate costs for all current samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        sample_list = self.cur[cond].sample_list
        cost_fcn = self.cost_function[cond]

        true_cost, cost_estimate, cost_compo = \
            self._eval_sample_list_cost(sample_list, cost_fcn)
        self.cur[cond].cs = true_cost  # True value of cost.
        self.cur[cond].cost_compo = cost_compo  # Cost 'composition'.

        # Cost estimate.
        self.cur[cond].traj_info.Cm = cost_estimate[0]  # Quadratic term (matrix).
        self.cur[cond].traj_info.cv = cost_estimate[1]  # Linear term (vector).
        self.cur[cond].traj_info.cc = cost_estimate[2]  # Constant term (scalar).

    def _eval_sample_list_cost(self, sample_list, cost_fcn):
        """
        Evaluate costs for a sample_list using a specific cost function.
        Args:
            cost: self.cost_function[cond]
            cond: Condition to evaluate cost on.
        """
        # Constants.
        T, dX, dU = self.T, self.dX, self.dU
        N = len(sample_list)

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX+dU))
        Cm = np.zeros((N, T, dX+dU, dX+dU))
        cost_composition = [None for _ in range(N)]
        for n in range(N):
            sample = sample_list[n]
            # Get costs.
            l, lx, lu, lxx, luu, lux, cost_composition[n] = cost_fcn.eval(sample)
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
            cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) \
                        + 0.5 * np.sum(rdiff * cv_update, axis=1)
            cv[n, :, :] += cv_update

        # Fill in cost estimate.
        cc = np.mean(cc, 0)  # Constant term (scalar).
        cv = np.mean(cv, 0)  # Linear term (vector).
        Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        return cs, (Cm, cv, cc), cost_composition

    def _update_dynamic_model(self):
        """
        Instantiate dynamics objects and update prior.
        Fit dynamics to current samples.
        """
        for m in range(self.M):
            cur_data = self.cur[m].sample_list
            X = cur_data.get_states()
            U = cur_data.get_actions()

            # Update prior and fit dynamics.
            self.cur[m].traj_info.dynamics.update_prior(cur_data)
            self.cur[m].traj_info.dynamics.fit(X, U)

            # Fit x0mu/x0sigma.
            x0 = X[:, 0, :]
            x0mu = np.mean(x0, axis=0)
            self.cur[m].traj_info.x0mu = x0mu
            self.cur[m].traj_info.x0sigma = \
                np.diag(np.maximum(np.var(x0, axis=0),
                                   self._hyperparams['initial_state_var']))

            prior = self.cur[m].traj_info.dynamics.get_prior()
            if prior:
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = len(cur_data)
                self.cur[m].traj_info.x0sigma += \
                    Phi + (N*priorm) / (N+priorm) * \
                    np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)

    def _update_policy(self):
        """
        Computes(updates) a new global policy.
        :return:
        """
        LOGGER = self.logger

        LOGGER.info('-->Updating Global policy...')
        dU, dO, T = self.dU, self.dO, self.T
        # Compute target mean, cov(precision), and weight for each sample; and concatenate them.
        obs_data, tgt_mu = np.zeros((0, T, dO)), np.zeros((0, T, dU))
        tgt_prc, tgt_wt = np.zeros((0, T, dU, dU)), np.zeros((0, T))
        for m in range(self.M):
            samples = self.cur[m].sample_list
            X = samples.get_states()
            N = len(samples)
            traj = self.new_traj_distr[m]
            pol_info = self.cur[m].pol_info
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

        logger = self.logger

        self.policy_opt.update(obs_data, tgt_mu, tgt_prc, tgt_wt, LOGGER=logger)

    def _update_policy_fit(self, cond):
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
        mode = self._hyperparams['algo_hyperparams']['policy_sample_mode']
        policy_prior.update(samples, self.policy_opt, mode)

        # Fit linearization and store in pol_info.
        pol_info.pol_K, pol_info.pol_k, pol_info.pol_S = \
            policy_prior.fit(X, pol_mu, pol_sig)
        for t in range(T):
            pol_info.chol_pol_S[t, :, :] = \
                sp.linalg.cholesky(pol_info.pol_S[t, :, :])

    def _update_step_size(self):
        """
        Calculate new step sizes. This version uses the same step size for all
        conditions.
        """
        LOGGER = self.logger

        estimate_cost_fcn = self.traj_opt.estimate_cost

        # Compute previous cost and previous expected cost.
        prev_M = len(self.prev)  # May be different in future.
        prev_laplace = np.empty(prev_M)
        prev_mc = np.empty(prev_M)
        prev_predicted = np.empty(prev_M)
        for m in range(prev_M):
            prev_nn = self.prev[m].pol_info.traj_distr()
            prev_lg = self.prev[m].new_traj_distr

            # Compute values under Laplace approximation. This is the policy
            # that the previous samples were actually drawn from under the
            # dynamics that were estimated from the previous samples.
            prev_laplace[m] = estimate_cost_fcn(prev_nn,
                                                self.prev[m].traj_info).sum()
            # This is the actual cost that we experienced.
            prev_mc[m] = self.prev[m].cs.mean(axis=0).sum()
            # This is the policy that we just used under the dynamics that
            # were estimated from the prev samples (so this is the cost
            # we thought we would have).
            prev_predicted[m] = estimate_cost_fcn(prev_lg,
                                                  self.prev[m].traj_info).sum()

        # Compute current cost.
        cur_laplace = np.empty(self.M)
        cur_mc = np.empty(self.M)
        for m in range(self.M):
            cur_nn = self.cur[m].pol_info.traj_distr()
            # This is the actual cost we have under the current trajectory
            # based on the latest samples.
            cur_laplace[m] = estimate_cost_fcn(cur_nn,
                                               self.cur[m].traj_info).sum()
            cur_mc[m] = self.cur[m].cs.mean(axis=0).sum()

        # Compute predicted and actual improvement.
        prev_laplace = prev_laplace.mean()
        prev_mc = prev_mc.mean()
        prev_predicted = prev_predicted.mean()
        cur_laplace = cur_laplace.mean()
        cur_mc = cur_mc.mean()
        if self._hyperparams['algo_hyperparams']['step_rule'] == 'laplace':
            predicted_impr = prev_laplace - prev_predicted
            actual_impr = prev_laplace - cur_laplace
        elif self._hyperparams['algo_hyperparams']['step_rule'] == 'mc':
            predicted_impr = prev_mc - prev_predicted
            actual_impr = prev_mc - cur_mc
        LOGGER.debug('%s: Previous cost: Laplace: %f, MC: %f',
                     type(self).__name__,
                     prev_laplace, prev_mc)
        LOGGER.debug('%s: Predicted cost: Laplace: %f', type(self).__name__,
                     prev_predicted)
        LOGGER.debug('%s: Actual cost: Laplace: %f, MC: %f',
                     type(self).__name__, cur_laplace, cur_mc)

        for m in range(self.M):
            self._set_new_mult(predicted_impr, actual_impr, m)

    def _set_new_mult(self, predicted_impr, actual_impr, m):
        """
        Adjust step size multiplier according to the predicted versus
        actual improvement.
        """
        # Model improvement as I = predicted_dI * KL + penalty * KL^2,
        # where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2).
        # Optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL =>
        # KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL.
        # Therefore, the new multiplier is given by pred/2*(pred-act).
        new_mult = predicted_impr / (2.0 * max(1e-4,
                                               predicted_impr - actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(
            min(new_mult * self.cur[m].step_mult,
                self._hyperparams['algo_hyperparams']['max_step_mult']),
            self._hyperparams['algo_hyperparams']['min_step_mult']
        )
        self.cur[m].step_mult = new_step

        if new_mult > 1:
            self.logger.info('%s: Increasing step size multiplier to %f',
                             type(self).__name__, new_step)
        else:
            self.logger.info('%s: Decreasing step size multiplier to %f',
                             type(self).__name__, new_step)

    def _update_trajectories(self):
        """
        Compute new linear Gaussian controllers using the TrajOpt algorithm.
        """
        self.logger.info('-->%s: Updating trajectories (local policies)...',
                         type(self).__name__)
        if self.new_traj_distr is None:
            self.new_traj_distr = [self.cur[cond].traj_distr
                                   for cond in range(self.M)]
        for cond in range(self.M):
            traj_opt_outputs = self.traj_opt.update(cond, self)
            self.new_traj_distr[cond] = traj_opt_outputs[0]
            self.cur[cond].eta = traj_opt_outputs[1]

    def _measure_ent(self, cond):
        """
        Measure the entropy of the current trajectory.
        :param cond: Condition
        :return: Entropy
        """
        ent = 0
        for t in range(self.T):
            ent = ent + np.sum(
                np.log(np.diag(self.cur[cond].traj_distr.chol_pol_covar[t, :, :]))
            )
        return ent

    def _take_fake_sample(self, noisy=True):
        """
        Create a fake sample.
        :param noisy:
        :return:
        """
        print('Fake sample!!!')

        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        # Create a sample class
        sample = Sample(self.env, self.T)

        all_obs = np.random.randn(self.T, self.dO)
        all_states = np.random.randn(self.T, self.dX)
        all_actions = np.random.randn(self.T, self.dU)

        sample.set_acts(all_actions)
        sample.set_obs(all_obs)
        sample.set_states(all_states)
        sample.set_noise(noise)

        return sample

    def _restore_algo_state(self, itr_load):
        """
        Initialize from the specified iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns:
            itr_start: Iteration to start from.
        """
        print('Loading previous GPS from iteration %d!' % itr_load)
        itr_load -= 1
        algorithm_file = '%s_algorithm_itr_%02d.pkl' % (type(self).__name__,
                                                        itr_load)
        prev_algorithm = self.data_logger.unpickle(algorithm_file)
        if prev_algorithm is None:
            print("Error: cannot find '%s.'" % algorithm_file)
            os._exit(1)
        else:
            self.__dict__.update(prev_algorithm.__dict__)

        print('Loading agent_itr...')
        agent_file = 'agent_itr_%02d.pkl' % itr_load
        prev_agent = self.data_logger.unpickle(agent_file)
        if prev_agent is None:
            print("Error: cannot find '%s.'" % agent_file)
            os._exit(1)
        else:
            self.agent.__dict__.update(prev_agent.__dict__)

            print('Loading policy_opt_itr...')
            traj_opt_file = 'policy_opt_itr_%02d.pkl' % itr_load
            prev_policy_opt = self.data_logger.unpickle(traj_opt_file)
            if prev_policy_opt is None:
                print("Error: cannot find '%s.'" % traj_opt_file)
                os._exit(1)
            else:
                self.agent.policy_opt.__dict__.update(prev_policy_opt.__dict__)
            self.agent.policy = self.agent.policy_opt.policy

        if type(self).__name__ == 'DualGPS':
            self.load_duality_vars(itr_load)

        # self.algorithm = self.data_logger.unpickle(algorithm_file)
        # if self.algorithm is None:
        #     print("Error: cannot find '%s.'" % algorithm_file)
        #     os._exit(1) # called instead of sys.exit(), since this is in a thread

        # if self.gui:
        #     traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
        #                                                   ('traj_sample_itr_%02d.pkl' % itr_load))
        #     if self.algorithm.cur[0].pol_info:
        #         pol_sample_lists = self.data_logger.unpickle(self._data_files_dir +
        #                                                      ('pol_sample_itr_%02d.pkl' % itr_load))
        #     else:
        #         pol_sample_lists = None
        #     self.gui.set_status_text(
        #         ('Resuming training from algorithm state at iteration %d.\n' +
        #          'Press \'go\' to begin.') % itr_load)
        return itr_load + 1

    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', and advance iteration counter of
        the algorithm.
        :return: None
        """
        self.iteration_count += 1
        self.prev = copy.deepcopy(self.cur)
        # TODO: change IterationData to reflect new stuff better
        for m in range(self.M):
            self.prev[m].new_traj_distr = self.new_traj_distr[m]

        # NEW IterationData object, and remove new_traj_distr
        self.cur = [IterationData() for _ in range(self.M)]
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].traj_info.dynamics = \
                copy.deepcopy(self.prev[m].traj_info.dynamics)
            self.cur[m].step_mult = self.prev[m].step_mult
            self.cur[m].eta = self.prev[m].eta
            self.cur[m].traj_distr = self.new_traj_distr[m]
            self.cur[m].traj_info.last_kl_step = \
                self.prev[m].traj_info.last_kl_step
            # MDGPS
            self.cur[m].pol_info = copy.deepcopy(self.prev[m].pol_info)
        self.new_traj_distr = None

    def _log_iter_data(self, itr, traj_sample_lists, pol_sample_lists=None,
                       pol_sample_lists_costs=None,
                       pol_sample_lists_cost_compositions=None):
        """
        Log data and algorithm.
        :param itr: Iteration number.
        :param traj_sample_lists: trajectory (local policies) samples as
                                  SampleList object.
        :param pol_sample_lists: global policy samples as SampleList object.
        :return: None
        """
        LOGGER = self.logger

        dir_path = self.data_logger.dir_path + ('/itr_%02d' % itr)

        # LOGGER.info("Logging Agent... ")
        # self.data_logger.pickle(
        #     ('dualgps_itr_%02d.pkl' % itr),
        #     # copy.copy(temp_dict)
        #     copy.copy(self.agent)
        # )
        LOGGER.info("Logging Policy_Opt... ")
        self.data_logger.pickle(
            ('policy_opt_itr_%02d.pkl' % itr),
            self.agent.policy_opt,
            dir_path=dir_path
        )

        #LOGGER.info("Logging Policy... ")
        #self.agent.policy_opt.policy.pickle_policy(self.dO, self.dU,
        #                                           self.data_logger.dir_path + '/' + ('dualgps_policy_itr_%02d' % itr),
        #                                           goal_state=None,
        #                                           should_hash=False)

        # print("TODO: CHECK HOW TO SOLVE LOGGING DUAL ALGO")
        # # print("Logging GPS algorithm state... ")
        # # self.data_logger.pickle(
        # #     ('%s_algorithm_itr_%02d.pkl' % (type(self).__name__, itr)),
        # #     copy.copy(self)
        # # )

        LOGGER.info("Logging %s iteration data... ", type(self).__name__)
        self.data_logger.pickle(
            ('iteration_data_itr_%02d.pkl' % itr),
            copy.copy(self.cur),
            dir_path=dir_path
        )

        LOGGER.info("Logging Trajectory samples... ")
        self.data_logger.pickle(
            ('traj_sample_itr_%02d.pkl' % itr),
            copy.copy(traj_sample_lists),
            dir_path=dir_path
        )

        if pol_sample_lists is not None:
            LOGGER.info("Logging Global Policy samples... ")
            self.data_logger.pickle(
                ('pol_sample_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists),
                dir_path=dir_path
            )

        if pol_sample_lists_costs is not None:
            LOGGER.info("Logging Global Policy samples costs... ")
            self.data_logger.pickle(
                ('pol_sample_cost_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists_costs),
                dir_path=dir_path
            )

        if pol_sample_lists_cost_compositions is not None:
            LOGGER.info("Logging Global Policy samples cost compositions... ")
            self.data_logger.pickle(
                ('pol_sample_cost_composition_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists_cost_compositions),
                dir_path=dir_path
            )

    # For pickling.
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'agent' in state:
            state.pop('agent')
        if 'env' in state:
            state.pop('env')
        if 'cost_function' in state:
            state.pop('cost_function')
        if '_hyperparams' in state:
            state.pop('_hyperparams')
        if 'max_iterations' in state:
            state.pop('max_iterations')
        if 'policy_opt' in state:
            state.pop('policy_opt')
        return state

    # For unpickling.
    def __setstate__(self, state):
        self.__dict__.update(state)
        # self.__dict__ = state
        # self.__dict__['agent'] = None
