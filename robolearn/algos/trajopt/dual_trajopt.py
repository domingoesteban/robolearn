import copy
import numpy as np

from robolearn.algos.trajopt.traj_opt import TrajOpt
from robolearn.algos.dualism import Dualism
from robolearn.algos.trajopt.trajopt_config import DEFAULT_DUALTRAJOPT_HYPERPARAMS

from robolearn.algos.gps.gps_utils import extract_condition
from robolearn.algos.gps.gps_utils import TrajectoryInfo

from robolearn.utils.traj_opt.dualist_traj_opt import DualistTrajOpt


class DualTrajOpt(TrajOpt, Dualism):
    def __init__(self, agent, env, **kwargs):
        TrajOpt.__init__(self, agent, env, DEFAULT_DUALTRAJOPT_HYPERPARAMS,
                         **kwargs)

        # Traj Opt (Local policy opt) method #
        # ---------------------------------- #
        if self._hyperparams['traj_opt']['type'].__name__ != 'DualistTrajOpt':
            raise ValueError("The selected traj_opt method is not %s!"
                             % DualistTrajOpt.__name__)
        self.traj_opt = \
            self._hyperparams['traj_opt']['type'](self._hyperparams['traj_opt'])
        self.traj_opt.set_logger(self.logger)

        # KL base values #
        # -------------- #
        self.base_kl_good = self._hyperparams['algo_hyperparams']['kl_good']
        self.base_kl_bad = self._hyperparams['algo_hyperparams']['kl_bad']
        # Set initial dual variables
        for m in range(self.M):
            self.cur[m].nu = self._hyperparams['algo_hyperparams']['init_nu']
            self.cur[m].omega = self._hyperparams['algo_hyperparams']['init_omega']

        # Duality Data #
        # ------------ #
        Dualism.__init__(self)

    def _iteration(self, itr):
        logger = self.logger

        # Sample from environment using current trajectory distributions
        logger.info('')
        logger.info('DualTrajOpt: itr:%02d | Sampling from local trajectories..'
                    % (itr+1))
        traj_sample_lists = self._take_sample(itr)
        #
        # # Get last samples from agent
        # n_samples = self._hyperparams['num_samples']
        # traj_sample_lists = [self.agent.get_samples(cond, -n_samples)
        #                      for cond in self._train_cond_idx]
        #
        # # TODO: Check if it is better to 'remember' these samples
        # # Clear agent sample
        # self.agent.clear_samples()

        for m, m_train in enumerate(self._train_cond_idx):
            self.cur[m_train].sample_list = traj_sample_lists[m]

        # Update dynamics model using all samples.
        logger.info('')
        logger.info('DualTrajOpt: itr:%02d | '
                    'Updating dynamics linearization...' % (itr+1))
        self._update_dynamic_model()

        # Evaluate cost function for all conditions and samples.
        logger.info('')
        logger.info('DualTrajOpt: itr:%02d | '
                    'Evaluating samples costs...' % (itr+1))
        for m in range(self.M):
            self._eval_iter_samples_cost(m)

        logger.info('')
        logger.info('DualTrajOpt: itr:%02d | '
                    'Getting good and bad trajectories...' % (itr+1))
        self._update_good_samples()
        self._update_bad_samples()

        logger.info('')
        logger.info('DualTrajOpt: itr:%02d | '
                    'Updating data of good and bad samples...' % (itr+1))
        logger.info('-DualTrajOpt: itr:%02d | '
                    'Update g/b costs...' % (itr+1))
        self._eval_good_bad_samples_costs()
        # logger.info('-DualTrajOpt: itr:%02d | '
        #             'Update g/b dynamics...' % (itr+1))
        # option = self._hyperparams['algo_hyperparams']['duality_dynamics_type']
        # self._update_good_bad_dynamics(option=option)
        logger.info('-DualTrajOpt: itr:%02d | '
                    'Update g/b traj dist...' % (itr+1))
        self._update_good_bad_fit()
        # logger.info('-DualTrajOpt: itr:%02d | '
        #             'Divergence btw good/bad trajs: ...' % (itr+1))
        # self._check_kl_div_good_bad()


        # Update KL step
        logger.info('')
        logger.info('DualTrajOpt: itr:%02d | Updating KL step size...'
                    % (itr+1))
        self._update_step_size()

        logger.info('DualTrajOpt: itr:%02d | Updating KL bad/good size...'
                    % (itr+1))
        self._update_good_bad_size()

        # C-step
        logger.info('')
        logger.info('DualTrajOpt: itr:%02d | '
                    'Updating trajectories...' % (itr+1))
        for ii in range(self._hyperparams['algo_hyperparams']
                        ['inner_iterations']):
            logger.info('-DualTrajOpt: itr:%02d | Inner iteration %d/%d'
                        % (itr+1, ii+1,
                           self._hyperparams['algo_hyperparams']
                           ['inner_iterations']))
            self._update_trajectories()

        sample_lists_costs = [None for _ in traj_sample_lists]
        sample_lists_cost_compositions = [None for _ in traj_sample_lists]
        for cc, sample_list in enumerate(traj_sample_lists):
            cost_fcn = self.cost_function[cc]
            costs = self._eval_sample_list_cost(sample_list, cost_fcn)
            sample_lists_costs[cc] = costs[0]
            sample_lists_cost_compositions[cc] = costs[2]

        for m, cond in enumerate(self._train_cond_idx):
            print('&'*10)
            print('Average Cost')
            print('Condition:%02d' % cond)
            print('Avg cost: %f'
                  % sample_lists_costs[m].sum(axis=1).mean())
            print('&'*10)

        # Log data
        self._log_iter_data(itr, traj_sample_lists, sample_lists_costs,
                            sample_lists_cost_compositions)

        # Prepare everything for next iteration
        self._advance_iteration_variables()

    def compute_traj_cost(self, cond, eta, nu, omega, augment=True):
        """
        Compute cost estimates used in the LQR backward pass.

        :param cond: Number of condition
        :param eta: Dual variable corresponding to KL divergence with
                    previous policy.
        :param nu: Dual variable(s) corresponding to KL divergence with
                   bad trajectories.
        :param omega: Dual variable(s) corresponding to KL divergence with
                      good trajectories.
        :param augment: True if we want a KL constraint for all time-steps.
                        False otherwise. True for MDGPS
        :return: Cm and cv
        """
        traj_info = self.cur[cond].traj_info
        traj_distr = self.cur[cond].traj_distr
        good_distr = self.good_duality_info[cond].traj_dist
        bad_distr = self.bad_duality_info[cond].traj_dist

        if not augment:  # Whether to augment cost with term to penalize KL
            return traj_info.Cm, traj_info.cv

        T = self.T
        dX = self.dX
        dU = self.dU

        Cm, cv = np.copy(traj_info.Cm), np.copy(traj_info.cv)

        # Weight of maximum entropy term in trajectory optimization
        multiplier = self._hyperparams['max_ent_traj']

        # Surrogate cost
        PKLm = np.zeros((T, dX+dU, dX+dU))
        PKLv = np.zeros((T, dX+dU))

        self.logger.warning('WARN: adding a beta to divisor in '
                            'compute_traj_cost')
        divisor = (eta + omega - nu + multiplier + 1e-6)
        fCm = Cm / divisor
        fcv = cv / divisor

        # We are dividing the surrogate cost calculation for debugging purposes

        # Add in the KL divergence with previous policy.
        for t in range(self.T-1, -1, -1):
            # Policy KL-divergence terms.
            KB = traj_distr.K[t, :, :]
            kB = traj_distr.k[t, :]
            inv_pol_S = traj_distr.inv_pol_covar[t, :, :]

            PKLm[t, :, :] = np.vstack([
                np.hstack([KB.T.dot(inv_pol_S).dot(KB), -KB.T.dot(inv_pol_S)]),
                np.hstack([-inv_pol_S.dot(KB), inv_pol_S])
            ])
            PKLv[t, :] = np.concatenate([
                KB.T.dot(inv_pol_S).dot(kB), -inv_pol_S.dot(kB)
            ])
            fCm[t, :, :] += PKLm[t, :, :] * eta / divisor
            fcv[t, :] += PKLv[t, :] * eta / divisor

        # Subtract in the KL divergence with bad trajectories.
        for t in range(self.T-1, -1, -1):
            # Bad KL-divergence terms.
            inv_pol_S = bad_distr.inv_pol_covar[t, :, :]
            KB = bad_distr.K[t, :, :]
            kB = bad_distr.k[t, :]

            PKLm[t, :, :] = np.vstack([
                np.hstack([KB.T.dot(inv_pol_S).dot(KB), -KB.T.dot(inv_pol_S)]),
                np.hstack([-inv_pol_S.dot(KB), inv_pol_S])
            ])
            PKLv[t, :] = np.concatenate([
                KB.T.dot(inv_pol_S).dot(kB), -inv_pol_S.dot(kB)
            ])
            fCm[t, :, :] -= PKLm[t, :, :] * nu / divisor
            fcv[t, :] -= PKLv[t, :] * nu / divisor

        # Add in the KL divergence with good trajectories.
        for t in range(self.T-1, -1, -1):
            # Good KL-divergence terms.
            inv_pol_S = good_distr.inv_pol_covar[t, :, :]
            KB = good_distr.K[t, :, :]
            kB = good_distr.k[t, :]

            PKLm[t, :, :] = np.vstack([
                np.hstack([KB.T.dot(inv_pol_S).dot(KB), -KB.T.dot(inv_pol_S)]),
                np.hstack([-inv_pol_S.dot(KB), inv_pol_S])
            ])
            PKLv[t, :] = np.concatenate([
                KB.T.dot(inv_pol_S).dot(kB), -inv_pol_S.dot(kB)
            ])
            fCm[t, :, :] += PKLm[t, :, :] * omega / divisor
            fcv[t, :] += PKLv[t, :] * omega / divisor

        return fCm, fcv

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

    def _update_trajectories(self):
        """
        Compute new linear Gaussian controllers using the TrajOpt algorithm.
        """
        LOGGER = self.logger

        LOGGER.info('-->TrajOpt: Updating trajectories (local policies)...')
        if self.new_traj_distr is None:
            self.new_traj_distr = [self.cur[cond].traj_distr
                                   for cond in range(self.M)]
        for cond in range(self.M):
            traj_opt_outputs = self.traj_opt.update(cond, self)
            self.new_traj_distr[cond] = traj_opt_outputs[0]
            self.cur[cond].eta = traj_opt_outputs[1]
            self.cur[cond].nu = traj_opt_outputs[2]
            self.cur[cond].omega = traj_opt_outputs[3]

    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', and advance iteration counter of
        the algorithm.
        :return: None
        """
        TrajOpt._advance_iteration_variables(self)

        # Duality variables
        for m in range(self.M):
            self.cur[m].nu = self.prev[m].nu
            self.cur[m].omega = self.prev[m].omega

    def _log_iter_data(self, itr, traj_sample_lists,
                       sample_lists_costs=None,
                       sample_lists_cost_compositions=None):
        """
        Log data and algorithm.
        :param itr: Iteration number.
        :param traj_sample_lists: trajectory (local policies) samples as
                                  SampleList object.
        :param pol_sample_lists: global policy samples as SampleList object.
        :return: None
        """
        TrajOpt._log_iter_data(self, itr, traj_sample_lists,
                               sample_lists_costs=sample_lists_costs,
                               sample_lists_cost_compositions=sample_lists_cost_compositions)

        LOGGER = self.logger
        dir_path = self.data_logger.dir_path + ('/itr_%02d' % itr)

        LOGGER.info("Logging God/Bad duality data")
        self.data_logger.pickle(
            ('good_trajectories_info_itr_%02d.pkl' % itr),
            copy.copy(self.good_trajs_info),
            dir_path=dir_path
        )
        self.data_logger.pickle(
            ('bad_trajectories_info_itr_%02d.pkl' % itr),
            copy.copy(self.bad_trajs_info),
            dir_path=dir_path
        )
        self.data_logger.pickle(
            ('good_duality_info_itr_%02d.pkl' % itr),
            copy.copy(self.good_duality_info),
            dir_path=dir_path
        )
        self.data_logger.pickle(
            ('bad_duality_info_itr_%02d.pkl' % itr),
            copy.copy(self.bad_duality_info),
            dir_path=dir_path
        )

