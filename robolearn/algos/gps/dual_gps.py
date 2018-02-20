"""
Author: Finn et al
Adapted by robolearn collaborators
"""

import os
import copy

from robolearn.algos.gps.mdgps import MDGPS
from robolearn.algos.dualism import Dualism
from robolearn.utils.experience_buffer import get_bigger_idx
from robolearn.utils.sample.sample_list import SampleList

from robolearn.utils.plot_utils import *


class DualGPS(MDGPS, Dualism):
    """
    Sample-based joint policy learning and trajectory optimization with
    (approximate) mirror descent guided policy search algorithm.
    """
    def __init__(self, agent, env, **kwargs):

        MDGPS.__init__(self, agent, env, **kwargs)

        # Duality Data #
        # ------------ #
        Dualism.__init__(self)

        # Dualist KL base values #
        self.base_kl_good = self._hyperparams['algo_hyperparams']['kl_good']
        self.base_kl_bad = self._hyperparams['algo_hyperparams']['kl_bad']
        # Set initial dual variables
        for m in range(self.M):
            self.cur[m].nu = self._hyperparams['algo_hyperparams']['init_nu']
            self.cur[m].omega = self._hyperparams['algo_hyperparams']['init_omega']

    def _iteration(self, itr):
        logger = self.logger

        # Sample from environment using current trajectory distributions
        logger.info('')
        logger.info('DualGPS: itr:%02d | Sampling from local trajectories...'
                    % (itr+1))
        traj_or_pol = 'traj'
        self._take_sample(traj_or_pol, itr, 'train')

        # Get last samples from agent
        n_samples = self._hyperparams['num_samples']
        traj_sample_lists = [self.agent.get_samples(cond, -n_samples)
                             for cond in self._train_cond_idx]

        # TODO: Check if it is better to 'remember' these samples
        # Clear agent sample
        self.agent.clear_samples()

        for m, m_train in enumerate(self._train_cond_idx):
            self.cur[m_train].sample_list = traj_sample_lists[m]

        # Update dynamics model using all samples.
        logger.info('')
        logger.info('%s: itr:%02d | '
                    'Updating dynamics linearization...'
                    % (type(self).__name__, itr+1))
        self._update_dynamic_model()

        logger.info('')
        logger.info('%s: itr:%02d | '
                    'Evaluating samples costs...'
                    % (type(self).__name__, itr+1))
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

        logger.info('')
        logger.info('DualGPS: itr:%02d | '
                    'Getting good and bad trajectories...' % (itr+1))
        self._update_good_samples()
        self._update_bad_samples()

        logger.info('')
        logger.info('DualGPS: itr:%02d | '
                    'Updating data of good and bad samples...' % (itr+1))
        logger.info('-DualGPS: itr:%02d | '
                    'Update g/b costs...' % (itr+1))
        self._eval_good_bad_samples_costs()
        # logger.info('-DualGPS: itr:%02d | '
        #             'Update g/b dynamics...' % (itr+1))
        # option = self._hyperparams['algo_hyperparams']['duality_dynamics_type']
        # self._update_good_bad_dynamics(option=option)
        logger.info('-DualGPS: itr:%02d | '
                    'Update g/b traj dist...' % (itr+1))
        self._update_good_bad_fit()
        # logger.info('-DualGPS: itr:%02d | '
        #             'Divergence btw good/bad trajs: ...' % (itr+1))
        # self._check_kl_div_good_bad()

        # C-step
        logger.info('')
        logger.info('DualGPS: itr:%02d | '
                    'Updating trajectories...' % (itr+1))
        for ii in range(self._hyperparams['algo_hyperparams']
                        ['inner_iterations']):
            logger.info('-DualGPS: itr:%02d | Inner iteration %d/%d'
                        % (itr+1, ii+1,
                           self._hyperparams['algo_hyperparams']
                           ['inner_iterations']))
            self._update_trajectories()

        # S-step
        logger.info('')
        logger.info('%s:itr:%02d | ->| S-step |<-'
                    % (type(self).__name__, itr+1))
        self._update_policy(
            self._hyperparams['algo_hyperparams']['forget_bad_samples'])

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

    def compute_traj_cost(self, cond, eta, omega, nu, augment=True):
        """
        Compute cost estimates used in the LQR backward pass.

        :param cond: Number of condition
        :param eta: Dual variable corresponding to KL divergence with
                    previous policy.
        :param omega: Dual variable(s) corresponding to KL divergence with
                      good trajectories.
        :param nu: Dual variable(s) corresponding to KL divergence with
                   bad trajectories.
        :param augment: True if we want a KL constraint for all time-steps.
                        False otherwise. True for MDGPS
        :return: Cm and cv
        """
        traj_info = self.cur[cond].traj_info
        traj_distr = self.cur[cond].traj_distr  # We do not use it
        good_distr = self.good_duality_info[cond].traj_dist
        bad_distr = self.bad_duality_info[cond].traj_dist

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
        divisor = (eta + omega - nu + multiplier + 1e-6)
        fCm = Cm / divisor
        fcv = cv / divisor

        # We are dividing the surrogate cost calculation for debugging purposes

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
        for t in range(self.T):
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

    def _update_trajectories(self):
        """
        Compute new linear Gaussian controllers using the TrajOpt algorithm.
        """
        LOGGER = self.logger

        LOGGER.info('-->DualGPS: Updating trajectories (local policies)...')
        if self.new_traj_distr is None:
            self.new_traj_distr = [self.cur[cond].traj_distr
                                   for cond in range(self.M)]
        for cond in range(self.M):
            traj_opt_outputs = self.traj_opt.update(cond, self)
            self.new_traj_distr[cond] = traj_opt_outputs[0]
            self.cur[cond].eta = traj_opt_outputs[1]
            self.cur[cond].nu = traj_opt_outputs[2]
            self.cur[cond].omega = traj_opt_outputs[3]

    def _update_policy(self, remove_bad=False):
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

            if remove_bad:
                n_bad = self._hyperparams['algo_hyperparams']['n_bad_samples']
                cs = self.cur[m].cs
                if n_bad == cs.shape[0]:
                    raise ValueError("We cannot remove all trajs in SL tep")
                else:
                    worst_indeces = get_bigger_idx(np.sum(cs, axis=1), n_bad)
                idxs = np.setdiff1d(np.arange(len(samples)), worst_indeces)
                samples = SampleList(samples.get_samples(idxs))

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

    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', and advance iteration counter of
        the algorithm.
        :return: None
        """
        MDGPS._advance_iteration_variables(self)

        # Duality variables
        for m in range(self.M):
            self.cur[m].nu = self.prev[m].nu
            self.cur[m].omega = self.prev[m].omega

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

    def _log_iter_data(self, itr, traj_sample_lists, pol_sample_lists=None,
                       pol_sample_lists_costs=None,
                       pol_sample_lists_cost_compositions=None):
        """
        log data and algorithm.
        :param itr: iteration number.
        :param traj_sample_lists: trajectory (local policies) samples as
                                  samplelist object.
        :param pol_sample_lists: global policy samples as samplelist object.
        :return: none
        """
        # logger = self.logger
        # logger.warning('*'*20)
        # logger.warning('no logging agent, pol and algo dataaaaaaaaaaaaaaaa')
        # logger.warning('*'*20)
        #
        # dir_path = self.data_logger.dir_path + ('/itr_%02d' % itr)
        #
        # # logger.info("logging agent... ")
        # # self.data_logger.pickle(
        # #     ('dualgps_itr_%02d.pkl' % itr),
        # #     # copy.copy(temp_dict)
        # #     copy.copy(self.agent)
        # # )
        # logger.info("logging policy_opt... ")
        # self.data_logger.pickle(
        #     ('policy_opt_itr_%02d.pkl' % itr),
        #     self.agent.policy_opt,
        #     dir_path=dir_path
        # )
        # print("todo: not logging policy!!!")
        # #logger.info("logging policy... ")
        # #self.agent.policy_opt.policy.pickle_policy(self.do, self.du,
        # #                                           self.data_logger.dir_path + '/' + ('dualgps_policy_itr_%02d' % itr),
        # #                                           goal_state=none,
        # #                                           should_hash=false)
        #
        # # print("todo: check how to solve logging dual algo")
        # # # print("logging gps algorithm state... ")
        # # # self.data_logger.pickle(
        # # #     ('%s_algorithm_itr_%02d.pkl' % (type(self).__name__, itr)),
        # # #     copy.copy(self)
        # # # )
        #
        # logger.info("logging gps iteration data... ")
        # self.data_logger.pickle(
        #     ('iteration_data_itr_%02d.pkl' % itr),
        #     copy.copy(self.cur),
        #     dir_path=dir_path
        # )
        #
        # logger.info("logging trajectory samples... ")
        # self.data_logger.pickle(
        #     ('traj_sample_itr_%02d.pkl' % itr),
        #     copy.copy(traj_sample_lists),
        #     dir_path=dir_path
        # )
        #
        # if pol_sample_lists is not none:
        #     logger.info("logging global policy samples... ")
        #     self.data_logger.pickle(
        #         ('pol_sample_itr_%02d.pkl' % itr),
        #         copy.copy(pol_sample_lists),
        #         dir_path=dir_path
        #     )
        #
        # if pol_sample_lists_costs is not none:
        #     logger.info("logging global policy samples costs... ")
        #     self.data_logger.pickle(
        #         ('pol_sample_cost_itr_%02d.pkl' % itr),
        #         copy.copy(pol_sample_lists_costs),
        #         dir_path=dir_path
        #     )
        #
        # if pol_sample_lists_cost_compositions is not none:
        #     logger.info("logging global policy samples cost compositions... ")
        #     self.data_logger.pickle(
        #         ('pol_sample_cost_composition_itr_%02d.pkl' % itr),
        #         copy.copy(pol_sample_lists_cost_compositions),
        #         dir_path=dir_path
        #     )

        MDGPS._log_iter_data(self, itr, traj_sample_lists,
                             pol_sample_lists=pol_sample_lists,
                             pol_sample_lists_costs=pol_sample_lists_costs,
                             pol_sample_lists_cost_compositions=pol_sample_lists_cost_compositions)

        dir_path = self.data_logger.dir_path + ('/itr_%02d' % itr)

        self.logger.info("Logging God/Bad duality data")
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

    # # For pickling.
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     if 'agent' in state:
    #         state.pop('agent')
    #     if 'env' in state:
    #         state.pop('env')
    #     if 'cost_function' in state:
    #         state.pop('cost_function')
    #     if '_hyperparams' in state:
    #         state.pop('_hyperparams')
    #     if 'max_iterations' in state:
    #         state.pop('max_iterations')
    #     if 'policy_opt' in state:
    #         state.pop('policy_opt')
    #     return state
    #
    # # For unpickling.
    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     # self.__dict__ = state
    #     # self.__dict__['agent'] = None
