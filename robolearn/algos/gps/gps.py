"""
GPS
Authors: Finn et al
Adapted by robolearn collaborators
"""

import os
import sys
import traceback
import numpy as np
import scipy as sp
import copy
import random
import datetime

from robolearn.algos.rl_algorithm import RLAlgorithm

from robolearn.algos.gps.gps_config import *
from robolearn.algos.gps.gps_utils import PolicyInfo
from robolearn.algos.gps.gps_utils import IterationData, TrajectoryInfo, extract_condition

from robolearn.utils.sample import Sample
from robolearn.utils.sample_list import SampleList

from robolearn.agents.agent_utils import generate_noise
from robolearn.utils.data_logger import DataLogger
from robolearn.utils.print_utils import *
from robolearn.utils.plot_utils import *

import logging
LOGGER = logging.getLogger(__name__)
# Logging into console AND file
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
LOGGER.addHandler(ch)


class GPS(RLAlgorithm):
    def __init__(self, agent, env, **kwargs):
        super(GPS, self).__init__(agent, env, default_gps_hyperparams, kwargs)

        # Number of initial conditions
        self.M = self._hyperparams['conditions']

        if 'train_conditions' in self._hyperparams and self._hyperparams['train_conditions'] is not None:
            self._train_cond_idx = self._hyperparams['train_conditions']
            self._test_cond_idx = self._hyperparams['test_conditions']
        else:
            self._train_cond_idx = self._test_cond_idx = range(self.M)
            self._hyperparams['train_conditions'] = self._train_cond_idx
            self._hyperparams['test_conditions'] = self._test_cond_idx

        if 'data_files_dir' in self._hyperparams:
            if self._hyperparams['data_files_dir'] is None:
                self._data_files_dir = 'GPS_'+str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
            else:
                self._data_files_dir = self._hyperparams['data_files_dir']
        else:
            self._data_files_dir = 'GPS_'+str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        self.data_logger = DataLogger(self._data_files_dir)

        self.max_iterations = self._hyperparams['iterations']

        # ############################### #
        # Code used in original Algorithm #
        # ############################### #
        self._cond_idx = self._train_cond_idx  # TODO: Review code so only one is used

        self.iteration_count = 0

        # Get some values from the environment.
        self.dU = self._hyperparams['dU'] = env.get_action_dim()
        self.dX = self._hyperparams['dX'] = env.get_state_dim()
        self.dO = self._hyperparams['dO'] = env.get_obs_dim()

        # Get time values from the 'hyperparams'
        self.T = self._hyperparams['T']
        self.dt = self._hyperparams['dt']

        # Initial trajectory hyperparams
        init_traj_distr = self._hyperparams['init_traj_distr']
        init_traj_distr['x0'] = env.get_conditions()  # TODO: Check if it is better get_x0() or get_state()
        init_traj_distr['dX'] = self.dX
        init_traj_distr['dU'] = self.dU
        init_traj_distr['dt'] = self.dt
        init_traj_distr['T'] = self.T
        init_traj_distr['dQ'] = self.dX + self.dU

        # IterationData objects for each condition.
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]

        # Trajectory Info #
        # --------------- #
        # Traj. Info: Trajectory related variables:
        if self._hyperparams['fit_dynamics']:
            # Add dynamics if the algorithm requires fit_dynamics (Same type for all the conditions)
            dynamics = self._hyperparams['dynamics']

        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()

            if self._hyperparams['fit_dynamics']:
                self.cur[m].traj_info.dynamics = dynamics['type'](dynamics)

            # Get the initial trajectory distribution hyperparams
            init_traj_distr = extract_condition(self._hyperparams['init_traj_distr'], self._cond_idx[m])

            # Instantiate Trajectory Distribution: init_lqr or init_pd
            self.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr)

        # Traj Opt (Local policy opt) method #
        # ---------------------------------- #
        # Options: LQR, PI2
        self.traj_opt = self._hyperparams['traj_opt']['type'](self._hyperparams['traj_opt'])

        # Cost function #
        # ------------- #
        if self._hyperparams['cost'] is None:
            raise AttributeError("Cost function has not been defined")
        if isinstance(type(self._hyperparams['cost']), list):
            # One cost function for each condition
            self.cost_function = [self._hyperparams['cost'][i]['type'](self._hyperparams['cost'][i])
                                  for i in range(self.M)]
        else:
            # Same cost function for all conditions
            self.cost_function = [self._hyperparams['cost']['type'](self._hyperparams['cost'])
                                  for _ in range(self.M)]

        # KL step #
        # ------- #
        self.base_kl_step = self._hyperparams['kl_step']

        # ############# #
        # GPS Algorithm #
        # ############# #
        self.gps_algo = self._hyperparams['gps_algo']

        if self.gps_algo == 'pigps':
            gps_algo_hyperparams = default_pigps_hyperparams.copy()
            gps_algo_hyperparams.update(self._hyperparams['gps_algo_hyperparams'])
            self._hyperparams.update(gps_algo_hyperparams)

        if self.gps_algo in ['pigps', 'mdgps']:
            gps_algo_hyperparams = default_mdgps_hyperparams.copy()
            gps_algo_hyperparams.update(self._hyperparams['gps_algo_hyperparams'])
            self._hyperparams.update(gps_algo_hyperparams)

            # Policy Prior #
            # ------------ #
            policy_prior = self._hyperparams['policy_prior']
            for m in range(self.M):
                self.cur[m].pol_info = PolicyInfo(self._hyperparams)
                self.cur[m].pol_info.policy_prior = policy_prior['type'](policy_prior)

        # Global Policy #
        # ------------- #
        self.policy_opt = self.agent.policy_opt

        # OTHERS #
        # ------ #
        self.avg_cost_local_policies = np.zeros((self.max_iterations, self._hyperparams['num_samples']))

    def run(self, itr_load=None):
        """
        Run GPS.
        If itr_load is specified, first loads the algorithm state from that iteration
         and resumes training at the next iteration.
        :param itr_load: desired iteration to load algorithm from
        :return: 
        """
        run_successfully = True

        try:
            itr_start = self._initialize(itr_load)

            print("iteration from %d to %d" % (itr_start, self.max_iterations))
            for itr in range(itr_start, self.max_iterations):
                # Collect samples
                for cond in self._train_cond_idx:
                    for i in range(self._hyperparams['num_samples']):
                        print("")
                        print("#"*40)
                        print("Sample itr:%d/%d, cond:%d/%d, i:%d/%d" % (itr+1, self.max_iterations,
                                                                         cond+1, len(self._train_cond_idx),
                                                                         i+1, self._hyperparams['num_samples']))
                        print("#"*40)
                        self._take_sample(itr, cond, i, noisy=self._hyperparams['noisy_samples'],
                                          on_policy=self._hyperparams['sample_on_policy'])
                # Get agent's sample list
                traj_sample_lists = [self.agent.get_samples(cond, -self._hyperparams['num_samples'])
                                     for cond in self._train_cond_idx]

                # Clear agent samples.
                self.agent.clear_samples()  # TODO: Check if it is better to 'remember' these samples

                self._take_iteration(itr, traj_sample_lists)

                if self._hyperparams['test_after_iter']:
                    pol_sample_lists = self._take_policy_samples()
                else:
                    pol_sample_lists = None

                # Log data
                self._log_data(itr, traj_sample_lists, pol_sample_lists)

        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            print("#"*30)
            print("#"*30)
            print_skull()
            print("Panic: ERROR IN GPS!!!!")
            print("#"*30)
            print("#"*30)
            run_successfully = False
        finally:
            self._end()
            return run_successfully

    def _end(self):
        """
        Finish GPS and exit.
        :return: None
        """
        print("")
        print("GPS has finished!")
        self.env.stop()

    def _initialize(self, itr_load):
        """
        Initialize from the specified iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns:
            itr_start: Iteration to start from.
        """
        if itr_load is None:
            print('Starting GPS from zero!')
            return 0
        else:
            print('Loading previous GPS from iteration %d!' % itr_load)
            algorithm_file = '%s_algorithm_itr_%02d.pkl' % (self.gps_algo.upper(), itr_load)
            prev_algorithm = self.data_logger.unpickle(algorithm_file)
            if prev_algorithm is None:
                print("Error: cannot find '%s.'" % algorithm_file)
                os._exit(1)
            else:
                self.__dict__.update(prev_algorithm.__dict__)

            agent_file = 'agent_itr_%02d.pkl' % itr_load
            prev_agent = self.data_logger.unpickle(agent_file)
            if prev_agent is None:
                print("Error: cannot find '%s.'" % agent_file)
                os._exit(1)
            else:
                self.agent.__dict__.update(prev_agent.__dict__)

                traj_opt_file = 'policy_opt_itr_%02d.pkl' % itr_load
                self.agent.traj_opt = self.data_logger.unpickle(traj_opt_file)
                if self.agent.traj_opt is None:
                    print("Error: cannot find '%s.'" % traj_opt_file)
                    os._exit(1)
                else:
                    self.agent.__dict__.update(prev_agent.__dict__)

                self.agent.policy = self.agent.policy_opt.policy

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

    def _take_sample(self, itr, cond, i, verbose=True, save=True, noisy=True, on_policy=False):
        """
        Collect a sample from the environment.
        :param itr: Iteration number.
        :param cond: Condition number.
        :param i: Sample number.
        :param verbose: 
        :param save: 
        :param noisy: 
        :return: None
        """

        # On-policy or Off-policy
        if on_policy and (self.iteration_count > 0 or
                     ('sample_pol_first_itr' in self._hyperparams and self._hyperparams['sample_pol_first_itr'])):
            policy = self.agent.policy  # DOM: Instead self.opt_pol.policy
            print("On-policy sampling: %s!" % type(policy))
        else:
            policy = self.cur[cond].traj_distr
            print("Off-policy sampling: %s!" % type(policy))

        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        # Create a sample class
        # TODO: In original GPS code this is done with self._init_sample(self, condition, feature_fn=None) in agent
        sample = Sample(self.env, self.T)
        history = [None] * self.T
        obs_hist = [None] * self.T

        print("Resetting environment...")
        self.env.reset(time=2, cond=cond)
        import rospy

        ros_rate = rospy.Rate(int(1/self.dt))  # hz
        # Collect history
        for t in range(self.T):
            if verbose:
                if on_policy:
                    print("On-policy sample itr:%d/%d, cond:%d/%d, i:%d/%d | t:%d/%d" % (itr+1, self.max_iterations,
                                                                               cond+1, len(self._train_cond_idx),
                                                                               i+1, self._hyperparams['num_samples'],
                                                                               t+1, self.T))
                else:
                    print("Sample itr:%d/%d, cond:%d/%d, i:%d/%d | t:%d/%d" % (itr+1, self.max_iterations,
                                                                               cond+1, len(self._train_cond_idx),
                                                                               i+1, self._hyperparams['num_samples'],
                                                                               t+1, self.T))
            obs = self.env.get_observation()
            state = self.env.get_state()
            action = policy.eval(state, obs, t, noise[t, :])
            # action = np.zeros_like(action)
            # action[3] = -0.15707963267948966
            self.env.send_action(action)
            obs_hist[t] = (obs, action)
            history[t] = (state, action)
            # sample.set_acts(action, t=i)  # Set action One by one
            # sample.set_obs(obs[:42], obs_name='joint_state', t=i)  # Set action One by one
            # sample.set_states(state[:7], state_name='link_position', t=i)  # Set action One by one

            ros_rate.sleep()

        # Stop environment
        self.env.stop()

        all_actions = np.array([hist[1] for hist in history])
        all_states = np.array([hist[0] for hist in history])
        all_obs = np.array([hist[0] for hist in obs_hist])
        sample.set_acts(all_actions)  # Set all actions at the same time
        sample.set_obs(all_obs)  # Set all obs at the same time
        sample.set_states(all_states)  # Set all states at the same time

        if save:  # Save sample in agent sample list
            sample_id = self.agent.add_sample(sample, cond)
            print("The sample was added to Agent's sample list. Now there are %d sample(s) for condition '%d'." %
                  (sample_id+1, cond))

        # print("Plotting sample %d" % (i+1))
        # plot_sample(sample, data_to_plot='actions', block=True)
        # #plot_sample(sample, data_to_plot='states', block=True)

        return sample

    def _take_iteration(self, itr, sample_lists):
        """
        One iteration of the RL algorithm.
        Args:
            itr : Iteration to start from
            sample_lists : A list of samples collected from exploration
        Returns: None
        """
        print("")
        total_samples = sum([len(sample) for sample in sample_lists])
        print("%s iteration %d | Using %d samples in total." % (self.gps_algo.upper(), itr+1, total_samples))

        if self.gps_algo == 'pigps':
            self.iteration_pigps(sample_lists)

        elif self.gps_algo == 'mdgps':
            self.iteration_mdgps(sample_lists)

        else:
            raise NotImplementedError("GPS algorithm:'%s' NOT IMPLEMENTED!" % self.gps_algo)

    def _take_policy_samples(self, N=1, verbose=True):
        """
        Take samples from the global policy.
        :param N: number of policy samples to take per condition
        :param verbose: Print messages
        :return: 
        """
        """
        Take samples from the policy to see how it's doing.
        Args:
            N  : number of policy samples to take per condition
        Returns: None
        """

        pol_samples = [list() for _ in range(len(self._test_cond_idx))]

        for itr in range(N):
            # Collect samples
            for cond in self._test_cond_idx:
                for i in range(self._hyperparams['num_samples']):
                    if verbose:
                        print("")
                        print("#"*50)
                        print("Sample with AGENT POLICY itr:%d/%d, cond:%d/%d, i:%d/%d" % (itr+1, self.max_iterations,
                                                                                           cond+1,
                                                                                           len(self._train_cond_idx),
                                                                                           i+1,
                                                                                           self._hyperparams['num_samples']))
                        print("#"*50)
                    pol_samples[cond].append(self._take_sample(itr, cond, i, on_policy=True, noisy=False, save=False))

        return [SampleList(samples) for samples in pol_samples]

    def _log_data(self, itr, traj_sample_lists, pol_sample_lists=None):
        """
        Log data and algorithm.
        :param itr: Iteration number.
        :param traj_sample_lists: trajectory (local policies) samples as SampleList object.
        :param pol_sample_lists: global policy samples as SampleList object.
        :return: None
        """
        print("Logging Agent... ")
        self.data_logger.pickle(
            ('agent_itr_%02d.pkl' % itr),
            # copy.copy(temp_dict)
            copy.copy(self.agent)
        )
        print("Logging Policy_Opt... ")
        self.data_logger.pickle(
            ('policy_opt_itr_%02d.pkl' % itr),
            self.agent.policy_opt
        )

        print("Logging GPS algorithm state... ")
        self.data_logger.pickle(
            ('%s_algorithm_itr_%02d.pkl' % (self.gps_algo.upper(), itr)),
            copy.copy(self)
        )

        print("Logging Trajectory samples... ")
        self.data_logger.pickle(
            ('traj_sample_itr_%02d.pkl' % itr),
            copy.copy(traj_sample_lists)
        )

        if pol_sample_lists:
            print("Logging Global Policy samples... ")
            self.data_logger.pickle(
                ('pol_sample_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists)
            )

    def _update_dynamics(self):
        """
        Instantiate dynamics objects and update prior. Fit dynamics to
        current samples.
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
            self.cur[m].traj_info.x0sigma = np.diag(np.maximum(np.var(x0, axis=0),
                                                    self._hyperparams['initial_state_var']))

            prior = self.cur[m].traj_info.dynamics.get_prior()
            if prior:
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = len(cur_data)
                self.cur[m].traj_info.x0sigma += Phi + (N*priorm) / (N+priorm) * np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)

    def _update_trajectories(self):
        """
        Compute new linear Gaussian controllers.
        """
        print('-->Updating trajectories (local policies)...')
        if not hasattr(self, 'new_traj_distr'):
            self.new_traj_distr = [self.cur[cond].traj_distr for cond in range(self.M)]
        for cond in range(self.M):
            self.new_traj_distr[cond], self.cur[cond].eta = self.traj_opt.update(cond, self)

    def _eval_cost(self, cond):
        """
        Evaluate costs for all samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        # Constants.
        T, dX, dU = self.T, self.dX, self.dU
        N = len(self.cur[cond].sample_list)

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX+dU))
        Cm = np.zeros((N, T, dX+dU, dX+dU))
        for n in range(N):
            sample = self.cur[cond].sample_list[n]
            # Get costs.
            l, lx, lu, lxx, luu, lux = self.cost_function[cond].eval(sample)
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
        self.cur[cond].traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        self.cur[cond].traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        self.cur[cond].traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        self.cur[cond].cs = cs  # True value of cost.
        print('$$$$$$$')
        traj_sum = np.sum(self.cur[cond].cs, axis=1)
        print("Traj costs: %s " % traj_sum)
        print("Expected cost E[l(tau)]: %f" % np.average(traj_sum))
        print('$$$$$$$')
        self.avg_cost_local_policies[self.iteration_count, :] = traj_sum

    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', and advance iteration
        counter.
        :return: None
        """
        self.iteration_count += 1
        self.prev = copy.deepcopy(self.cur)
        # TODO: change IterationData to reflect new stuff better
        for m in range(self.M):
            self.prev[m].new_traj_distr = self.new_traj_distr[m]
        self.cur = [IterationData() for _ in range(self.M)]
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].traj_info.dynamics = copy.deepcopy(self.prev[m].traj_info.dynamics)
            self.cur[m].step_mult = self.prev[m].step_mult
            self.cur[m].eta = self.prev[m].eta
            self.cur[m].traj_distr = self.new_traj_distr[m]
        delattr(self, 'new_traj_distr')

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
                self._hyperparams['max_step_mult']),
            self._hyperparams['min_step_mult']
        )
        self.cur[m].step_mult = new_step

        if new_mult > 1:
            LOGGER.debug('Increasing step size multiplier to %f', new_step)
        else:
            LOGGER.debug('Decreasing step size multiplier to %f', new_step)

    def _measure_ent(self, m):
        """ Measure the entropy of the current trajectory. """
        ent = 0
        for t in range(self.T):
            ent = ent + np.sum(
                np.log(np.diag(self.cur[m].traj_distr.chol_pol_covar[t, :, :]))
            )
        return ent

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
        return state

    # For unpickling.
    def __setstate__(self, state):
        self.__dict__.update(state)
        # self.__dict__ = state
        # self.__dict__['agent'] = None

    """
    # ################### #
    # ################### #
    # ###### PIGPS ###### #
    # ################### #
    # ################### #
    PIGPS algorithm. 
    Author: C.Finn et al
    Reference:
    Y. Chebotar, M. Kalakrishnan, A. Yahya, A. Li, S. Schaal, S. Levine. 
    Path Integral Guided Policy Search. 2016. https://arxiv.org/abs/1610.00529.
    """
    def iteration_pigps(self, sample_lists):
        """
        Run iteration of PI-based guided policy search.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        # Store the samples and evaluate the costs.
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
            self._eval_cost(m)

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.iteration_count == 0:
            self.new_traj_distr = [self.cur[cond].traj_distr for cond in range(self.M)]
            self.update_policy_mdgps()

        # Update policy linearizations.
        for m in range(self.M):
            self.update_policy_fit_mdgps(m)

        # C-step
        self._update_trajectories()

        # S-step
        self.update_policy_mdgps()

        # Prepare for next iteration
        self.advance_iteration_variables_mdgps()

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
    def iteration_mdgps(self, sample_lists):
        """
        Run iteration of MDGPS-based guided policy search.
        :param sample_lists: List of SampleList objects for each condition.
        :return: None
        """
        # Store the samples and evaluate the costs.
        print('->Evaluating samples costs...')
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
            self._eval_cost(m)

        # Update dynamics linearizations (linear-Gaussian dynamics).
        print('->Updating dynamics linearization...')
        self._update_dynamics()

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.iteration_count == 0:
            self.new_traj_distr = [self.cur[cond].traj_distr for cond in range(self.M)]
            print('->S-step for init_traj_distribution (iter=0)...')
            self.update_policy_mdgps()

        # Update global policy linearizations.
        print('->Updating global policy linearization...')
        for m in range(self.M):
            self.update_policy_fit_mdgps(m)

        # C-step
        print('->| C-step |<-')
        if self.iteration_count > 0:
            print('-->Adjust step size multiplier (epsilon)...')
            self.stepadjust_mdgps()
        self._update_trajectories()

        # S-step
        print('->| S-step |<-')
        self.update_policy_mdgps()

        # Prepare for next iteration
        self.advance_iteration_variables_mdgps()

    def update_policy_mdgps(self):
        """
        Computes(updates) a new global policy.
        :return: 
        """
        print('-->Updating Global policy...')
        dU, dO, T = self.dU, self.dO, self.T
        # Compute target mean, cov, and weight for each sample.
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
                prc[:, t, :, :] = np.tile(traj.inv_pol_covar[t, :, :],
                                          [N, 1, 1])
                for i in range(N):
                    mu[i, t, :] = (traj.K[t, :, :].dot(X[i, t, :]) + traj.k[t, :])

                wt[:, t].fill(pol_info.pol_wt[t])

            tgt_mu = np.concatenate((tgt_mu, mu))
            tgt_prc = np.concatenate((tgt_prc, prc))
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs_data = np.concatenate((obs_data, samples.get_obs()))

        self.policy_opt.update(obs_data, tgt_mu, tgt_prc, tgt_wt)

    def update_policy_fit_mdgps(self, cond):
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
        X = samples.get_states()
        obs = samples.get_obs().copy()
        pol_mu, pol_sig = self.policy_opt.prob(obs)[:2]
        pol_info.pol_mu, pol_info.pol_sig = pol_mu, pol_sig

        # Update policy prior.
        policy_prior = pol_info.policy_prior
        samples = SampleList(self.cur[cond].sample_list)
        mode = self._hyperparams['policy_sample_mode']
        policy_prior.update(samples, self.policy_opt, mode)

        # Fit linearization and store in pol_info.
        pol_info.pol_K, pol_info.pol_k, pol_info.pol_S = \
            policy_prior.fit(X, pol_mu, pol_sig)
        for t in range(T):
            pol_info.chol_pol_S[t, :, :] = \
                sp.linalg.cholesky(pol_info.pol_S[t, :, :])

    def advance_iteration_variables_mdgps(self):
        """
        Move all 'cur' variables to 'prev', reinitialize 'cur'
        variables, and advance iteration counter.
        :return: None
        """
        self._advance_iteration_variables()
        for m in range(self.M):
            self.cur[m].traj_info.last_kl_step = \
                self.prev[m].traj_info.last_kl_step
            self.cur[m].pol_info = copy.deepcopy(self.prev[m].pol_info)

    def stepadjust_mdgps(self):
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

            # Compute values under Laplace approximation. This is the policy
            # that the previous samples were actually drawn from under the
            # dynamics that were estimated from the previous samples.
            prev_laplace[m] = self.traj_opt.estimate_cost(prev_nn, self.prev[m].traj_info).sum()
            # This is the actual cost that we experienced.
            prev_mc[m] = self.prev[m].cs.mean(axis=0).sum()
            # This is the policy that we just used under the dynamics that
            # were estimated from the prev samples (so this is the cost
            # we thought we would have).
            prev_predicted[m] = self.traj_opt.estimate_cost(prev_lg, self.prev[m].traj_info).sum()

        # Compute current cost.
        cur_laplace = np.empty(self.M)
        cur_mc = np.empty(self.M)
        for m in range(self.M):
            cur_nn = self.cur[m].pol_info.traj_distr()
            # This is the actual cost we have under the current trajectory
            # based on the latest samples.
            cur_laplace[m] = self.traj_opt.estimate_cost(
                cur_nn, self.cur[m].traj_info
            ).sum()
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
        if not self.gps_algo in ['pigps', 'mdgps']:
            NotImplementedError("Function not implemented for %s gps" % self.gps_algo)

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
            inv_pol_S = np.linalg.solve(
                pol_info.chol_pol_S[t, :, :],
                np.linalg.solve(pol_info.chol_pol_S[t, :, :].T, np.eye(dU))
            )
            KB, kB = pol_info.pol_K[t, :, :], pol_info.pol_k[t, :]
            PKLm[t, :, :] = np.vstack([
                np.hstack([KB.T.dot(inv_pol_S).dot(KB), -KB.T.dot(inv_pol_S)]),
                np.hstack([-inv_pol_S.dot(KB), inv_pol_S])
            ])
            PKLv[t, :] = np.concatenate([
                KB.T.dot(inv_pol_S).dot(kB), -inv_pol_S.dot(kB)
            ])
            fCm[t, :, :] = (Cm[t, :, :] + PKLm[t, :, :] * eta) / (eta + multiplier)
            fcv[t, :] = (cv[t, :] + PKLv[t, :] * eta) / (eta + multiplier)

        return fCm, fcv

    def get_avg_local_policy_costs(self):
        return self.avg_cost_local_policies
