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
import datetime

from robolearn.algos.rl_algorithm import RLAlgorithm

from robolearn.algos.gps.gps_config import *
from robolearn.algos.gps.gps_utils import PolicyInfo
from robolearn.algos.gps.gps_utils import IterationData, TrajectoryInfo, extract_condition

from robolearn.utils.sample.sample import Sample
from robolearn.utils.sample.sample_list import SampleList

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
                self._data_files_dir = 'robolearn_log/' + \
                                       'GPS_'+str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
            else:
                self._data_files_dir = 'robolearn_log/' + self._hyperparams['data_files_dir']
        else:
            self._data_files_dir = 'GPS_'+str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        self.data_logger = DataLogger(self._data_files_dir)

        self.max_iterations = self._hyperparams['iterations']

        self.iteration_count = 0

        # Get some values from the environment.
        self.dU = self._hyperparams['dU'] = env.get_action_dim()
        self.dX = self._hyperparams['dX'] = env.get_state_dim()
        self.dO = self._hyperparams['dO'] = env.get_obs_dim()

        # Get time values from 'hyperparams'
        self.T = self._hyperparams['T']
        self.dt = self._hyperparams['dt']

        # Initial trajectory hyperparams
        init_traj_distr = self._hyperparams['init_traj_distr']
        init_traj_distr['x0'] = env.get_conditions()  # TODO: Check if it is better get_x0() or get_state()
        init_traj_distr['dX'] = self.dX
        init_traj_distr['dU'] = self.dU
        init_traj_distr['dt'] = self.dt
        init_traj_distr['T'] = self.T

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
            init_traj_distr = extract_condition(self._hyperparams['init_traj_distr'], self._train_cond_idx[m])

            # Instantiate Trajectory Distribution: init_lqr or init_pd
            self.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr)

        self.new_traj_distr = None  # Last trajectory distribution optimized in C-step

        # Traj Opt (Local policy opt) method #
        # ---------------------------------- #
        # Options: LQR, PI2
        self.traj_opt = self._hyperparams['traj_opt']['type'](self._hyperparams['traj_opt'])

        # Cost function #
        # ------------- #
        if self._hyperparams['cost'] is None:
            raise AttributeError("Cost function has not been defined")
        if isinstance(type(self._hyperparams['cost']), list):
            # One cost function type for each condition
            self.cost_function = [self._hyperparams['cost'][i]['type'](self._hyperparams['cost'][i])
                                  for i in range(self.M)]
        else:
            # Same cost function type for all conditions
            self.cost_function = [self._hyperparams['cost']['type'](self._hyperparams['cost'])
                                  for _ in range(self.M)]

        # KL step #
        # ------- #
        self.base_kl_step = self._hyperparams['kl_step']

        # Global Policy #
        # ------------- #
        self.policy_opt = self.agent.policy_opt

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
                        print("Sample itr:%d/%d, cond:%d/%d, s:%d/%d" % (itr+1, self.max_iterations,
                                                                         cond+1, len(self._train_cond_idx),
                                                                         i+1, self._hyperparams['num_samples']))
                        print("#"*40)
                        sample = self._take_sample(itr, cond, i, noisy=self._hyperparams['noisy_samples'],
                                                   on_policy=self._hyperparams['sample_on_policy'],
                                                   verbose=False)
                        # if cond != 0:
                        #     sample = self._take_sample(itr, cond, i, noisy=self._hyperparams['noisy_samples'],
                        #                                on_policy=self._hyperparams['sample_on_policy'],
                        #                                verbose=False)
                        # else:
                        #     print("TODO: LOADING SAMPLE")
                        #     temp_dir_path = self.data_logger.dir_path
                        #     self.data_logger.dir_path = "TEMP_SAMPLES"
                        #     sample = self.data_logger.unpickle(('temp_mdreps_sample_%02d_%02d.pkl' % (itr, i)))
                        #     sample_id = self.agent.add_sample(sample, cond)
                        #     print("Adding to agent sample %d" % sample_id)
                        #     self.data_logger.dir_path = temp_dir_path

                        # if cond == 0:
                        #     print("TODO: Logging Sample... ")
                        #     temp_dir_path = self.data_logger.dir_path
                        #     self.data_logger.dir_path = "TEMP_SAMPLES"
                        #     self.data_logger.pickle(
                        #         ('temp_mdreps_sample_%02d_%02d.pkl' % (itr, i)),
                        #         # copy.copy(temp_dict)
                        #         copy.copy(sample)
                        #     )
                        #     self.data_logger.dir_path = temp_dir_path

                        # print("TODO: SAMPLING FAKE")
                        # self._take_fake_sample(itr, cond, i, noisy=self._hyperparams['noisy_samples'],
                        #                        on_policy=self._hyperparams['sample_on_policy'],
                        #                        verbose=False)

                # Get agent's sample list (TODO: WHAT HAPPEN IF SAVE SAMPLE WAS FALSE??)
                traj_sample_lists = [self.agent.get_samples(cond, -self._hyperparams['num_samples'])
                                     for cond in self._train_cond_idx]

                # for hh in range(len(traj_sample_lists)):
                #     plot_sample_list(traj_sample_lists[hh], data_to_plot='actions', block=False, cols=3)
                #     plot_sample_list(traj_sample_lists[hh], data_to_plot='states', block=False, cols=3)
                #     plot_sample_list(traj_sample_lists[hh], data_to_plot='obs', block=False, cols=3)

                # Clear agent samples.
                self.agent.clear_samples()  # TODO: Check if it is better to 'remember' these samples

                self._take_iteration(itr, traj_sample_lists)

                if self._hyperparams['test_after_iter']:
                    pol_sample_lists = self._take_policy_samples(N=self._hyperparams['test_samples'])

                    pol_sample_lists_costs, pol_sample_lists_cost_compositions = self._eval_conditions_sample_list_cost(pol_sample_lists)

                else:
                    pol_sample_lists = None
                    pol_sample_lists_costs = None
                    pol_sample_lists_cost_compositions = None

                # Log data
                self._log_data(itr, traj_sample_lists, pol_sample_lists, pol_sample_lists_costs,
                               pol_sample_lists_cost_compositions)

        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            print("#"*30)
            print("#"*30)
            print_skull()
            print("Panic: ERROR IN GPS ALGORITHM!!!!")
            print("#"*30)
            print("#"*30)
            run_successfully = False
        finally:
            self._end()
            return run_successfully

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
            itr_load -= 1
            algorithm_file = '%s_algorithm_itr_%02d.pkl' % (self.gps_algo.upper(), itr_load)
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

            if self.gps_algo.upper() == 'MDREPS':
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

        #plot_multi_info([noise], block=False, cols=3, legend=True, labels=range(7))

        # Create a sample class
        sample = Sample(self.env, self.T)
        history = [None] * self.T
        obs_hist = [None] * self.T

        print("Resetting environment...")
        from robolearn.envs.gym_environment import GymEnv
        self.env.reset(time=2, cond=cond)

        if issubclass(type(self.env), GymEnv):
            import time
            gym_ts = self.dt
        else:
            import rospy
            ros_rate = rospy.Rate(int(1/self.dt))  # hz

        # Collect history
        sampling_bar = ProgressBar(self.T, bar_title='Sampling')
        for t in range(self.T):
            sampling_bar.update(t)
            if verbose:
                if on_policy:
                    print("On-policy sample itr:%d/%d, cond:%d/%d, i:%d/%d | t:%d/%d" % (itr+1, self.max_iterations,
                                                                                         cond+1, len(self._train_cond_idx),
                                                                                         i+1, self._hyperparams['num_samples'],
                                                                                         t+1, self.T))
                else:
                    print("Sample itr:%d/%d, cond:%d/%d, s:%d/%d | t:%d/%d" % (itr+1, self.max_iterations,
                                                                               cond+1, len(self._train_cond_idx),
                                                                               i+1, self._hyperparams['num_samples'],
                                                                               t+1, self.T))
            obs = self.env.get_observation()
            state = self.env.get_state()
            # action = policy.eval(state, obs, t, noise[t, :])
            action = policy.eval(state.copy(), obs.copy(), t, noise[t, :].copy())  # TODO: Avoid TF policy writes in obs
            # print(obs[-6:])
            # print(state[-6:])
            # print("---")
            # action = np.zeros_like(action)
            # action[6] = -0.2
            # action[3] = -0.15707963267948966
            # print(obs)
            # print(state)
            # print(action)
            # print('----')
            self.env.send_action(action)
            obs_hist[t] = (obs, action)
            history[t] = (state, action)
            # sample.set_acts(action, t=i)  # Set action One by one
            # sample.set_obs(obs[:42], obs_name='joint_state', t=i)  # Set action One by one
            # sample.set_states(state[:7], state_name='link_position', t=i)  # Set action One by one

            if issubclass(type(self.env), GymEnv):
                time.sleep(gym_ts)
            else:
                ros_rate.sleep()

        sampling_bar.end()

        # Stop environment
        self.env.stop()
        print("Generating sample data...")

        all_actions = np.array([hist[1] for hist in history])
        all_states = np.array([hist[0] for hist in history])
        all_obs = np.array([hist[0] for hist in obs_hist])
        sample.set_acts(all_actions)   # Set all actions at the same time
        sample.set_obs(all_obs)        # Set all obs at the same time
        sample.set_states(all_states)  # Set all states at the same time
        sample.set_noise(noise)        # Set all noise at the same time

        #plot_multi_info([all_actions], block=False, cols=3, legend=True, labels=range(7))
        #raw_input('actions')

        if save:  # Save sample in agent sample list
            sample_id = self.agent.add_sample(sample, cond)
            print("The sample was added to Agent's sample list. Now there are %d sample(s) for condition '%d'." %
                  (sample_id+1, cond))

        # print("Plotting sample %d" % (i+1))
        # plot_sample(sample, data_to_plot='actions', block=True)
        # #plot_sample(sample, data_to_plot='states', block=True)

        return sample

    def _take_fake_sample(self, itr, cond, i, verbose=True, save=True, noisy=True, on_policy=False):
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

        sample.set_acts(all_actions)   # Set all actions at the same time
        sample.set_obs(all_obs)        # Set all obs at the same time
        sample.set_states(all_states)  # Set all states at the same time
        sample.set_noise(noise)        # Set all noise at the same time

        if save:  # Save sample in agent sample list
            sample_id = self.agent.add_sample(sample, cond)
            print("The sample was added to Agent's sample list. Now there are %d sample(s) for condition '%d'." %
                  (sample_id+1, cond))

        return sample

    def _take_policy_samples(self, N=1, verbose=True):
        """
        Take samples from the global policy.
        :param N: number of policy samples to take per condition
        :param verbose: Print messages
        :return: 
        """

        pol_samples = [list() for _ in range(len(self._test_cond_idx))]

        itr = self.iteration_count - 1  # Because it is called after self._advance_iteration_variables()

        # Collect samples
        for cond in self._test_cond_idx:
            for i in range(N):
                if verbose:
                    print("")
                    print("#"*50)
                    print("Sample with AGENT POLICY itr:%d/%d, cond:%d/%d, i:%d/%d" % (itr+1, self.max_iterations,
                                                                                       cond+1,
                                                                                       len(self._train_cond_idx),
                                                                                       i+1, N))
                    print("#"*50)
                pol_samples[cond].append(self._take_sample(itr, cond, i, on_policy=True, noisy=False, save=False,
                                                           verbose=False))

        sample_lists = [SampleList(samples) for samples in pol_samples]

        for cond, sample_list in enumerate(sample_lists):
            self.prev[cond].pol_info.policy_samples = sample_list  # prev because it is called after advance_iteration_variables

        return sample_lists

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

        self.iteration(sample_lists)

    def _eval_conditions_sample_list_cost(self, cond_sample_list):
        # costs = [list() for _ in range(len(sample_list))]
        # # Collect samples
        # for cond in range(len(sample_list)):
        #     for n_sample in range(len(sample_list[cond])):
        #         costs[cond].append(self.cost_function[cond].eval(sample_list[cond][n_sample])[0])
        costs = list()
        cost_compositions = list()
        total_cond = len(cond_sample_list)
        for cond in range(total_cond):
            N = len(cond_sample_list[cond])
            cs = np.zeros((N, self.T))
            cond_cost_composition = [None for _ in range(N)]
            for n in range(N):
                sample = cond_sample_list[cond][n]
                # Get costs.
                result = np.array(self.cost_function[cond].eval(sample))
                cs[n, :] = result[0]
                cond_cost_composition[n] = result[-1]
            costs.append(cs)
            cost_compositions.append(cond_cost_composition)
        return costs, cost_compositions
        #costs = list()
        ## Collect samples
        #for cond in range(len(sample_list)):
        #    cost = np.zeros((len(sample_list[cond]), self.T))
        #    for n_sample in range(len(sample_list[cond])):
        #        cost[n_sample, :] = self.cost_function[cond].eval(sample_list[cond][n_sample])[0]
        #    costs.append(cost)
        #return costs

    def _log_data(self, itr, traj_sample_lists, pol_sample_lists=None, pol_sample_lists_costs=None,
                  pol_sample_lists_cost_compositions=None):
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
        print("Logging Policy... ")
        self.agent.policy_opt.policy.pickle_policy(self.dO, self.dU,
                                                   self.data_logger.dir_path + '/' + ('policy_itr_%02d' % itr),
                                                   goal_state=None,
                                                   should_hash=False)

        print("Logging GPS algorithm state... ")
        self.data_logger.pickle(
            ('%s_algorithm_itr_%02d.pkl' % (self.gps_algo.upper(), itr)),
            copy.copy(self)
        )

        print("Logging GPS iteration data... ")
        self.data_logger.pickle(
            ('%s_iteration_data_itr_%02d.pkl' % (self.gps_algo.upper(), itr)),
            copy.copy(self.prev)  # prev instead of cur
        )

        print("Logging Trajectory samples... ")
        self.data_logger.pickle(
            ('traj_sample_itr_%02d.pkl' % itr),
            copy.copy(traj_sample_lists)
        )

        if pol_sample_lists is not None:
            print("Logging Global Policy samples... ")
            self.data_logger.pickle(
                ('pol_sample_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists)
            )

        if pol_sample_lists_costs is not None:
            print("Logging Global Policy samples costs... ")
            self.data_logger.pickle(
                ('pol_sample_cost_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists_costs)
            )

        if pol_sample_lists_cost_compositions is not None:
            print("Logging Global Policy samples cost compositions... ")
            self.data_logger.pickle(
                ('pol_sample_cost_composition_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists_cost_compositions)
            )

        if self.gps_algo.upper() == 'MDREPS':
            self.log_duality_vars(itr)

    def _update_dynamics(self):
        """
        Instantiate dynamics objects and update prior. Fit dynamics to current samples.
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
        if self.new_traj_distr is None:
            self.new_traj_distr = [self.cur[cond].traj_distr for cond in range(self.M)]
        for cond in range(self.M):
            self.new_traj_distr[cond], self.cur[cond].eta = self.traj_opt.update(cond, self)

    def _eval_sample_list_cost(self, sample_list, cond):
        """
        Evaluate costs for all current samples for a condition.
        Args:
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
        for n in range(N):
            sample = sample_list[n]
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

        return cs, (Cm, cv, cc)

    def _eval_cost(self, cond):
        """
        Evaluate costs for all current samples for a condition.
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
        self.cur[cond].traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        self.cur[cond].traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        self.cur[cond].traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        self.cur[cond].cs = cs  # True value of cost.

    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', and advance iteration counter.
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
            self.cur[m].traj_info.dynamics = copy.deepcopy(self.prev[m].traj_info.dynamics)
            self.cur[m].step_mult = self.prev[m].step_mult
            self.cur[m].eta = self.prev[m].eta
            self.cur[m].traj_distr = self.new_traj_distr[m]
        self.new_traj_distr = None

    def _set_new_mult(self, predicted_impr, actual_impr, m):
        """
        Adjust step size multiplier according to the predicted versus actual improvement.
        """
        # Model improvement as I = predicted_dI * KL + penalty * KL^2,
        # where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2).
        # Optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL =>
        # KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL.
        # Therefore, the new multiplier is given by pred/2*(pred-act).
        new_mult = predicted_impr / (2.0 * max(1e-4, predicted_impr - actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(min(new_mult * self.cur[m].step_mult, self._hyperparams['max_step_mult']),
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
        if 'policy_opt' in state:
            state.pop('policy_opt')
        return state

    # For unpickling.
    def __setstate__(self, state):
        self.__dict__.update(state)
        # self.__dict__ = state
        # self.__dict__['agent'] = None
