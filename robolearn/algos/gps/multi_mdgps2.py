"""
Multi MDGPS
Authors: Robolearn Collaborators
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

from robolearn.utils.sample.sample import Sample
from robolearn.utils.sample.sample_list import SampleList

from robolearn.agents.agent_utils import generate_noise
from robolearn.utils.data_logger import DataLogger
from robolearn.utils.print_utils import *
from robolearn.utils.plot_utils import *


class MultiMDGPS(RLAlgorithm):
    def __init__(self, agent, env, **kwargs):
        super(MultiMDGPS, self).__init__(agent, env, default_gps_hyperparams, kwargs)

        # Initial conditions
        self.M = self._hyperparams['conditions']
        self._train_cond_idx = self._hyperparams['train_conditions']
        self._test_cond_idx = self._hyperparams['test_conditions']

        # Number of Local Agents
        self.local_agent_state_masks = self._hyperparams['local_agent_state_masks']
        self.local_agent_action_masks = self._hyperparams['local_agent_action_masks']
        self.n_local_agents = len(self.local_agent_action_masks)

        # Data Logger
        if self._hyperparams['data_files_dir'] is None:
            self._data_files_dir = 'robolearn_log/' + 'GPS_'+str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        else:
            self._data_files_dir = 'robolearn_log/' + self._hyperparams['data_files_dir']
        self.data_logger = DataLogger(self._data_files_dir)

        # Training hyperparameters
        self.max_iterations = self._hyperparams['iterations']  # TODO: Move this to run
        self.iteration_count = 0

        # Get some values from the environment.
        self.dU = env.get_action_dim()
        self.dX = env.get_state_dim()
        self.dO = env.get_obs_dim()

        # Get time values from 'hyperparams'
        self.T = self._hyperparams['T']
        self.dt = self._hyperparams['dt']

        # Previous and Current IterationData objects for each local agent and condition
        self.cur = [[IterationData() for _ in range(self.M)] for _ in range(self.n_local_agents)]
        self.prev = [[IterationData() for _ in range(self.M)] for _ in range(self.n_local_agents)]

        # Trajectory Distribution
        dynamics_hyperparams = self._hyperparams['dynamics']  # Same dynamics for all local agents and conditions
        for a in range(self.n_local_agents):
            # Local Agent initial trajectory hyperparams for all conditions
            local_agent_init_traj_distr_hyperparams = self._hyperparams['init_traj_distr'][a]
            local_agent_init_traj_distr_hyperparams['x0'] = env.get_conditions()
            local_agent_init_traj_distr_hyperparams['dX'] = len(self.local_agent_state_masks[a])
            local_agent_init_traj_distr_hyperparams['dU'] = len(self.local_agent_action_masks[a])
            local_agent_init_traj_distr_hyperparams['dt'] = self.dt
            local_agent_init_traj_distr_hyperparams['T'] = self.T

            # Local Agent initial trajectory hyperparams and trajectory info for each condition
            for m in range(self.M):
                self.cur[a][m].traj_info = TrajectoryInfo()
                self.cur[a][m].traj_info.dynamics = dynamics_hyperparams['type'](dynamics_hyperparams)
                init_traj_distr_hyperparams = extract_condition(local_agent_init_traj_distr_hyperparams,
                                                                self._train_cond_idx[m])
                init_traj_distr_hyperparams['x0'] = init_traj_distr_hyperparams['x0'][self.local_agent_state_masks[a]]

                # Instantiate Trajectory Distribution: init_lqr or init_pd
                self.cur[a][m].traj_distr = init_traj_distr_hyperparams['type'](init_traj_distr_hyperparams)

        # Last trajectory distribution optimized in C-step
        self.new_traj_distr = [None for _ in range(self.n_local_agents)]

        # Traj Opt (Local policy opt) method (LQR or PI2)
        self.traj_opt = self._hyperparams['traj_opt']['type'](self._hyperparams['traj_opt'])

        # Cost function
        if isinstance(type(self._hyperparams['cost']), list):
            # One cost function type for each condition
            self.cost_function = [self._hyperparams['cost'][i]['type'](self._hyperparams['cost'][i])
                                  for i in range(self.M)]
        else:
            # Same cost function type for all conditions
            self.cost_function = [self._hyperparams['cost']['type'](self._hyperparams['cost'])
                                  for _ in range(self.M)]

        # Local Agents cost functions
        self.local_agent_costs = list()
        for a in range(self.n_local_agents):
            if isinstance(type(self._hyperparams['local_agent_costs'][a]), list):
                # One cost function type for each condition
                self.local_agent_costs.append([self._hyperparams['local_agent_costs'][a][i]['type'](self._hyperparams['local_agent_costs'][a][i])
                                      for i in range(self.M)])
            else:
                # Same cost function type for all conditions
                self.local_agent_costs.append([self._hyperparams['local_agent_costs'][a]['type'](self._hyperparams['local_agent_costs'][a])
                                      for _ in range(self.M)])

        # Base KL step
        self.base_kl_step = self._hyperparams['kl_step']

        # MDGPS variables
        policy_prior_hyperparams = self._hyperparams['policy_prior']
        for a in range(self.n_local_agents):
            pol_info_hyperparams = {'T': self._hyperparams['T'],
                                    'dU': len(self.local_agent_action_masks[a]),
                                    'dX': len(self.local_agent_state_masks[a]),
                                    'init_pol_wt': self._hyperparams['init_pol_wt']}
            for m in range(self.M):
                # Same policy prior type for all conditions and local agents
                self.cur[a][m].pol_info = PolicyInfo(pol_info_hyperparams)
                self.cur[a][m].pol_info.policy_prior = policy_prior_hyperparams['type'](policy_prior_hyperparams)

        # Global Policy
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
                        self._take_sample(itr, cond, i, noisy=self._hyperparams['noisy_samples'],
                                          on_policy=self._hyperparams['sample_on_policy'],
                                          verbose=False)
                # Get agent's sample list
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

                    pol_sample_lists_costs = self._eval_conditions_sample_list_cost(pol_sample_lists)

                else:
                    pol_sample_lists = None
                    pol_sample_lists_costs = None

                # Log data
                self._log_data(itr, traj_sample_lists, pol_sample_lists, pol_sample_lists_costs)

        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            print("#"*30)
            print_skull()
            print("Panic: ERROR IN GPS ALGORITHM!!!!")
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
            print('Starting MDGPS from zero!')
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
            policy = self.agent.policy
            print("On-policy sampling: %s!" % type(policy))
        else:
            policy = list()
            for a in range(self.n_local_agents):
                policy.append(self.cur[a][cond].traj_distr)
                print("Off-policy sampling for local agent %d: %s!" % (a, type(policy[a])))

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
            obs = self.env.get_observation().copy()  # TODO: Avoid TF policy writes in obs
            state = self.env.get_state().copy()
            # action = policy.eval(state, obs, t, noise[t, :])
            if isinstance(policy, list):
                action = np.zeros(self.dU)
                for a in range(self.n_local_agents):
                    local_agent_state = state[self.local_agent_state_masks[a]]
                    local_agent_noise = noise[t, self.local_agent_action_masks[a]].copy()
                    action[self.local_agent_action_masks[a]] = policy[a].eval(local_agent_state, obs, t, local_agent_noise)
            else:
                action = policy.eval(state, obs, t, noise[t, :].copy())
            # action[3] = -0.15707963267948966
            # print(obs)
            # print(state)
            # print(action)
            # print('----')
            #raw_input('TODAVIA NADAAA')
            self.env.send_action(action)
            obs_hist[t] = (obs, action)
            history[t] = (state, action)
            # sample.set_acts(action, t=i)  # Set action One by one
            # sample.set_obs(obs[:42], obs_name='joint_state', t=i)  # Set action One by one
            # sample.set_states(state[:7], state_name='link_position', t=i)  # Set action One by one

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

        if save:  # Save sample in agent sample list
            sample_id = self.agent.add_sample(sample, cond)
            print("The sample was added to Agent's sample list. Now there are %d sample(s) for condition '%d'." %
                  (sample_id+1, cond))

        # print("Plotting sample %d" % (i+1))
        # plot_sample(sample, data_to_plot='actions', block=True)
        # #plot_sample(sample, data_to_plot='states', block=True)

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

        for a in range(self.n_local_agents):
            for cond, sample_list in enumerate(sample_lists):
                self.prev[a][cond].pol_info.policy_samples = sample_list  # prev because it is called after advance_iteration_variables

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

        if self.gps_algo == 'pigps':
            self.iteration_pigps(sample_lists)

        elif self.gps_algo == 'mdgps':
            self.iteration_mdgps(sample_lists)

        else:
            raise NotImplementedError("GPS algorithm:'%s' NOT IMPLEMENTED!" % self.gps_algo)

    def _eval_conditions_sample_list_cost(self, cond_sample_list):
        # costs = [list() for _ in range(len(sample_list))]
        # # Collect samples
        # for cond in range(len(sample_list)):
        #     for n_sample in range(len(sample_list[cond])):
        #         costs[cond].append(self.cost_function[cond].eval(sample_list[cond][n_sample])[0])
        costs = list()
        total_cond = len(cond_sample_list)
        for cond in range(total_cond):
            N = len(cond_sample_list[cond])
            cs = np.zeros((N, self.T))
            for n in range(N):
                sample = cond_sample_list[cond][n]
                # Get costs.
                cs[n, :] = self.cost_function[cond].eval(sample)[0]
            costs.append(cs)
        return costs
        #costs = list()
        ## Collect samples
        #for cond in range(len(sample_list)):
        #    cost = np.zeros((len(sample_list[cond]), self.T))
        #    for n_sample in range(len(sample_list[cond])):
        #        cost[n_sample, :] = self.cost_function[cond].eval(sample_list[cond][n_sample])[0]
        #    costs.append(cost)
        #return costs

    def _log_data(self, itr, traj_sample_lists, pol_sample_lists=None, pol_sample_lists_costs=None):
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

    def _update_dynamics(self, a):
        """
        Instantiate dynamics objects and update prior. Fit dynamics to current samples.
        :param a: Local agent id
        :return: 
        """
        for m in range(self.M):
            cur_data = self.cur[a][m].sample_list
            X = cur_data.get_states()[:, :, self.local_agent_state_masks[a]]
            U = cur_data.get_actions()[:, :, self.local_agent_action_masks[a]]

            # Update prior and fit dynamics.
            self.cur[a][m].traj_info.dynamics.update_prior(cur_data, state_idx=self.local_agent_state_masks[a],
                                                           action_idx=self.local_agent_action_masks[a])
            self.cur[a][m].traj_info.dynamics.fit(X, U)

            # Fit x0mu/x0sigma.
            x0 = X[:, 0, :]
            x0mu = np.mean(x0, axis=0)
            self.cur[a][m].traj_info.x0mu = x0mu
            self.cur[a][m].traj_info.x0sigma = np.diag(np.maximum(np.var(x0, axis=0),
                                                    self._hyperparams['initial_state_var']))

            prior = self.cur[a][m].traj_info.dynamics.get_prior()
            if prior:
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = len(cur_data)
                self.cur[a][m].traj_info.x0sigma += Phi + (N*priorm) / (N+priorm) * np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)

    def _update_trajectories(self):
        """
        Compute new linear Gaussian controllers.
        """
        print('-->Updating trajectories (local policies)...')
        if self.new_traj_distr is None:
            self.new_traj_distr = [[self.cur[a][cond].traj_distr for cond in range(self.M)] for a in range(self.n_local_agents)]

        for a in range(self.n_local_agents):
            for cond in range(self.M):
                self.new_traj_distr[a][cond], self.cur[a][cond].eta = self.traj_opt.update(cond, self, a)

    def _eval_cost(self, cond, a):
        """
        Evaluate costs for all current samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
            a: Local agent id
        """
        # Constants.
        T = self.T

        N = len(self.cur[a][cond].sample_list)
        dX = len(self.local_agent_state_masks[a])
        dU = len(self.local_agent_action_masks[a])

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX+dU))
        Cm = np.zeros((N, T, dX+dU, dX+dU))


        act_idx = np.ix_(self.local_agent_action_masks[a], self.local_agent_action_masks[a])
        state_idx = np.ix_(self.local_agent_state_masks[a], self.local_agent_state_masks[a])
        for n in range(N):
            sample = self.cur[a][cond].sample_list[n]
            # Get costs.
            #l, lx, lu, lxx, luu, lux = self.cost_function[cond].eval(sample)
            l, lx, lu, lxx, luu, lux = self.local_agent_costs[a][cond].eval(sample)
            cc[n, :] = l
            cs[n, :] = l

            # Reshape for each local policy
            lx = lx[:, self.local_agent_state_masks[a]]
            lu = lu[:, self.local_agent_action_masks[a]]
            lxx = lxx[:, state_idx[0], state_idx[1]]
            luu = luu[:, act_idx[0], act_idx[1]]
            lux = lux[:, act_idx[0], state_idx[1]]

            # Assemble matrix and vector.
            cv[n, :, :] = np.c_[lx, lu]
            Cm[n, :, :, :] = np.concatenate(
                (np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]),
                axis=1
            )

            # Adjust for expanding cost around a sample.
            X = sample.get_states()[:, self.local_agent_state_masks[a]]
            U = sample.get_acts()[:, self.local_agent_action_masks[a]]
            yhat = np.c_[X, U]
            rdiff = -yhat
            rdiff_expand = np.expand_dims(rdiff, axis=2)
            cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)
            cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) + 0.5 * np.sum(rdiff * cv_update, axis=1)
            cv[n, :, :] += cv_update

        # Fill in cost estimate.
        self.cur[a][cond].traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        self.cur[a][cond].traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        self.cur[a][cond].traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        self.cur[a][cond].cs = cs  # True value of cost.

    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', and advance iteration counter.
        :return: None
        """
        self.iteration_count += 1
        self.prev = copy.deepcopy(self.cur)
        # TODO: change IterationData to reflect new stuff better
        for a in range(self.n_local_agents):
            for m in range(self.M):
                self.prev[a][m].new_traj_distr = self.new_traj_distr[a][m]

        # NEW IterationData object, and remove new_traj_distr
        self.cur = [[IterationData() for _ in range(self.M)] for _ in range(self.n_local_agents)]
        for a in range(self.n_local_agents):
            for m in range(self.M):
                self.cur[a][m].traj_info = TrajectoryInfo()
                self.cur[a][m].traj_info.dynamics = copy.deepcopy(self.prev[a][m].traj_info.dynamics)
                self.cur[a][m].step_mult = self.prev[a][m].step_mult
                self.cur[a][m].eta = self.prev[a][m].eta
                self.cur[a][m].traj_distr = self.new_traj_distr[a][m]
        self.new_traj_distr = None

    def _set_new_mult(self, predicted_impr, actual_impr, m, a):
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
        new_step = max(min(new_mult * self.cur[a][m].step_mult, self._hyperparams['max_step_mult']),
                       self._hyperparams['min_step_mult'])
        self.cur[a][m].step_mult = new_step

        if new_mult > 1:
            LOGGER.debug('Increasing step size multiplier for cond %d to %f', m, new_step)
        else:
            LOGGER.debug('Decreasing step size multiplier for cond %d to %f', m, new_step)

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
        if 'local_agent_costs' in state:
            state.pop('local_agent_costs')
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
        for a in range(self.n_local_agents):
            for m in range(self.M):
                # TODO: CHECK IF IT IS A GOOD IDEA TO SAVE ALL SAMPLE_LIST
                self.cur[a][m].sample_list = sample_lists[m]
                self._eval_cost(m, a)

        # Update dynamics linearizations (linear-Gaussian dynamics).
        print('->Updating dynamics linearization...')
        for a in range(self.n_local_agents):
            self._update_dynamics(a)

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.iteration_count == 0:
            self.new_traj_distr = list()
            # for a in range(self.n_local_agents):
            self.new_traj_distr = [[self.cur[a][cond].traj_distr for cond in range(self.M)] for a in range(self.n_local_agents)]
            print('->S-step for init_traj_distribution (iter=0)...')
            self.update_policy_mdgps()

        # Update global policy linearizations.
        print('->Updating global policy linearization...')
        for m in range(self.M):
            self.update_policy_fit_mdgps(m)

        # C-step
        if self.iteration_count > 0:
            print('-->Adjust step size (epsilon) multiplier...')
            self.stepadjust_mdgps()
        print('->| C-step |<-')
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
        # Compute target mean, cov(precision), and weight for each sample; and concatenate them.
        obs_data = np.zeros((0, T, dO))
        tgt_mu, tgt_prc, tgt_wt = np.zeros((0, T, dU)), np.zeros((0, T, dU, dU)), np.zeros((0, T))
        for m in range(self.M):
            for a in range(self.n_local_agents):
                local_act_idx = np.ix_(self.local_agent_action_masks[a], self.local_agent_action_masks[a])
                samples = self.cur[a][m].sample_list
                X = samples.get_states()[:, :, self.local_agent_state_masks[a]]
                N = len(samples)
                if a == 0:
                    mu = np.zeros((N, T, dU))
                    prc = np.zeros((N, T, dU, dU))
                    wt = np.zeros((N, T))
                traj = self.new_traj_distr[a][m]
                pol_info = self.cur[a][m].pol_info
                # Get time-indexed actions.
                for t in range(T):
                    # Compute actions along this trajectory.
                    prc[:, t, local_act_idx[0], local_act_idx[1]] = np.tile(traj.inv_pol_covar[t, :, :], [N, 1, 1])
                    for i in range(N):
                        mu[i, t, self.local_agent_action_masks[a]] = (traj.K[t, :, :].dot(X[i, t, :]) + traj.k[t, :])

                    wt[:, t].fill(pol_info.pol_wt[t])  # TODO: WE NEED TO DISCOVER WHAT POL_WT DOES!!

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
        T = self.T

        # Update for all local agents trajectories
        for a in range(self.n_local_agents):
            local_act_idx = np.ix_(self.local_agent_action_masks[a], self.local_agent_action_masks[a])

            # Choose samples to use.
            samples = self.cur[a][cond].sample_list
            pol_info = self.cur[a][cond].pol_info
            X = samples.get_states()[:, :, self.local_agent_state_masks[a]].copy()
            obs = samples.get_obs().copy()
            pol_mu, pol_sig = self.policy_opt.prob(obs)[:2]

            # Get corresponding pol_mu and pol_sigma
            pol_mu = pol_mu[:, :, self.local_agent_action_masks[a]]
            pol_sig = pol_sig[:, :, local_act_idx[0], local_act_idx[1]]

            pol_info.pol_mu, pol_info.pol_sig = pol_mu, pol_sig

            # Update policy prior.  # TODO: THIS STEPS ARE UNUSEFUL FOR CONSTPRIOR
            policy_prior = pol_info.policy_prior
            samples = SampleList(self.cur[a][cond].sample_list)
            mode = self._hyperparams['policy_sample_mode']
            policy_prior.update(samples, self.policy_opt, mode)

            # Fit linearization and store in pol_info.
            pol_info.pol_K, pol_info.pol_k, pol_info.pol_S = policy_prior.fit(X, pol_mu, pol_sig)
            for t in range(T):
                pol_info.chol_pol_S[t, :, :] = sp.linalg.cholesky(pol_info.pol_S[t, :, :])

    def advance_iteration_variables_mdgps(self):
        """
        Move all 'cur' variables to 'prev', reinitialize 'cur' variables, and advance iteration counter.
        :return: None
        """
        self._advance_iteration_variables()
        for a in range(self.n_local_agents):
            for m in range(self.M):
                self.cur[a][m].traj_info.last_kl_step = self.prev[a][m].traj_info.last_kl_step
                self.cur[a][m].pol_info = copy.deepcopy(self.prev[a][m].pol_info)

    def stepadjust_mdgps(self):
        """
        Calculate new step sizes. This version uses the same step size for all conditions.
        """
        for a in range(self.n_local_agents):
            # Compute previous cost and previous expected cost.
            prev_M = len(self.prev[a])  # May be different in future.
            prev_laplace = np.empty(prev_M)
            prev_mc = np.empty(prev_M)
            prev_predicted = np.empty(prev_M)
            for m in range(prev_M):
                prev_nn = self.prev[a][m].pol_info.traj_distr()
                prev_lg = self.prev[a][m].new_traj_distr

                # Compute values under Laplace approximation. This is the policy that the previous samples were actually
                # drawn from under the dynamics that were estimated from the previous samples.
                prev_laplace[m] = self.traj_opt.estimate_cost(prev_nn, self.prev[a][m].traj_info).sum()
                # This is the actual cost that we experienced.
                prev_mc[m] = self.prev[a][m].cs.mean(axis=0).sum()
                # This is the policy that we just used under the dynamics that were estimated from the prev samples (so
                # this is the cost we thought we would have).
                prev_predicted[m] = self.traj_opt.estimate_cost(prev_lg, self.prev[a][m].traj_info).sum()

            # Compute current cost.
            cur_laplace = np.empty(self.M)
            cur_mc = np.empty(self.M)
            for m in range(self.M):
                cur_nn = self.cur[a][m].pol_info.traj_distr()
                # This is the actual cost we have under the current trajectory based on the latest samples.
                cur_laplace[m] = self.traj_opt.estimate_cost(cur_nn, self.cur[a][m].traj_info).sum()
                cur_mc[m] = self.cur[a][m].cs.mean(axis=0).sum()

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
            LOGGER.debug('Previous cost Local Agent %d: Laplace: %f, MC: %f', a, prev_laplace, prev_mc)
            LOGGER.debug('Predicted cost: Laplace: %f', prev_predicted)
            LOGGER.debug('Actual cost Local Agent %d: Laplace: %f, MC: %f', a, cur_laplace, cur_mc)

            for m in range(self.M):
                self._set_new_mult(predicted_impr, actual_impr, m, a)

    def compute_costs_mdgps(self, a, m, eta, augment=True):
        """ Compute cost estimates used in the LQR backward pass. """

        traj_info, traj_distr = self.cur[a][m].traj_info, self.cur[a][m].traj_distr
        if not augment:  # Whether to augment cost with term to penalize KL
            return traj_info.Cm, traj_info.cv

        pol_info = self.cur[a][m].pol_info
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
