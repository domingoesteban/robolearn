"""
input('aa')n et al
Adapted by robolearn collaborators
"""

import os
import sys
import traceback
import numpy as np
import scipy as sp
import copy
import datetime

from robolearn.algos.algorithm import Algorithm

from robolearn.algos.gps.gps_config import *
from robolearn.algos.gps.gps_utils import PolicyInfo
from robolearn.algos.gps.gps_utils import IterationData, TrajectoryInfo, extract_condition, DualityInfo

from robolearn.utils.sample.sample import Sample
from robolearn.utils.sample.sample_list import SampleList

from robolearn.agents.agent_utils import generate_noise
from robolearn.utils.data_logger import DataLogger
from robolearn.utils.print_utils import *
from robolearn.utils.plot_utils import *
from robolearn.utils.traj_opt.traj_opt_utils import traj_distr_kl, traj_distr_kl_alt

from robolearn.policies.lin_gauss_policy import LinearGaussianPolicy
from robolearn.algos.gps.gps_utils import gauss_fit_joint_prior

from robolearn.envs.gym_environment import GymEnv
from robolearn.envs.pybullet.bullet_env import BulletEnv
import rospy
import time

import logging


class DualGPS(Algorithm):
    def __init__(self, agent, env, **kwargs):
        super(DualGPS, self).__init__(DEFAULT_GPS_HYPERPARAMS, kwargs)
        self.agent = agent
        self.env = env

        # Get dimensions from the environment
        self.dU = self._hyperparams['dU'] = env.get_action_dim()
        self.dX = self._hyperparams['dX'] = env.get_state_dim()
        self.dO = self._hyperparams['dO'] = env.get_obs_dim()

        # Get time values from 'hyperparams'
        self.T = self._hyperparams['T']
        self.dt = self._hyperparams['dt']

        # GPS algorithm
        self.gps_algo = 'mdgps_mdreps'
        # self.gps_algo = 'mdreps'

        # Specifies if it is MDGPS or only TrajOpt version
        self.use_global_policy = self._hyperparams['use_global_policy']

        # Number of initial conditions
        self.M = self._hyperparams['conditions']

        # Get/Define train and test conditions
        if 'train_conditions' in self._hyperparams \
                and self._hyperparams['train_conditions'] is not None:
            self._train_cond_idx = self._hyperparams['train_conditions']
            self._test_cond_idx = self._hyperparams['test_conditions']
        else:
            self._train_cond_idx = self._test_cond_idx = range(self.M)
            self._hyperparams['train_conditions'] = self._train_cond_idx
            self._hyperparams['test_conditions'] = self._test_cond_idx

        # Log and Data files
        if 'data_files_dir' in self._hyperparams:
            self._data_files_dir = self._hyperparams['data_files_dir']
        else:
            self._data_files_dir = 'GPS_' + \
                                   str(datetime.datetime.now().
                                       strftime("%Y-%m-%d_%H:%M:%S"))
        self.data_logger = DataLogger(self._data_files_dir)
        self.logger = self.setup_logger('log_dualgps', self._data_files_dir,
                                        '/log_dualgps.log', also_screen=True)

        # Get max number of iterations and define counter
        self.max_iterations = self._hyperparams['iterations']
        self.iteration_count = 0

        # Noise to be used with trajectory distributions
        self.noise_data = np.zeros((self.max_iterations, self.M,
                                    self._hyperparams['num_samples'],
                                    self.T, self.dU))
        if self._hyperparams['noisy_samples']:
            for ii in range(self.max_iterations):
                for cond in range(self.M):
                    for n in range(self._hyperparams['num_samples']):
                        self.noise_data[ii, cond, n, :, :] = self.get_noise()

        # IterationData objects for each condition.
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]

        # Trajectory Info
        # Add same dynamics for all the condition if the algorithm requires it
        if self._hyperparams['fit_dynamics']:
            dynamics = self._hyperparams['dynamics']

        # Initial trajectory hyperparams
        init_traj_distr = self._hyperparams['init_traj_distr']
        init_traj_distr['x0'] = env.get_conditions()
        init_traj_distr['dX'] = self.dX
        init_traj_distr['dU'] = self.dU
        init_traj_distr['dt'] = self.dt
        init_traj_distr['T'] = self.T

        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()

            if self._hyperparams['fit_dynamics']:
                self.cur[m].traj_info.dynamics = dynamics['type'](dynamics)

            # Get the initial trajectory distribution hyperparams
            init_traj_distr = extract_condition(self._hyperparams['init_traj_distr'], self._train_cond_idx[m])

            # Instantiate Trajectory Distribution: init_lqr or init_pd
            self.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr)

        # Last trajectory distribution optimized in C-step
        self.new_traj_distr = None

        # Traj Opt (Local policy opt) method #
        # ---------------------------------- #
        # Options: LQR, PI2, MDREPS
        self.traj_opt = self._hyperparams['traj_opt']['type'](self._hyperparams['traj_opt'])
        self.traj_opt.set_logger(self.logger)

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

        # Duality Data #
        # ------------ #
        # Duality data with: [sample_list, samples_cost, cs_traj, traj_dist, pol_info]
        self.good_duality_info = [DualityInfo() for _ in range(self.M)]
        self.bad_duality_info = [DualityInfo() for _ in range(self.M)]

        # MDGPS data #
        # ---------- #
        if self.use_global_policy:
            self._hyperparams['gps_algo_hyperparams']['T'] = self.T
            self._hyperparams['gps_algo_hyperparams']['dU'] = self.dU
            self._hyperparams['gps_algo_hyperparams']['dX'] = self.dX
            policy_prior = self._hyperparams['gps_algo_hyperparams']['policy_prior']
            for m in range(self.M):
                # Same policy prior type for all conditions
                self.cur[m].pol_info = PolicyInfo(self._hyperparams['gps_algo_hyperparams'])
                self.cur[m].pol_info.policy_prior = policy_prior['type'](policy_prior)

        # TrajectoryInfo for good and bad trajectories
        self.good_trajectories_info = [None for _ in range(self.M)]
        self.bad_trajectories_info = [None for _ in range(self.M)]
        self.base_kl_good = None
        self.base_kl_bad = None
        for m in range(self.M):
            self.good_trajectories_info[m] = TrajectoryInfo()
            self.bad_trajectories_info[m] = TrajectoryInfo()

            if self._hyperparams['fit_dynamics']:
                self.good_trajectories_info[m].dynamics = dynamics['type'](dynamics)
                self.bad_trajectories_info[m].dynamics = dynamics['type'](dynamics)

            # TODO: Use demonstration trajectories
            # # Get the initial trajectory distribution hyperparams
            # init_traj_distr = extract_condition(self._hyperparams['init_traj_distr'], self._train_cond_idx[m])
            # Instantiate Trajectory Distribution: init_lqr or init_pd
            # self.good_duality_infor = init_traj_distr['type'](init_traj_distr)
            # self.bad_duality_infor = init_traj_distr['type'](init_traj_distr)

            # TODO: Using same init traj
            # Get the initial trajectory distribution hyperparams
            init_traj_distr = extract_condition(self._hyperparams['init_traj_distr'], self._train_cond_idx[m])
            # Instantiate Trajectory Distribution: init_lqr or init_pd
            self.good_duality_info[m].traj_dist = init_traj_distr['type'](init_traj_distr)
            self.bad_duality_info[m].traj_dist = init_traj_distr['type'](init_traj_distr)

            # Set initial dual variables
            self.cur[m].eta = self._hyperparams['gps_algo_hyperparams']['init_eta']
            self.cur[m].nu = self._hyperparams['gps_algo_hyperparams']['init_nu']
            self.cur[m].omega = self._hyperparams['gps_algo_hyperparams']['init_omega']

        # Good/Bad bounds
        self.base_kl_good = self._hyperparams['gps_algo_hyperparams']['base_kl_good']
        self.base_kl_bad = self._hyperparams['gps_algo_hyperparams']['base_kl_bad']

        # MDGPS data
        if self.use_global_policy:
            for m in range(self.M):
                # Same policy prior in MDGPS for good/bad
                self.good_duality_info[m].pol_info = PolicyInfo(self._hyperparams['gps_algo_hyperparams'])
                self.good_duality_info[m].pol_info.policy_prior = policy_prior['type'](policy_prior)
                self.bad_duality_info[m].pol_info = PolicyInfo(self._hyperparams['gps_algo_hyperparams'])
                self.bad_duality_info[m].pol_info.policy_prior = policy_prior['type'](policy_prior)

    def run(self, itr_load=None):
        """
        Run GPS. If itr_load is specified, first loads the algorithm state from
        that iteration and resumes training at the next iteration.
        
        Args:
            itr_load: Desired iteration to load algorithm from

        Returns: True/False if all the gps algorithms have finished properly

        """
        run_successfully = True

        try:
            # Run GPS
            itr_start = self._initialize(itr_load)
            self._iteration(itr_start)

        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            print("#"*30)
            print("#"*30)
            print_skull()
            print("Panic: ERROR IN DualGPS ALGORITHM!!!!")
            print("#"*30)
            print("#"*30)
            run_successfully = False

        finally:
            # self._end()
            return run_successfully

    def _iteration(self, itr_start):
        logger = self.logger

        try:
            for itr in range(itr_start, self.max_iterations):
                # Sample from environment
                logger.info('')
                logger.info('->DualGPS: itr:%02d | '
                            'Sampling from local trajectories...' % (itr+1))
                traj_or_pol = 'traj'
                self._take_sample(traj_or_pol, itr)

                # Get samples from agent
                n_samples = self._hyperparams['num_samples']
                traj_sample_lists = [self.agent.get_samples(cond, -n_samples)
                                     for cond in self._train_cond_idx]
                # Clear agent sample
                self.agent.clear_samples()  # TODO: Check if it is better to 'remember' these samples

                for m, m_train in enumerate(self._train_cond_idx):
                    self.cur[m_train].sample_list = traj_sample_lists[m]

                # Update dynamics model using all samples.
                logger.info('')
                logger.info('->DualGPS: itr:%02d | '
                            'Updating dynamics linearization...' % (itr+1))
                self._update_dynamics()

                logger.info('')
                logger.info('->DualGPS: itr:%02d | '
                            'Evaluating samples costs...' % (itr+1))
                for m in range(self.M):
                    self._eval_cost(m)

                # On the first iteration, need to catch policy up to init_traj_distr.
                if self.use_global_policy and self.iteration_count == 0:
                    self.new_traj_distr = [self.cur[cond].traj_distr for cond in range(self.M)]
                    logger.info("\n"*2)
                    logger.info('->DualGPS: itr:%02d | '
                                'S-step for init_traj_distribution (iter=0)...'
                                % (itr+1))
                    self._update_policy()

                # Update global policy linearizations.
                logger.info('')
                if self.use_global_policy:
                    logger.info('->DualGPS: itr:%02d | '
                                'Updating global policy linearization...'
                                % (itr+1))
                    for m in range(self.M):
                        self._update_policy_fit(m)

                logger.info('')
                if self.use_global_policy and self.iteration_count > 0:
                    logger.info('->DualGPS: itr:%02d | '
                                'Updating KL step size with GLOBAL policy...'
                                % (itr+1))
                    self._update_step_size_global_policy()
                elif self.iteration_count > 0:
                    logger.info('->DualGPS: itr:%02d | '
                                'Updating KL step size with prev LOCAL policy.'
                                % (itr+1))
                    self._update_step_size()

                #logger.info('')
                #if self.use_global_policy and self.iteration_count > 0:
                #    logger.info('->DualGPS: itr:%02d | Updating KL step size with GLOBAL policy...' % itr+1)
                #    self._update_good_size_global_policy()
                #    self._update_bad_size_global_policy()
                #elif self.iteration_count > 0:
                #    logger.info('->DualGPS: itr:%02d | Updating KL step size with previous LOCAL policy...' % (itr+1))
                #    raise NotImplementedError
                #    #self._update_step_size()  # KL Divergence step size.

                logger.info('')
                logger.info('->DualGPS: itr:%02d | '
                            'Getting good and bad trajectories...' % (itr+1))
                self._get_good_trajs(option=self._hyperparams['gps_algo_hyperparams']
                                     ['good_traj_selection_type'])
                self._get_bad_trajs(option=self._hyperparams['gps_algo_hyperparams']
                                    ['bad_traj_selection_type'])

                logger.info('')
                logger.info('->DualGPS: itr:%02d | '
                            'Updating data of good and bad samples...' % (itr+1))
                logger.info('-->DualGPS: itr:%02d | '
                            'Update g/b dynamics...' % (itr+1))
                self._update_good_bad_dynamics(option=self._hyperparams['gps_algo_hyperparams']['duality_dynamics_type'])
                logger.info('-->DualGPS: itr:%02d | '
                            'Update g/b costs...' % (itr+1))
                self._eval_good_bad_costs()
                logger.info('-->DualGPS: itr:%02d | '
                            'Update g/b traj dist...' % (itr+1))
                self._fit_good_bad_traj_dist()
                logger.info('-->DualGPS: itr:%02d | '
                            'Divergence btw good/bad trajs: ...' % (itr+1))
                self._check_kl_div_good_bad()

                # Run inner loop to compute new policies.
                logger.info('')
                logger.info('->DualGPS: itr:%02d | '
                            'Updating trajectories...' % (itr+1))
                for ii in range(self._hyperparams['gps_algo_hyperparams']
                                ['inner_iterations']):
                    logger.info('-->DualGPS: itr:%02d | Inner iteration %d/%d'
                                % (itr+1, ii+1,
                                   self._hyperparams['gps_algo_hyperparams']
                                   ['inner_iterations']))
                    self._update_trajectories()

                if self.use_global_policy:
                    logger.info('')
                    logger.info('DualGPS:itr:%02d | ->| S-step |<-' % (itr+1))
                    self._update_policy()

                self.advance_duality_iteration_variables()

                # test_after_iter
                if self._hyperparams['test_after_iter']:
                    logger.info('')
                    logger.info('-->DualGPS: itr:%02d | '
                                'Testing global policy...' % (itr+1))
                    traj_or_pol = 'pol'
                    self._take_sample(traj_or_pol, itr)

                    pol_sample_lists = list()
                    for m in range(self.M):
                        pol_sample_lists.append(self.prev[m].pol_info.policy_samples)  # Because after advance_iter

                    pol_sample_lists_costs,\
                    pol_sample_lists_cost_compositions = \
                        self._eval_conditions_sample_list_cost(pol_sample_lists)

                    for m in range(self.M):
                        print('&'*10)
                        print('Average Cost')
                        print('Condition:%02d' % m)
                        print('Avg cost: %f'
                              % pol_sample_lists_costs[m].sum(axis=1).mean())
                        print('&'*10)

                else:
                    pol_sample_lists = None
                    pol_sample_lists_costs = None
                    pol_sample_lists_cost_compositions = None

                # Log data
                print('*'*10)
                print('*'*10)
                print('*'*10)
                print("NO LOGGING DATAAAAAAAAAAAAAAAAAAAAA!!!")
                print('*'*10)
                print('*'*10)
                # self._log_data(itr, traj_sample_lists, pol_sample_lists,
                #                pol_sample_lists_costs,
                #                pol_sample_lists_cost_compositions)

        except Exception as e:
            logger.exception("Error in GPS!!!")

    def _take_sample(self, traj_or_pol, itr):
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

        zero_noise = np.zeros(self.dU)

        print("Sampling with mode:'%s'" % traj_or_pol)
        verbose = False

        for cond in range(self.M):
            if traj_or_pol == 'traj':
                on_policy = self._hyperparams['sample_on_policy']
                total_samples = self._hyperparams['num_samples']
                save = True

            elif traj_or_pol == 'pol':
                on_policy = True
                total_samples = self._hyperparams['test_samples']
                save = False  # TODO: CHECK THIS
                pol_samples = [list() for _ in range(len(self._test_cond_idx))]
                itr -= 1  # Because it is called after self._advance_iteration_variables()
            else:
                raise ValueError("Wrong traj_or_pol option %s" % traj_or_pol)

            # On-policy or Off-policy
            if on_policy and (self.iteration_count > 0 or
                              ('sample_pol_first_itr' in self._hyperparams
                               and self._hyperparams['sample_pol_first_itr'])):
                policy = self.agent.policy  # DOM: Instead self.opt_pol.policy
                print("On-policy sampling: %s!" % type(policy).__name__)
            else:
                policy = self.cur[cond].traj_distr
                print("Off-policy sampling: %s!" % type(policy).__name__)

            for i in range(total_samples):
                # Create a sample class
                sample = Sample(self.env, self.T)
                history = [None] * self.T
                obs_hist = [None] * self.T

                # sample = self._take_fake_sample(itr, cond, i, verbose=True,
                #                                 save=True, noisy=True,
                #                                 on_policy=False)

                self.logger.info("Resetting environment to condition %02d..."
                                 % cond)
                # self.env.reset(time=2, cond=cond)  # For bigman
                self.env.reset(condition=cond)

                # if issubclass(type(self.env), GymEnv):
                if issubclass(type(self.env), BulletEnv):
                    gym_ts = self.dt
                else:
                    ros_rate = rospy.Rate(int(1/self.dt))  # hz

                # Collect history
                if on_policy:
                    sample_text = "On-policy sampling | itr:%d/%d, cond:%d/%d, " \
                                  "i:%d/%d" % (itr+1, self.max_iterations,
                                               cond+1, len(self._train_cond_idx),
                                               i+1, total_samples)
                else:
                    sample_text = "Sampling | itr:%d/%d, cond:%d/%d, s:%d/%d"\
                                  % (itr+1, self.max_iterations,
                                     cond+1, len(self._train_cond_idx),
                                     i+1, total_samples)
                if verbose:
                    print(sample_text)

                sampling_bar = ProgressBar(self.T, bar_title=sample_text,
                                           total_lines=20)

                for t in range(self.T):
                    sampling_bar.update(t)
                    if verbose:
                        if on_policy:
                            print("On-policy sample | "
                                  "itr:%d/%d, cond:%d/%d, i:%d/%d | t:%d/%d"
                                  % (itr+1, self.max_iterations,
                                     cond+1, len(self._train_cond_idx),
                                     i+1, total_samples,
                                     t+1, self.T))
                        else:
                            print("Sample | "
                                  "itr:%d/%d, cond:%d/%d, s:%d/%d | t:%d/%d"
                                  % (itr+1, self.max_iterations,
                                     cond+1, len(self._train_cond_idx),
                                     i+1, total_samples,
                                     t+1, self.T))
                    obs = self.env.get_observation()
                    # print(obs)
                    # input('pruebame')
                    # Checking NAN
                    nan_number = np.isnan(obs)
                    if np.any(nan_number):
                        print("\e[31mERROR OBSERVATION: NAN!!!!! | "
                              "type:%s | itr %d | cond%d | i:%d t:%d"
                              % (traj_or_pol, itr+1, cond+1, i+1, t+1))
                    state = self.env.get_state()
                    if traj_or_pol == 'traj':
                        noise = self.noise_data[itr, cond, i, t, :]
                    else:
                        noise = zero_noise
                    # TODO: Avoid TF policy writes in obs
                    action = policy.eval(state.copy(), obs.copy(), t,
                                         noise.copy())
                    # Checking NAN
                    nan_number = np.isnan(action)
                    if np.any(nan_number):
                        print("\e[31mERROR ACTION: NAN!!!!! | type:%s | "
                              "itr %d | cond%d | i:%d t:%d"
                              % (traj_or_pol, itr+1, cond+1, i+1, t+1))
                    action[nan_number] = 0
                    # self.env.send_action(action)
                    self.env.step(action)
                    obs_hist[t] = (obs, action)
                    history[t] = (state, action)
                    # if issubclass(type(self.env), GymEnv):
                    if issubclass(type(self.env), BulletEnv):
                        time.sleep(gym_ts)
                    else:
                        ros_rate.sleep()

                sampling_bar.end()

                # Stop environment
                self.env.stop()

                self.logger.info("Generating sample data...")

                all_actions = np.array([hist[1] for hist in history])
                all_states = np.array([hist[0] for hist in history])
                all_obs = np.array([hist[0] for hist in obs_hist])
                sample.set_acts(all_actions)   # Set all actions at the same time
                sample.set_obs(all_obs)        # Set all obs at the same time
                sample.set_states(all_states)  # Set all states at the same time
                sample.set_noise(self.noise_data[itr])        # Set all noise at the same time

                if save:  # Save sample in agent sample list
                    sample_id = self.agent.add_sample(sample, cond)
                    self.logger.info("The sample was added to Agent's sample "
                                     "list. Now there are %d sample(s) for "
                                     "condition '%d'." % (sample_id+1, cond))

                if traj_or_pol == 'pol':
                    pol_samples[cond].append(sample)

            if traj_or_pol == 'pol':
                self.prev[cond].pol_info.policy_samples = \
                    SampleList(pol_samples[cond])  # prev because it is called after advance_iteration_variables

    def _check_kl_div_good_bad(self):
        for cond in range(self.M):
            good_distr = self.good_duality_info[cond].traj_dist
            bad_distr = self.bad_duality_info[cond].traj_dist
            mu_good, sigma_good = lqr_forward(good_distr, self.good_trajectories_info[cond])
            mu_bad, sigma_bad = lqr_forward(bad_distr, self.bad_trajectories_info[cond])
            kl_div_good_bad = traj_distr_kl_alt(mu_good, sigma_good, good_distr, bad_distr, tot=True)
            #print("G/B KL_div: %f " % kl_div_good_bad)
            self.logger.info('--->Divergence btw good/bad trajs is: %f'
                             % kl_div_good_bad)

    def get_noise(self):
        return generate_noise(self.T, self.dU, self._hyperparams)

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

    def compute_costs(self, m, eta, omega, nu, augment=True):
        """
        Compute cost estimates used in the LQR backward pass.
        
        Args:
            m: Condition
            eta: Dual variable corresponding to KL divergence with previous policy.
            omega: Dual variable(s) corresponding to KL divergence with good trajectories.
            nu: Dual variable(s) corresponding to KL divergence with bad trajectories.
            augment: True if we want a KL constraint for all time-steps. False otherwise.

        Returns: Cm and cv

        """
        traj_info = self.cur[m].traj_info
        traj_distr = self.cur[m].traj_distr
        good_distr = self.good_duality_info[m].traj_dist
        bad_distr = self.bad_duality_info[m].traj_dist
        if not augment:  # Whether to augment cost with term to penalize KL
            return traj_info.Cm, traj_info.cv

        multiplier = self._hyperparams['max_ent_traj']  # Weight of maximum entropy term in trajectory optimization

        # TVLGC terms from previous traj
        K, ipc, k = traj_distr.K, traj_distr.inv_pol_covar, traj_distr.k
        # TVLGC terms from good traj
        K_good, ipc_good, k_good = good_distr.K, good_distr.inv_pol_covar, good_distr.k
        # TVLGC terms from bad traj
        K_bad, ipc_bad, k_bad = bad_distr.K, bad_distr.inv_pol_covar, bad_distr.k

        # print("TODO: SETTING DUMMY IPC GOOD AND BAD")
        # for t in range(self.T):
        #     ipc_good[t, :, :] = np.eye(7)
        #     ipc_bad[t, :, :] = np.eye(7)

        # omega = 0
        # nu = 0

        # Surrogate cost
        fCm = traj_info.Cm / (eta + omega - nu + multiplier)
        fcv = traj_info.cv / (eta + omega - nu + multiplier)

        # idx_u = slice(self.dX, self.dX+self.dU)
        # print("")
        # print(fCm[-1, idx_u, idx_u])
        # print("Original fCm/eta[%d] PD?: %s" % (-1, np.all(np.linalg.eigvals(fCm[-1, idx_u, idx_u]) > 0)))

        # Add in the KL divergence with previous policy.
        for t in range(self.T - 1, -1, -1):
            fCm[t, :, :] += eta / (eta + omega - nu + multiplier) * np.vstack([
                np.hstack([  # dX x (dX + dU)
                    K[t, :, :].T.dot(ipc[t, :, :]).dot(K[t, :, :]),
                    -K[t, :, :].T.dot(ipc[t, :, :])
                ]),
                np.hstack([  # dU x (dX + dU)
                    -ipc[t, :, :].dot(K[t, :, :]), ipc[t, :, :]
                ])
            ])
            fcv[t, :] += eta / (eta + omega - nu + multiplier) * np.hstack([
                K[t, :, :].T.dot(ipc[t, :, :]).dot(k[t, :]),  # dX
                -ipc[t, :, :].dot(k[t, :])  # dU
            ])

        # print("")
        # print(ipc[-1, :, :])
        # print("-")
        # print(fCm[-1, idx_u, idx_u])
        # print("Surrogate fCm/eta[%d] PD?: %s" % (-1, np.all(np.linalg.eigvals(fCm[-1, idx_u, idx_u]) > 0)))
        # raw_input("avercomo")


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
            fcv[t, :] += omega / (eta + omega - nu + multiplier) * np.hstack([
                K_good[t, :, :].T.dot(ipc_good[t, :, :]).dot(k_good[t, :]),
                -ipc_good[t, :, :].dot(k_good[t, :])
            ])

        # print("")
        # print(ipc_good[-1, :, :])
        # print("-")
        # print(fCm[-1, idx_u, idx_u])
        # print("Surrogate fCm/omega[%d] PD?: %s" % (-1, np.all(np.linalg.eigvals(fCm[-1, idx_u, idx_u]) > 0)))
        # raw_input("avercomo")

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
            fcv[t, :] -= nu / (eta + omega - nu + multiplier) * np.hstack([
                K_bad[t, :, :].T.dot(ipc_bad[t, :, :]).dot(k_bad[t, :]),
                -ipc_bad[t, :, :].dot(k_bad[t, :])
            ])

        # print("")
        # print(ipc_bad[-1, :, :])
        # print("-")
        # print(fCm[-1, idx_u, idx_u])
        # print("Surrogate fCm/nu[%d] PD?: %s" % (-1, np.all(np.linalg.eigvals(fCm[-1, idx_u, idx_u]) > 0)))
        # raw_input("avercomo")

        return fCm, fcv

    def advance_duality_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', reinitialize 'cur' variables, and advance iteration counter.
        :return: None
        """
        self._advance_iteration_variables()
        for m in range(self.M):
            self.cur[m].nu = self.prev[m].nu
            self.cur[m].omega = self.prev[m].omega

            if self.use_global_policy:
                self.cur[m].traj_info.last_kl_step = self.prev[m].traj_info.last_kl_step
                self.cur[m].pol_info = copy.deepcopy(self.prev[m].pol_info)

    def _fit_good_bad_traj_dist(self):
        min_good_var = self._hyperparams['gps_algo_hyperparams']['min_good_var']
        min_bad_var = self._hyperparams['gps_algo_hyperparams']['min_bad_var']

        for cond in range(self.M):
            self.good_duality_info[cond].traj_dist = \
                self.fit_traj_dist(self.good_duality_info[cond].sample_list,
                                   min_good_var)
            self.bad_duality_info[cond].traj_dist = \
                self.fit_traj_dist(self.bad_duality_info[cond].sample_list,
                                   min_bad_var)

    def _eval_good_bad_costs(self):
        """
        Evaluate costs for all current samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        for cond in range(self.M):
            cs, cc, cv, Cm = self._eval_dual_costs(cond, self.good_duality_info[cond].sample_list)
            self.good_duality_info[cond].traj_cost = cs
            self.good_trajectories_info[cond].cc = cc
            self.good_trajectories_info[cond].cv = cv
            self.good_trajectories_info[cond].Cm = Cm

            cs, cc, cv, Cm = self._eval_dual_costs(cond, self.bad_duality_info[cond].sample_list)
            self.bad_duality_info[cond].traj_cost = cs
            self.bad_trajectories_info[cond].cc = cc
            self.bad_trajectories_info[cond].cv = cv
            self.bad_trajectories_info[cond].Cm = Cm

    def _eval_dual_costs(self, cond, dual_sample_list):
        # Constants.
        T, dX, dU = self.T, self.dX, self.dU
        N = len(dual_sample_list)

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX+dU))
        Cm = np.zeros((N, T, dX+dU, dX+dU))
        for n in range(N):
            sample = dual_sample_list[n]
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

        return cs, cc, cv, Cm

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
            self.good_trajectories_info[m].dynamics.update_prior(good_data)
            self.good_trajectories_info[m].dynamics.fit(X_good, U_good)
            self.bad_trajectories_info[m].dynamics.update_prior(bad_data)
            self.bad_trajectories_info[m].dynamics.fit(X_bad, U_bad)

            # Fit x0mu/x0sigma.
            x0_good = X_good[:, 0, :]
            x0mu_good = np.mean(x0_good, axis=0)  # TODO: SAME X0 FOR ALL??
            self.good_trajectories_info[m].x0mu = x0mu_good
            self.good_trajectories_info[m].x0sigma = np.diag(np.maximum(np.var(x0_good, axis=0),
                                                                        self._hyperparams['initial_state_var']))
            x0_bad = X_bad[:, 0, :]
            x0mu_bad = np.mean(x0_bad, axis=0)  # TODO: SAME X0 FOR ALL??
            self.bad_trajectories_info[m].x0mu = x0mu_bad
            self.bad_trajectories_info[m].x0sigma = np.diag(np.maximum(np.var(x0_bad, axis=0),
                                                                       self._hyperparams['initial_state_var']))

            prior_good = self.good_trajectories_info[m].dynamics.get_prior()
            if prior_good:
                mu0, Phi, priorm, n0 = prior_good.initial_state()
                N = len(good_data)
                self.good_trajectories_info[m].x0sigma += Phi + (N*priorm) / (N+priorm) * np.outer(x0mu_good-mu0, x0mu_good-mu0) / (N+n0)

            prior_bad = self.good_trajectories_info[m].dynamics.get_prior()
            if prior_bad:
                mu0, Phi, priorm, n0 = prior_bad.initial_state()
                N = len(bad_data)
                self.bad_trajectories_info[m].x0sigma += Phi + (N*priorm) / (N+priorm) * np.outer(x0mu_bad-mu0, x0mu_bad-mu0) / (N+n0)

    def _get_bad_trajs(self, option='only_traj'):
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
            n_bad = self._hyperparams['gps_algo_hyperparams']['n_bad_samples']
            if n_bad == cs.shape[0]:
                worst_indeces = range(n_bad)
            else:
                worst_indeces = np.argpartition(np.sum(cs, axis=1), -n_bad)[-n_bad:]

            # Get current best trajectory
            if self.bad_duality_info[cond].sample_list is None:
                for bb, bad_index in enumerate(worst_indeces):
                    LOGGER.info("DualGPS: Defining BAD trajectory sample %d | cur_cost=%f from sample %d" % (bb,
                                                                                          np.sum(cs[bad_index, :]), bad_index))
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
                    raise ValueError("DualGPS: Wrong get_bad_grajectories option: %s" % option)

    def _get_good_trajs(self, option='only_traj'):
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
            n_good = self._hyperparams['gps_algo_hyperparams']['n_good_samples']
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
                    raise ValueError("DualGPS: Wrong get_good_grajectories option: %s" % option)

    def _update_step_size(self, number_policy):
        """ Evaluate costs on samples, and adjust the step size. """
        # Evaluate cost function for all conditions and samples.
        #for m in range(self.M):
        #    self._eval_cost(m)

        # Adjust step size relative to the previous iteration.
        for m in range(self.M):
            if self.iteration_count[number_policy] >= 1 and self.prev[number_policy][m].sample_list:
                self._stepadjust(number_policy, m)

    def _stepadjust(self, m):
        """
        Calculate new step sizes.
        Args:
            m: Condition
        """
        LOGGER = self.logger

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

        LOGGER.debug('DualGPS: Trajectory step: ent: %f cost: %f -> %f', ent, previous_mc_obj, new_mc_obj)

        # Compute predicted and actual improvement.
        predicted_impr = np.sum(previous_laplace_obj) - \
                         np.sum(new_predicted_laplace_obj)
        actual_impr = np.sum(previous_laplace_obj) - \
                      np.sum(new_actual_laplace_obj)

        # Print improvement details.
        LOGGER.debug('DualGPS: Previous cost: Laplace: %f MC: %f',
                     np.sum(previous_laplace_obj), previous_mc_obj)
        LOGGER.debug('DualGPS: Predicted new cost: Laplace: %f MC: %f',
                     np.sum(new_predicted_laplace_obj), new_mc_obj)
        LOGGER.debug('DualGPS: Actual new cost: Laplace: %f MC: %f',
                     np.sum(new_actual_laplace_obj), new_mc_obj)
        LOGGER.debug('DualGPS: Predicted/actual improvement: %f / %f',
                     predicted_impr, actual_impr)

        self._set_new_mult(predicted_impr, actual_impr, m)

    def _update_step_size_global_policy(self):
        """
        Calculate new step sizes. This version uses the same step size for all conditions.
        """
        LOGGER = self.logger


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
        if self._hyperparams['gps_algo_hyperparams']['step_rule'] == 'laplace':
            predicted_impr = prev_laplace - prev_predicted
            actual_impr = prev_laplace - cur_laplace
        elif self._hyperparams['gps_algo_hyperparams']['step_rule'] == 'mc':
            predicted_impr = prev_mc - prev_predicted
            actual_impr = prev_mc - cur_mc
        LOGGER.debug('DualGPS: Previous cost: Laplace: %f, MC: %f', prev_laplace, prev_mc)
        LOGGER.debug('DualGPS: Predicted cost: Laplace: %f', prev_predicted)
        LOGGER.debug('DualGPS: Actual cost: Laplace: %f, MC: %f', cur_laplace, cur_mc)

        for m in range(self.M):
            self._set_new_mult(predicted_impr, actual_impr, m)

    def _update_good_size_global_policy(self):
        """
        Calculate new step sizes. This version uses the same step size for all conditions.
        """
        LOGGER = self.logger


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
        if self._hyperparams['gps_algo_hyperparams']['step_rule'] == 'laplace':
            predicted_impr = prev_laplace - prev_predicted
            actual_impr = prev_laplace - cur_laplace
        elif self._hyperparams['gps_algo_hyperparams']['step_rule'] == 'mc':
            predicted_impr = prev_mc - prev_predicted
            actual_impr = prev_mc - cur_mc
        LOGGER.debug('DualGPS: Previous cost: Laplace: %f, MC: %f', prev_laplace, prev_mc)
        LOGGER.debug('DualGPS: Predicted cost: Laplace: %f', prev_predicted)
        LOGGER.debug('DualGPS: Actual cost: Laplace: %f, MC: %f', cur_laplace, cur_mc)

        for m in range(self.M):
            self._set_new_mult(predicted_impr, actual_impr, m)

    def _update_bad_size_global_policy(self):
        """
        Calculate new step sizes. This version uses the same step size for all conditions.
        """
        LOGGER = self.logger


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
        if self._hyperparams['gps_algo_hyperparams']['step_rule'] == 'laplace':
            predicted_impr = prev_laplace - prev_predicted
            actual_impr = prev_laplace - cur_laplace
        elif self._hyperparams['gps_algo_hyperparams']['step_rule'] == 'mc':
            predicted_impr = prev_mc - prev_predicted
            actual_impr = prev_mc - cur_mc
        LOGGER.debug('DualGPS: Previous cost: Laplace: %f, MC: %f', prev_laplace, prev_mc)
        LOGGER.debug('DualGPS: Predicted cost: Laplace: %f', prev_predicted)
        LOGGER.debug('DualGPS: Actual cost: Laplace: %f, MC: %f', cur_laplace, cur_mc)

        for m in range(self.M):
            self._set_new_mult(predicted_impr, actual_impr, m)

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
        mode = self._hyperparams['gps_algo_hyperparams']['policy_sample_mode']
        policy_prior.update(samples, self.policy_opt, mode)

        # Fit linearization and store in pol_info.
        pol_info.pol_K, pol_info.pol_k, pol_info.pol_S = policy_prior.fit(X, pol_mu, pol_sig)
        for t in range(T):
            pol_info.chol_pol_S[t, :, :] = sp.linalg.cholesky(pol_info.pol_S[t, :, :])

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

        return sample

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
        LOGGER = self.logger
        LOGGER.info("Logging Agent... ")
        self.data_logger.pickle(
            ('dualgps_itr_%02d.pkl' % itr),
            # copy.copy(temp_dict)
            copy.copy(self.agent)
        )
        if self.use_global_policy:
            LOGGER.info("Logging Policy_Opt... ")
            self.data_logger.pickle(
                ('dualgps_policy_opt_itr_%02d.pkl' % itr),
                self.agent.policy_opt
            )
            print("TODO: NOT LOGGING POLICY!!!")
            #LOGGER.info("Logging Policy... ")
            #self.agent.policy_opt.policy.pickle_policy(self.dO, self.dU,
            #                                           self.data_logger.dir_path + '/' + ('dualgps_policy_itr_%02d' % itr),
            #                                           goal_state=None,
            #                                           should_hash=False)

        print("TODO: CHECK HOW TO SOLVE LOGGING DUAL ALGO")
        # print("Logging GPS algorithm state... ")
        # self.data_logger.pickle(
        #     ('%s_algorithm_itr_%02d.pkl' % (self.gps_algo.upper(), itr)),
        #     copy.copy(self)
        # )

        LOGGER.info("Logging GPS iteration data... ")
        self.data_logger.pickle(
            ('dualgps_%s_iteration_data_itr_%02d.pkl' % (self.gps_algo.upper(), itr)),
            copy.copy(self.prev)  # prev instead of cur
        )

        LOGGER.info("Logging Trajectory samples... ")
        self.data_logger.pickle(
            ('dualgps_traj_sample_itr_%02d.pkl' % itr),
            copy.copy(traj_sample_lists)
        )

        if pol_sample_lists is not None:
            LOGGER.info("Logging Global Policy samples... ")
            self.data_logger.pickle(
                ('dualgps_pol_sample_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists)
            )

        if pol_sample_lists_costs is not None:
            LOGGER.info("Logging Global Policy samples costs... ")
            self.data_logger.pickle(
                ('dualgps_pol_sample_cost_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists_costs)
            )

        if pol_sample_lists_cost_compositions is not None:
            LOGGER.info("Logging Global Policy samples cost compositions... ")
            self.data_logger.pickle(
                ('dualgps_pol_sample_cost_composition_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists_cost_compositions)
            )

        LOGGER.info("Logging God/Bad duality data")
        self.data_logger.pickle(
            ('dualgps_good_trajectories_info_itr_%02d.pkl' % itr),
            copy.copy(self.good_trajectories_info)
        )
        self.data_logger.pickle(
            ('dualgps_bad_trajectories_info_itr_%02d.pkl' % itr),
            copy.copy(self.bad_trajectories_info)
        )
        self.data_logger.pickle(
            ('dualgps_good_duality_info_itr_%02d.pkl' % itr),
            copy.copy(self.good_duality_info)
        )
        self.data_logger.pickle(
            ('dualgps_bad_duality_info_itr_%02d.pkl' % itr),
            copy.copy(self.bad_duality_info)
        )

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
            self.cur[m].traj_info.x0sigma = \
                np.diag(np.maximum(np.var(x0, axis=0),
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
        LOGGER = self.logger

        LOGGER.info('-->DualGPS: Updating trajectories (local policies)...')
        if self.new_traj_distr is None:
            self.new_traj_distr = [self.cur[cond].traj_distr for cond in range(self.M)]
        for cond in range(self.M):
            traj_opt_outputs = self.traj_opt.update(cond, self)
            self.new_traj_distr[cond] = traj_opt_outputs[0]
            self.cur[cond].eta = traj_opt_outputs[1]
            self.cur[cond].omega = traj_opt_outputs[2]
            self.cur[cond].nu = traj_opt_outputs[3]

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
        LOGGER = self.logger

        # Model improvement as I = predicted_dI * KL + penalty * KL^2,
        # where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2).
        # Optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL =>
        # KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL.
        # Therefore, the new multiplier is given by pred/2*(pred-act).
        new_mult = predicted_impr / (2.0 * max(1e-4, predicted_impr - actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(min(new_mult * self.cur[m].step_mult, self._hyperparams['max_step_mult']),
                       self._hyperparams['min_step_mult'])
        self.cur[m].step_mult = new_step

        if new_mult > 1:
            LOGGER.debug('DualGPS: Increasing step size multiplier to %f', new_step)
        else:
            LOGGER.debug('DualGPS: Decreasing step size multiplier to %f', new_step)

    def _set_new_good_mult(self, predicted_impr, actual_impr, m):
        """
        Adjust good multiplier according to the predicted versus actual improvement.
        """
        LOGGER = self.logger

        # Model improvement as I = predicted_dI * KL + penalty * KL^2,
        # where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2).
        # Optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL =>
        # KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL.
        # Therefore, the new multiplier is given by pred/2*(pred-act).
        new_mult = predicted_impr / (2.0 * max(1e-4, predicted_impr - actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(min(new_mult * self.cur[m].step_mult, self._hyperparams['max_step_mult']),
                       self._hyperparams['min_step_mult'])
        self.cur[m].step_mult = new_step

        if new_mult > 1:
            LOGGER.debug('DualGPS: Increasing good multiplier to %f', new_good)
        else:
            LOGGER.debug('DualGPS: Decreasing good multiplier to %f', new_good)

    def _measure_ent(self, m):
        """ Measure the entropy of the current trajectory. """
        ent = 0
        for t in range(self.T):
            ent = ent + np.sum(
                np.log(np.diag(self.cur[m].traj_distr.chol_pol_covar[t, :, :]))
            )
        return ent


    @staticmethod
    def fit_traj_dist(sample_list, min_variance):
        samples = sample_list

        X = samples.get_states()
        obs = samples.get_obs()
        U = samples.get_actions()

        N, T, dX = X.shape
        dU = U.shape[2]
        if N == 1:
            raise ValueError("Cannot fit traj_dist on 1 sample")

        # import matplotlib.pyplot as plt
        # plt.plot(U[1, :])
        # plt.show(block=False)
        # raw_input(U.shape)

        pol_mu = U
        pol_sig = np.zeros((N, T, dU, dU))

        print("TODO: WE ARE GIVING MIN GOOD/BAD VARIANCE")
        for t in range(T):
            # Using only diagonal covariances
            # pol_sig[:, t, :, :] = np.tile(np.diag(np.diag(np.cov(U[:, t, :].T))), (N, 1, 1))
            current_diag = np.diag(np.cov(U[:, t, :].T))
            new_diag = np.max(np.vstack((current_diag, min_variance)), axis=0)
            pol_sig[:, t, :, :] = np.tile(np.diag(new_diag), (N, 1, 1))

        # Collapse policy covariances. (This is only correct because the policy doesn't depend on state).
        pol_sig = np.mean(pol_sig, axis=0)

        # Allocate.
        pol_K = np.zeros([T, dU, dX])
        pol_k = np.zeros([T, dU])
        pol_S = np.zeros([T, dU, dU])
        chol_pol_S = np.zeros([T, dU, dU])
        inv_pol_S = np.zeros([T, dU, dU])

        # Update policy prior.
        def eval_prior(Ts, Ps):
            strength = 1e-4
            dX, dU = Ts.shape[-1], Ps.shape[-1]
            prior_fd = np.zeros((dU, dX))
            prior_cond = 1e-5 * np.eye(dU)
            sig = np.eye(dX)
            Phi = strength * np.vstack([np.hstack([sig, sig.dot(prior_fd.T)]),
                                        np.hstack([prior_fd.dot(sig), prior_fd.dot(sig).dot(prior_fd.T) + prior_cond])])
            return np.zeros(dX+dU), Phi, 0, strength


        # Fit linearization with least squares regression
        dwts = (1.0 / N) * np.ones(N)
        for t in range(T):
            Ts = X[:, t, :]
            Ps = pol_mu[:, t, :]
            Ys = np.concatenate([Ts, Ps], axis=1)
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = eval_prior(Ts, Ps)
            sig_reg = np.zeros((dX+dU, dX+dU))
            # Slightly regularize on first timestep.
            if t == 0:
                sig_reg[:dX, :dX] = 1e-8
            pol_K[t, :, :], pol_k[t, :], pol_S[t, :, :] = gauss_fit_joint_prior(Ys, mu0, Phi, mm, n0, dwts, dX, dU, sig_reg)
        pol_S += pol_sig  # Add policy covariances mean

        # plt.subplots()
        # plt.plot(pol_S[:, 0, 0], label='0')
        # plt.plot(pol_S[:, 1, 1], label='1')
        # plt.plot(pol_S[:, 2, 2], label='2')
        # plt.plot(pol_S[:, 3, 3], label='3')
        # plt.plot(pol_S[:, 4, 4], label='4')
        # plt.plot(pol_S[:, 5, 5], label='5')
        # plt.legend()
        # plt.show(block=False)
        # raw_input('ploteandooo')

        for t in range(T):
            chol_pol_S[t, :, :] = sp.linalg.cholesky(pol_S[t, :, :])
            inv_pol_S[t, :, :] = np.linalg.inv(pol_S[t, :, :])

        # plt.subplots()
        # plt.plot(inv_pol_S[:, 0, 0], label='0')
        # plt.plot(inv_pol_S[:, 1, 1], label='1')
        # plt.plot(inv_pol_S[:, 2, 2], label='2')
        # plt.plot(inv_pol_S[:, 3, 3], label='3')
        # plt.plot(inv_pol_S[:, 4, 4], label='4')
        # plt.plot(inv_pol_S[:, 5, 5], label='5')
        # plt.legend()
        # plt.show(block=False)
        # raw_input('ploteandooo')

        # max_inv_pol = np.zeros(7)
        # max_inv_pol[0] = np.argmax(inv_pol_S[:, 0, 0])
        # max_inv_pol[1] = np.argmax(inv_pol_S[:, 1, 1])
        # max_inv_pol[2] = np.argmax(inv_pol_S[:, 2, 2])
        # max_inv_pol[3] = np.argmax(inv_pol_S[:, 3, 3])
        # max_inv_pol[4] = np.argmax(inv_pol_S[:, 4, 4])
        # max_inv_pol[5] = np.argmax(inv_pol_S[:, 5, 5])
        # max_inv_pol[6] = np.argmax(inv_pol_S[:, 6, 6])

        # print("%f, %d" % (np.max(inv_pol_S[:, 0, 0]), np.argmax(inv_pol_S[:, 0, 0])))
        # print("%f, %d" % (np.max(inv_pol_S[:, 1, 1]), np.argmax(inv_pol_S[:, 1, 1])))
        # print("%f, %d" % (np.max(inv_pol_S[:, 2, 2]), np.argmax(inv_pol_S[:, 2, 2])))
        # print("%f, %d" % (np.max(inv_pol_S[:, 3, 3]), np.argmax(inv_pol_S[:, 3, 3])))
        # print("%f, %d" % (np.max(inv_pol_S[:, 4, 4]), np.argmax(inv_pol_S[:, 4, 4])))
        # print("%f, %d" % (np.max(inv_pol_S[:, 5, 5]), np.argmax(inv_pol_S[:, 5, 5])))
        # print("%f, %d" % (np.max(inv_pol_S[:, 6, 6]), np.argmax(inv_pol_S[:, 6, 6])))

        # print("COVARIANCES:")
        # print("%f, %d" % (pol_S[int(max_inv_pol[0]), 0, 0], max_inv_pol[0]))
        # print("%f, %d" % (pol_S[int(max_inv_pol[1]), 1, 1], max_inv_pol[1]))
        # print("%f, %d" % (pol_S[int(max_inv_pol[2]), 2, 2], max_inv_pol[2]))
        # print("%f, %d" % (pol_S[int(max_inv_pol[3]), 3, 3], max_inv_pol[3]))
        # print("%f, %d" % (pol_S[int(max_inv_pol[4]), 4, 4], max_inv_pol[4]))
        # print("%f, %d" % (pol_S[int(max_inv_pol[5]), 5, 5], max_inv_pol[5]))
        # print("%f, %d" % (pol_S[int(max_inv_pol[6]), 6, 6], max_inv_pol[6]))

        # raw_input("AAAA")

        return LinearGaussianPolicy(pol_K, pol_k, pol_S, chol_pol_S, inv_pol_S)

    @staticmethod
    def setup_logger(logger_name, dir_path, log_file, level=logging.INFO,
                     also_screen=False):

        logger = logging.getLogger(logger_name)

        formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                      "%H:%M:%S")

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        fileHandler = logging.FileHandler(dir_path+log_file, mode='w')
        fileHandler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(fileHandler)

        # if also_screen:
        #     streamHandler = logging.StreamHandler()
        #     streamHandler.setFormatter(formatter)
        #     l.addHandler(streamHandler)

        return logger

        # For pickling.
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'agents' in state:
            state.pop('agents')
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