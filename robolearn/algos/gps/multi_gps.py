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
from Queue import Queue
from threading import Thread

from robolearn.algos.algorithm import Algorithm

from robolearn.algos.gps.gps_config import *
from robolearn.algos.gps.gps_utils import PolicyInfo
from robolearn.algos.gps.gps_utils import IterationData, TrajectoryInfo, extract_condition, DualityInfo

from robolearn.utils.sample import Sample
from robolearn.utils.sample_list import SampleList

from robolearn.agents.agent_utils import generate_noise
from robolearn.utils.data_logger import DataLogger
from robolearn.utils.print_utils import *
from robolearn.utils.plot_utils import *
from robolearn.utils.traj_opt.traj_opt_utils import traj_distr_kl, traj_distr_kl_alt

from robolearn.policies.lin_gauss_policy import LinearGaussianPolicy
from robolearn.algos.gps.gps_utils import gauss_fit_joint_prior

from robolearn.envs.gym_environment import GymEnv
import rospy
import time

import logging
# LOGGER = logging.getLogger(__name__)
# # Logging into console AND file
# LOGGER.setLevel(logging.DEBUG)
# ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.DEBUG)
# LOGGER.addHandler(ch)


class MULTIGPS(Algorithm):
    def __init__(self, agents, env, **kwargs):
        super(MULTIGPS, self).__init__(default_gps_hyperparams, kwargs)
        self.agents = agents
        self.env = env

        # Get dimensions from the environment
        self.dU = self._hyperparams['dU'] = env.get_action_dim()
        self.dX = self._hyperparams['dX'] = env.get_state_dim()
        self.dO = self._hyperparams['dO'] = env.get_obs_dim()

        # Get time values from 'hyperparams'
        self.T = self._hyperparams['T']
        self.dt = self._hyperparams['dt']

        # Get total GPS algos
        self.n_gps = len(self.agents)
        self.gps_algo = ['mdgps_mdreps' for _ in range(self.n_gps)]

        # Specifies if it is MDGPS or only TrajOpt version
        self.use_global_policy = self._hyperparams['use_global_policy']

        # Number of initial conditions
        self.M = self._hyperparams['conditions']

        # Get/Define train and test conditions
        if 'train_conditions' in self._hyperparams and self._hyperparams['train_conditions'] is not None:
            self._train_cond_idx = self._hyperparams['train_conditions']
            self._test_cond_idx = self._hyperparams['test_conditions']
        else:
            self._train_cond_idx = self._test_cond_idx = range(self.M)
            self._hyperparams['train_conditions'] = self._train_cond_idx
            self._hyperparams['test_conditions'] = self._test_cond_idx

        # Log and Data files
        if 'data_files_dir' in self._hyperparams:
            if self._hyperparams['data_files_dir'] is None:
                self._data_files_dir = 'robolearn_log/' + \
                                       'GPS_'+str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
            else:
                self._data_files_dir = 'robolearn_log/' + self._hyperparams['data_files_dir']
        else:
            self._data_files_dir = 'GPS_'+str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        self.data_logger = DataLogger(self._data_files_dir)
        for gps in range(self.n_gps):
            self.setup_logger('log%d' % gps, self._data_files_dir, '/log%d.log' % gps, also_screen=False)

        # Get max number of iterations
        self.max_iterations = self._hyperparams['iterations']

        # Define a iteration counter for each GPS algo
        self.iteration_count = [0 for _ in range(self.n_gps)]

        # Noise to be used for all gps algorithms
        self.noise_data = np.zeros((self.max_iterations, self.M, self._hyperparams['num_samples'], self.T, self.dU))

        # IterationData objects for each condition.
        self.cur = [[IterationData() for _ in range(self.M)] for _ in range(self.n_gps)]
        self.prev = [[IterationData() for _ in range(self.M)] for _ in range(self.n_gps)]

        # Trajectory Info #
        # --------------- #
        # Add dynamics if the algorithm requires fit_dynamics (Same type for all the conditions)
        if self._hyperparams['fit_dynamics']:
            dynamics = self._hyperparams['dynamics']

        # Initial trajectory hyperparams
        init_traj_distr = self._hyperparams['init_traj_distr']
        init_traj_distr['x0'] = env.get_conditions()
        init_traj_distr['dX'] = self.dX
        init_traj_distr['dU'] = self.dU
        init_traj_distr['dt'] = self.dt
        init_traj_distr['T'] = self.T

        for gps in range(self.n_gps):
            for m in range(self.M):
                self.cur[gps][m].traj_info = TrajectoryInfo()

                if self._hyperparams['fit_dynamics']:
                    self.cur[gps][m].traj_info.dynamics = dynamics['type'](dynamics)

                # Get the initial trajectory distribution hyperparams
                init_traj_distr = extract_condition(self._hyperparams['init_traj_distr'], self._train_cond_idx[m])

                # Instantiate Trajectory Distribution: init_lqr or init_pd
                self.cur[gps][m].traj_distr = init_traj_distr['type'](init_traj_distr)

        # Last trajectory distribution optimized in C-step
        self.new_traj_distr = [None for _ in range(self.n_gps)]

        # Traj Opt (Local policy opt) method #
        # ---------------------------------- #
        # Options: LQR, PI3
        self.traj_opt = [self._hyperparams['traj_opt'][gps]['type'](self._hyperparams['traj_opt'][gps])
                         for gps in range(self.n_gps)]
        for gps in range(self.n_gps):
            self.traj_opt[gps].set_logger(logging.getLogger('log%d' % gps))

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
        self.policy_opt = [self.agents[gps].policy_opt for gps in range(self.n_gps)]

        # Duality Data #
        # ------------ #
        # Duality data with: [sample_list, samples_cost, cs_traj, traj_dist, pol_info]
        self.good_duality_info = [[DualityInfo() for _ in range(self.M)] for _ in range(self.n_gps)]
        self.bad_duality_info = [[DualityInfo() for _ in range(self.M)] for _ in range(self.n_gps)]

        # MDGPS data #
        # ---------- #
        if self.use_global_policy:
            for gps in range(self.n_gps):
                self._hyperparams['gps_algo_hyperparams'][gps]['T'] = self.T
                self._hyperparams['gps_algo_hyperparams'][gps]['dU'] = self.dU
                self._hyperparams['gps_algo_hyperparams'][gps]['dX'] = self.dX
                policy_prior = self._hyperparams['gps_algo_hyperparams'][gps]['policy_prior']
                for m in range(self.M):
                    # Same policy prior type for all conditions
                    self.cur[gps][m].pol_info = PolicyInfo(self._hyperparams['gps_algo_hyperparams'][gps])
                    self.cur[gps][m].pol_info.policy_prior = policy_prior['type'](policy_prior)

        # TrajectoryInfo for good and bad trajectories
        self.good_trajectories_info = [[None for _ in range(self.M)] for _ in range(self.n_gps)]
        self.bad_trajectories_info = [[None for _ in range(self.M)] for _ in range(self.n_gps)]
        self.base_kl_good = [None for _ in range(self.n_gps)]
        self.base_kl_bad = [None for _ in range(self.n_gps)]
        for gps in range(self.n_gps):
            for m in range(self.M):
                self.good_trajectories_info[gps][m] = TrajectoryInfo()
                self.bad_trajectories_info[gps][m] = TrajectoryInfo()

                if self._hyperparams['fit_dynamics']:
                    self.good_trajectories_info[gps][m].dynamics = dynamics['type'](dynamics)
                    self.bad_trajectories_info[gps][m].dynamics = dynamics['type'](dynamics)

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
                self.good_duality_info[gps][m].traj_dist = init_traj_distr['type'](init_traj_distr)
                self.bad_duality_info[gps][m].traj_dist = init_traj_distr['type'](init_traj_distr)

                # Set initial dual variables
                self.cur[gps][m].eta = self._hyperparams['gps_algo_hyperparams'][gps]['init_eta']
                self.cur[gps][m].nu = self._hyperparams['gps_algo_hyperparams'][gps]['init_nu']
                self.cur[gps][m].omega = self._hyperparams['gps_algo_hyperparams'][gps]['init_omega']

            # Good/Bad bounds
            self.base_kl_good[gps] = self._hyperparams['gps_algo_hyperparams'][gps]['base_kl_good']
            self.base_kl_bad[gps] = self._hyperparams['gps_algo_hyperparams'][gps]['base_kl_bad']

            # MDGPS data
            if self.use_global_policy:
                for m in range(self.M):
                    # Same policy prior in MDGPS for good/bad
                    self.good_duality_info[gps][m].pol_info = PolicyInfo(self._hyperparams['gps_algo_hyperparams'][gps])
                    self.good_duality_info[gps][m].pol_info.policy_prior = policy_prior['type'](policy_prior)
                    self.bad_duality_info[gps][m].pol_info = PolicyInfo(self._hyperparams['gps_algo_hyperparams'][gps])
                    self.bad_duality_info[gps][m].pol_info.policy_prior = policy_prior['type'](policy_prior)

        # Threads data #
        # ------------ #
        # Queue to request environment
        self.environment_queue = Queue(maxsize=0)

        # Flag to confirm that samples are ready
        self.samples_done = [False for _ in range(self.n_gps)]

        # Thread to manage the environment
        self.multi_sampler_worker = Thread(target=self._multi_take_sample, args=())
        self.multi_sampler_worker.setDaemon(True)

        # List for each gps thread
        self.gps_workers = list()

    def run(self, itr_load=None):
        """
        Run GPS. If itr_load is specified, first loads the algorithm state from that iteration and resumes training at
        the next iteration.
        
        Args:
            itr_load: Desired iteration to load algorithm from

        Returns: True/False if all the gps algorithms have finished properly

        """
        run_successfully = True

        # Generate same noise for all gps algorithms
        if self._hyperparams['noisy_samples']:
            for ii in range(self.max_iterations):
                for cond in range(self.M):
                    for n in range(self._hyperparams['num_samples']):
                        self.noise_data[ii, cond, n, :, :] = self.get_noise()

        try:
            # Run gazebo sampler
            self.multi_sampler_worker.start()

            # Run each gps
            for gps in range(self.n_gps):
                itr_start = self._initialize(itr_load)
                self.gps_workers.append(Thread(target=self._multi_iteration, args=(gps, itr_start)))
                self.gps_workers[gps].setDaemon(True)
                self.gps_workers[gps].start()

            # Join/Wait until each gps thread finishes
            for gps_worker in self.gps_workers:
                gps_worker.join()

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

    def _multi_iteration(self, number_gps, itr_start):
        logger = logging.getLogger('log%d' % number_gps)

        try:
            for itr in range(itr_start, self.max_iterations):
                # Sample
                logger.info('')
                logger.info('->GPS:%02d itr:%02d | Sampling from local trajectories...' % (number_gps, itr+1))
                traj_or_pol = 'traj'
                self.environment_queue.put((number_gps, traj_or_pol, itr))
                while not self.samples_done[number_gps]:
                    pass
                self.samples_done[number_gps] = False

                # Get samples from agent
                traj_sample_lists = [self.agents[number_gps].get_samples(cond, -self._hyperparams['num_samples'])
                                     for cond in self._train_cond_idx]
                # Clear agent sample
                self.agents[number_gps].clear_samples()  # TODO: Check if it is better to 'remember' these samples

                for m, m_train in enumerate(self._train_cond_idx):
                    self.cur[number_gps][m_train].sample_list = traj_sample_lists[m]

                # Update dynamics model using all samples.
                logger.info('')
                logger.info('->GPS:%02d itr:%02d | Updating dynamics linearization...' % (number_gps, itr+1))
                self._update_dynamics(number_gps)

                logger.info('')
                logger.info('->GPS:%02d itr:%02d | Evaluating samples costs...' % (number_gps, itr+1))
                for m in range(self.M):
                    self._eval_cost(number_gps, m)

                # On the first iteration, need to catch policy up to init_traj_distr.
                if self.use_global_policy and self.iteration_count[number_gps] == 0:
                    self.new_traj_distr[number_gps] = [self.cur[number_gps][cond].traj_distr for cond in range(self.M)]
                    logger.info("\n"*2)
                    logger.info('->GPS:%02d itr:%02d | S-step for init_traj_distribution (iter=0)...' % (number_gps, itr+1))
                    self.update_policy(number_gps)

                # Update global policy linearizations.
                logger.info('')
                if self.use_global_policy:
                    logger.info('->GPS:%02d itr:%02d | Updating global policy linearization...' % (number_gps, itr+1))
                    for m in range(self.M):
                        self.update_policy_fit(number_gps, m)

                logger.info('')
                if self.use_global_policy and self.iteration_count[number_gps] > 0:
                    logger.info('->GPS:%02d itr:%02d | Updating KL step size with GLOBAL policy...' % (number_gps, itr+1))
                    self._update_step_size_global_policy(number_gps)
                else:
                    logger.info('->GPS:%02d itr:%02d | Updating KL step size with previous LOCAL policy...' % (number_gps, itr+1))
                    self._update_step_size(number_gps)  # KL Divergence step size.

                logger.info('')
                logger.info('->GPS:%02d itr:%02d | Getting good and bad trajectories...' % (number_gps, itr+1))
                self._get_good_trajectories(number_gps,
                                            option=self._hyperparams['gps_algo_hyperparams'][number_gps]['good_traj_selection_type'])
                self._get_bad_trajectories(number_gps,
                                           option=self._hyperparams['gps_algo_hyperparams'][number_gps]['bad_traj_selection_type'])

                logger.info('')
                logger.info('->GPS:%02d itr:%02d | Updating data of good and bad samples...' % (number_gps, itr+1))
                logger.info('-->GPS:%02d itr:%02d | Update g/b dynamics...' % (number_gps, itr+1))
                self._update_good_bad_dynamics(number_gps,
                                               option=self._hyperparams['gps_algo_hyperparams'][number_gps]['duality_dynamics_type'])
                logger.info('-->GPS:%02d itr:%02d | Update g/b costs...' % (number_gps, itr+1))
                self._eval_good_bad_costs(number_gps)
                logger.info('-->GPS:%02d itr:%02d | Update g/b traj dist...' % (number_gps, itr+1))
                self._fit_good_bad_traj_dist(number_gps)
                logger.info('-->GPS:%02d itr:%02d | Divergence btw good/bad trajs: ...' % (number_gps, itr+1))
                self._check_kl_div_good_bad(number_gps)

                # Run inner loop to compute new policies.
                logger.info('')
                logger.info('->GPS:%02d itr:%02d | Updating trajectories...' % (number_gps, itr+1))
                for ii in range(self._hyperparams['gps_algo_hyperparams'][number_gps]['inner_iterations']):
                    logger.info('-->GPS:%02d itr:%02d | Inner iteration %d/%d'
                                % (number_gps, itr+1, ii+1,
                                   self._hyperparams['gps_algo_hyperparams'][number_gps]['inner_iterations']))
                    self._update_trajectories(number_gps)

                if self.use_global_policy:
                    logger.info('')
                    logger.info('GPS:%02d itr:%02d | ->| S-step |<-' % (number_gps, itr+1))
                    self.update_policy(number_gps)

                self.advance_duality_iteration_variables(number_gps)

                # test_after_iter
                if self._hyperparams['test_after_iter']:
                    logger.info('')
                    logger.info('-->GPS:%02d itr:%02d | Testing global policy...' % (number_gps, itr+1))
                    traj_or_pol = 'pol'
                    self.environment_queue.put((number_gps, traj_or_pol, itr))

                    while not self.samples_done[number_gps]:
                        pass
                    self.samples_done[number_gps] = False

                    pol_sample_lists = list()
                    for m in range(self.M):
                        pol_sample_lists.append(self.prev[number_gps][m].pol_info.policy_samples)  # Because after advance_iter

                    pol_sample_lists_costs, pol_sample_lists_cost_compositions = self._eval_conditions_sample_list_cost(pol_sample_lists)

                else:
                    pol_sample_lists = None
                    pol_sample_lists_costs = None
                    pol_sample_lists_cost_compositions = None

                # Log data
                self._log_data(number_gps, itr, traj_sample_lists, pol_sample_lists, pol_sample_lists_costs,
                               pol_sample_lists_cost_compositions)

        except Exception as e:
            logger.exception("Error in GPS!!!")

    def _multi_take_sample(self):
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

        while True:
            verbose = False
            if self.environment_queue.empty():  # queue empty
                #print("NOthing in queue")
                time.sleep(0.5)
                pass
            else:
                last_in_queue = self.environment_queue.get()
                gps = last_in_queue[0]
                traj_or_pol = last_in_queue[1]
                itr = last_in_queue[2]

                print("Sampling for GPS:%02d | mode:%s" % (gps, traj_or_pol))

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
                    if on_policy and (self.iteration_count[gps] > 0 or
                                          ('sample_pol_first_itr' in self._hyperparams and self._hyperparams['sample_pol_first_itr'])):
                        policy = self.agents[gps].policy  # DOM: Instead self.opt_pol.policy
                        print("On-policy sampling: %s!" % type(policy))
                    else:
                        policy = self.cur[gps][cond].traj_distr
                        print("Off-policy sampling: %s!" % type(policy))

                    for i in range(total_samples):
                        # Create a sample class
                        sample = Sample(self.env, self.T)
                        history = [None] * self.T
                        obs_hist = [None] * self.T

                        #sample = self._take_fake_sample(itr, cond, i, verbose=True, save=True, noisy=True, on_policy=False)

                        print("Resetting environment...")
                        self.env.reset(time=2, cond=cond)

                        if issubclass(type(self.env), GymEnv):
                            gym_ts = self.dt
                        else:
                            ros_rate = rospy.Rate(int(1/self.dt))  # hz

                        # Collect history
                        if on_policy:
                            print("On-policy sample gps:%d | itr:%d/%d, cond:%d/%d, i:%d/%d" % (gps, itr+1, self.max_iterations,
                                                                                                cond+1, len(self._train_cond_idx),
                                                                                                i+1, total_samples))
                        else:
                            print("Sample gps:%d | itr:%d/%d, cond:%d/%d, s:%d/%d" % (gps, itr+1, self.max_iterations,
                                                                                      cond+1, len(self._train_cond_idx),
                                                                                      i+1, total_samples))
                        sampling_bar = ProgressBar(self.T, bar_title='Sampling')
                        
                        for t in range(self.T):
                            sampling_bar.update(t)
                            if verbose:
                                if on_policy:
                                    print("On-policy sample gps:%d | itr:%d/%d, cond:%d/%d, i:%d/%d | t:%d/%d" % (gps, itr+1, self.max_iterations,
                                                                                                                  cond+1, len(self._train_cond_idx),
                                                                                                                  i+1, total_samples,
                                                                                                                  t+1, self.T))
                                else:
                                    print("Sample gps:%d | itr:%d/%d, cond:%d/%d, s:%d/%d | t:%d/%d" % (gps, itr+1, self.max_iterations,
                                                                                                        cond+1, len(self._train_cond_idx),
                                                                                                        i+1, total_samples,
                                                                                                t+1, self.T))
                            obs = self.env.get_observation()
                            # Checking NAN
                            nan_number = np.isnan(obs)
                            if np.any(nan_number):
                                print("\e[31mERROR OBSERVATION: NAN!!!!! gps:%d | type:%s | itr %d | cond%d | i:%d t:%d" % (gps, traj_or_pol, itr+1, cond+1, i+1, t+1))
                            state = self.env.get_state()
                            if traj_or_pol == 'traj':
                                noise = self.noise_data[itr, cond, i, t, :]
                            else:
                                noise = zero_noise
                            action = policy.eval(state.copy(), obs.copy(), t, noise.copy())  # TODO: Avoid TF policy writes in obs
                            # Checking NAN
                            nan_number = np.isnan(action)
                            if np.any(nan_number):
                                print("\e[31mERROR ACTION: NAN!!!!! gps:%d | type:%s | itr %d | cond%d | i:%d t:%d" % (gps, traj_or_pol, itr+1, cond+1, i+1, t+1))
                            action[nan_number] = 0
                            self.env.send_action(action)
                            obs_hist[t] = (obs, action)
                            history[t] = (state, action)
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
                        sample.set_noise(self.noise_data[itr])        # Set all noise at the same time

                        if save:  # Save sample in agent sample list
                            sample_id = self.agents[gps].add_sample(sample, cond)
                            print("The sample was added to Agent's sample list. Now there are %d sample(s) for condition '%d'." %
                                  (sample_id+1, cond))

                        if traj_or_pol == 'pol':
                            pol_samples[cond].append(sample)

                    if traj_or_pol == 'pol':
                        self.prev[gps][cond].pol_info.policy_samples = SampleList(pol_samples[cond])  # prev because it is called after advance_iteration_variables

                self.samples_done[gps] = True

    def _check_kl_div_good_bad(self, number_gps):
        logger = logging.getLogger('log%d' % number_gps)
        for cond in range(self.M):
            good_distr = self.good_duality_info[number_gps][cond].traj_dist
            bad_distr = self.bad_duality_info[number_gps][cond].traj_dist
            mu_good, sigma_good = lqr_forward(good_distr, self.good_trajectories_info[number_gps][cond])
            mu_bad, sigma_bad = lqr_forward(bad_distr, self.bad_trajectories_info[number_gps][cond])
            kl_div_good_bad = traj_distr_kl_alt(mu_good, sigma_good, good_distr, bad_distr, tot=True)
            #print("G/B KL_div: %f " % kl_div_good_bad)
            logger.info('--->Divergence btw good/bad trajs is: %f' % kl_div_good_bad)

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

    def compute_costs(self, number_gps, m, eta, omega, nu, augment=True):
        """
        Compute cost estimates used in the LQR backward pass.
        
        Args:
            number_gps: Index gps algorithm 
            m: Condition
            eta: Dual variable corresponding to KL divergence with previous policy.
            omega: Dual variable(s) corresponding to KL divergence with good trajectories.
            nu: Dual variable(s) corresponding to KL divergence with bad trajectories.
            augment: True if we want a KL constraint for all time-steps. False otherwise.

        Returns: Cm and cv

        """
        traj_info = self.cur[number_gps][m].traj_info
        traj_distr = self.cur[number_gps][m].traj_distr
        good_distr = self.good_duality_info[number_gps][m].traj_dist
        bad_distr = self.bad_duality_info[number_gps][m].traj_dist
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

    def advance_duality_iteration_variables(self, number_gps):
        """
        Move all 'cur' variables to 'prev', reinitialize 'cur' variables, and advance iteration counter.
        :return: None
        """
        self._advance_iteration_variables(number_gps)
        for m in range(self.M):
            self.cur[number_gps][m].nu = self.prev[number_gps][m].nu
            self.cur[number_gps][m].omega = self.prev[number_gps][m].omega

            if self.use_global_policy:
                self.cur[number_gps][m].traj_info.last_kl_step = self.prev[number_gps][m].traj_info.last_kl_step
                self.cur[number_gps][m].pol_info = copy.deepcopy(self.prev[number_gps][m].pol_info)

    def _fit_good_bad_traj_dist(self, number_gps):
        min_good_var = self._hyperparams['gps_algo_hyperparams'][number_gps]['min_good_var']
        min_bad_var = self._hyperparams['gps_algo_hyperparams'][number_gps]['min_good_var']

        for cond in range(self.M):
            self.good_duality_info[number_gps][cond].traj_dist = self.fit_traj_dist(self.good_duality_info[number_gps][cond].sample_list, min_good_var)
            self.bad_duality_info[number_gps][cond].traj_dist = self.fit_traj_dist(self.bad_duality_info[number_gps][cond].sample_list, min_bad_var)

    def _eval_good_bad_costs(self, number_gps):
        """
        Evaluate costs for all current samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        for cond in range(self.M):
            cs, cc, cv, Cm = self._eval_dual_costs(cond, self.good_duality_info[number_gps][cond].sample_list)
            self.good_duality_info[number_gps][cond].traj_cost = cs
            self.good_trajectories_info[number_gps][cond].cc = cc
            self.good_trajectories_info[number_gps][cond].cv = cv
            self.good_trajectories_info[number_gps][cond].Cm = Cm

            cs, cc, cv, Cm = self._eval_dual_costs(cond, self.bad_duality_info[number_gps][cond].sample_list)
            self.bad_duality_info[number_gps][cond].traj_cost = cs
            self.bad_trajectories_info[number_gps][cond].cc = cc
            self.bad_trajectories_info[number_gps][cond].cv = cv
            self.bad_trajectories_info[number_gps][cond].Cm = Cm

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

    def _update_good_bad_dynamics(self, number_gps, option='duality'):
        """
        Instantiate dynamics objects and update prior. Fit dynamics to sample(s).
        """
        for m in range(self.M):
            if option == 'duality':
                good_data = self.good_duality_info[number_gps][m].sample_list
                bad_data = self.bad_duality_info[number_gps][m].sample_list
            else:
                good_data = self.cur[number_gps][m].sample_list
                bad_data = self.cur[number_gps][m].sample_list

            X_good = good_data.get_states()
            U_good = good_data.get_actions()
            X_bad = bad_data.get_states()
            U_bad = bad_data.get_actions()

            # Update prior and fit dynamics.
            self.good_trajectories_info[number_gps][m].dynamics.update_prior(good_data)
            self.good_trajectories_info[number_gps][m].dynamics.fit(X_good, U_good)
            self.bad_trajectories_info[number_gps][m].dynamics.update_prior(bad_data)
            self.bad_trajectories_info[number_gps][m].dynamics.fit(X_bad, U_bad)

            # Fit x0mu/x0sigma.
            x0_good = X_good[:, 0, :]
            x0mu_good = np.mean(x0_good, axis=0)  # TODO: SAME X0 FOR ALL??
            self.good_trajectories_info[number_gps][m].x0mu = x0mu_good
            self.good_trajectories_info[number_gps][m].x0sigma = np.diag(np.maximum(np.var(x0_good, axis=0),
                                                                        self._hyperparams['initial_state_var']))
            x0_bad = X_bad[:, 0, :]
            x0mu_bad = np.mean(x0_bad, axis=0)  # TODO: SAME X0 FOR ALL??
            self.bad_trajectories_info[number_gps][m].x0mu = x0mu_bad
            self.bad_trajectories_info[number_gps][m].x0sigma = np.diag(np.maximum(np.var(x0_bad, axis=0),
                                                                       self._hyperparams['initial_state_var']))

            prior_good = self.good_trajectories_info[number_gps][m].dynamics.get_prior()
            if prior_good:
                mu0, Phi, priorm, n0 = prior_good.initial_state()
                N = len(good_data)
                self.good_trajectories_info[number_gps][m].x0sigma += Phi + (N*priorm) / (N+priorm) * np.outer(x0mu_good-mu0, x0mu_good-mu0) / (N+n0)

            prior_bad = self.good_trajectories_info[number_gps][m].dynamics.get_prior()
            if prior_bad:
                mu0, Phi, priorm, n0 = prior_bad.initial_state()
                N = len(bad_data)
                self.bad_trajectories_info[number_gps][m].x0sigma += Phi + (N*priorm) / (N+priorm) * np.outer(x0mu_bad-mu0, x0mu_bad-mu0) / (N+n0)

    def _get_bad_trajectories(self, number_gps, option='only_traj'):
        """
        Get bad trajectory samples.
        
        Args:
            number_gps: Index of agent(gps method) to update
            option (str): 'only_traj': update good_duality_info sample list only when the trajectory sample is worse
                                       than any previous sample.
                          'always': update bad_duality_info sample list with the worst trajectory samples in the current
                                    iteration.

        Returns:
            None

        """

        LOGGER = logging.getLogger('log%d' % number_gps)

        for cond in range(self.M):
            # Sample costs estimate.
            cs = self.cur[number_gps][cond].cs
            sample_list = self.cur[number_gps][cond].sample_list

            # Get index of sample with worst Return
            #worst_index = np.argmax(np.sum(cs, axis=1))
            n_bad = self._hyperparams['gps_algo_hyperparams'][number_gps]['n_bad_samples']
            if n_bad == cs.shape[0]:
                worst_indeces = range(n_bad)
            else:
                worst_indeces = np.argpartition(np.sum(cs, axis=1), -n_bad)[-n_bad:]

            # Get current best trajectory
            if self.bad_duality_info[number_gps][cond].sample_list is None:
                for bb, bad_index in enumerate(worst_indeces):
                    LOGGER.info("GPS:%02d | Defining BAD trajectory sample %d | cur_cost=%f from sample %d" % (number_gps, bb,
                                                                                          np.sum(cs[bad_index, :]), bad_index))
                self.bad_duality_info[number_gps][cond].sample_list = SampleList([sample_list[bad_index] for bad_index in worst_indeces])
                self.bad_duality_info[number_gps][cond].samples_cost = cs[worst_indeces, :]
            else:
                # Update only if it is better than before
                if option == 'only_traj':
                    for bad_index in worst_indeces:
                        least_worst_index = np.argpartition(np.sum(self.bad_duality_info[number_gps][cond].samples_cost, axis=1), 1)[:1]
                        if np.sum(self.bad_duality_info[number_gps][cond].samples_cost[least_worst_index, :]) < np.sum(cs[bad_index, :]):
                            LOGGER.info("GPS:%02d | Updating BAD trajectory sample %d | cur_cost=%f < new_cost=%f"
                                  % (number_gps, least_worst_index,
                                     np.sum(self.bad_duality_info[number_gps][cond].samples_cost[least_worst_index, :]),
                                     np.sum(cs[bad_index, :])))
                            self.bad_duality_info[number_gps][cond].sample_list.set_sample(least_worst_index, sample_list[bad_index])
                            self.bad_duality_info[number_gps][cond].samples_cost[least_worst_index, :] = cs[bad_index, :]
                elif option == 'always':
                    for bb, bad_index in enumerate(worst_indeces):
                        print("Worst bad index is %d | and replaces %d" % (bad_index, bb))
                        LOGGER.info("GPS:%02d | Updating BAD trajectory sample %d | cur_cost=%f < new_cost=%f"
                              % (number_gps, bb, np.sum(self.bad_duality_info[number_gps][cond].samples_cost[bb, :]),
                                 np.sum(cs[bad_index, :])))
                        self.bad_duality_info[number_gps][cond].sample_list.set_sample(bb, sample_list[bad_index])
                        self.bad_duality_info[number_gps][cond].samples_cost[bb, :] = cs[bad_index, :]
                else:
                    raise ValueError("GPS:%02d | Wrong get_bad_grajectories option: %s" % (number_gps, option))

    def _get_good_trajectories(self, number_gps, option='only_traj'):
        """
        Get good trajectory samples.
        
        Args:
            number_gps: Index of agent(gps method) to update
            option (str): 'only_traj': update good_duality_info sample list only when the trajectory sample is better 
                                       than any previous sample.
                          'always': update good_duality_info sample list with the best trajectory samples in the current
                                    iteration.

        Returns:
            None

        """

        LOGGER = logging.getLogger('log%d' % number_gps)

        for cond in range(self.M):
            # Sample costs estimate.
            cs = self.cur[number_gps][cond].cs
            sample_list = self.cur[number_gps][cond].sample_list

            # Get index of sample with best Return
            #best_index = np.argmin(np.sum(cs, axis=1))
            n_good = self._hyperparams['gps_algo_hyperparams'][number_gps]['n_good_samples']
            if n_good == cs.shape[0]:
                best_indeces = range(n_good)
            else:
                best_indeces = np.argpartition(np.sum(cs, axis=1), n_good)[:n_good]

            # Get current best trajectory
            if self.good_duality_info[number_gps][cond].sample_list is None:
                for gg, good_index in enumerate(best_indeces):
                    LOGGER.info("GPS:%02d | Defining GOOD trajectory sample %d | cur_cost=%f from sample %d" % (number_gps, gg,
                                                                                                                np.sum(cs[good_index, :]), good_index))
                self.good_duality_info[number_gps][cond].sample_list = SampleList([sample_list[good_index] for good_index in best_indeces])
                self.good_duality_info[number_gps][cond].samples_cost = cs[best_indeces, :]
            else:
                # Update only if it is better than previous traj_dist
                if option == 'only_traj':
                    # If there is a better trajectory, replace only that trajectory to previous ones
                    for good_index in best_indeces:
                        least_best_index = np.argpartition(np.sum(self.good_duality_info[number_gps][cond].samples_cost, axis=1), -1)[-1:]
                        if np.sum(self.good_duality_info[number_gps][cond].samples_cost[least_best_index, :]) > np.sum(cs[good_index, :]):
                            LOGGER.info("GPS:%02d | Updating GOOD trajectory sample %d | cur_cost=%f > new_cost=%f"
                                  % (number_gps, least_best_index,
                                     np.sum(self.good_duality_info[number_gps][cond].samples_cost[least_best_index, :]),
                                     np.sum(cs[good_index, :])))
                            self.good_duality_info[number_gps][cond].sample_list.set_sample(least_best_index, sample_list[good_index])
                            self.good_duality_info[number_gps][cond].samples_cost[least_best_index, :] = cs[good_index, :]
                elif option == 'always':
                    for gg, good_index in enumerate(best_indeces):
                        print("Best good index is %d | and replaces %d" % (good_index, gg))
                        LOGGER.info("GPS:%02d | Updating GOOD trajectory sample %d | cur_cost=%f > new_cost=%f"
                              % (number_gps, gg, np.sum(self.good_duality_info[number_gps][cond].samples_cost[gg, :]),
                                 np.sum(cs[good_index, :])))
                        self.good_duality_info[number_gps][cond].sample_list.set_sample(gg, sample_list[good_index])
                        self.good_duality_info[number_gps][cond].samples_cost[gg, :] = cs[good_index, :]
                else:
                    raise ValueError("GPS:%02d | Wrong get_good_grajectories option: %s" % (number_gps, option))

    def _update_step_size(self, number_policy):
        """ Evaluate costs on samples, and adjust the step size. """
        # Evaluate cost function for all conditions and samples.
        #for m in range(self.M):
        #    self._eval_cost(m)

        # Adjust step size relative to the previous iteration.
        for m in range(self.M):
            if self.iteration_count[number_policy] >= 1 and self.prev[number_policy][m].sample_list:
                self._stepadjust(number_policy, m)

    def _stepadjust(self, number_gps, m):
        """
        Calculate new step sizes.
        Args:
            m: Condition
        """
        LOGGER = logging.getLogger('log%d' % number_gps)

        # Compute values under Laplace approximation. This is the policy
        # that the previous samples were actually drawn from under the
        # dynamics that were estimated from the previous samples.
        previous_laplace_obj = self.traj_opt[number_gps].estimate_cost(
            self.prev[number_gps][m].traj_distr, self.prev[number_gps][m].traj_info
        )
        # This is the policy that we just used under the dynamics that
        # were estimated from the previous samples (so this is the cost
        # we thought we would have).
        new_predicted_laplace_obj = self.traj_opt[number_gps].estimate_cost(
            self.cur[number_gps][m].traj_distr, self.prev[number_gps][m].traj_info
        )

        # This is the actual cost we have under the current trajectory
        # based on the latest samples.
        new_actual_laplace_obj = self.traj_opt[number_gps].estimate_cost(
            self.cur[number_gps][m].traj_distr, self.cur[number_gps][m].traj_info
        )

        # Measure the entropy of the current trajectory (for printout).
        ent = self._measure_ent(number_gps, m)

        # Compute actual objective values based on the samples.
        previous_mc_obj = np.mean(np.sum(self.prev[number_gps][m].cs, axis=1), axis=0)
        new_mc_obj = np.mean(np.sum(self.cur[number_gps][m].cs, axis=1), axis=0)

        LOGGER.debug('GPS:%02d | Trajectory step: ent: %f cost: %f -> %f', number_gps, ent, previous_mc_obj, new_mc_obj)

        # Compute predicted and actual improvement.
        predicted_impr = np.sum(previous_laplace_obj) - \
                         np.sum(new_predicted_laplace_obj)
        actual_impr = np.sum(previous_laplace_obj) - \
                      np.sum(new_actual_laplace_obj)

        # Print improvement details.
        LOGGER.debug('GPS:%02d | Previous cost: Laplace: %f MC: %f', number_gps,
                     np.sum(previous_laplace_obj), previous_mc_obj)
        LOGGER.debug('GPS:%02d | Predicted new cost: Laplace: %f MC: %f', number_gps,
                     np.sum(new_predicted_laplace_obj), new_mc_obj)
        LOGGER.debug('GPS:%02d | Actual new cost: Laplace: %f MC: %f', number_gps,
                     np.sum(new_actual_laplace_obj), new_mc_obj)
        LOGGER.debug('GPS:%02d | Predicted/actual improvement: %f / %f', number_gps,
                     predicted_impr, actual_impr)

        self._set_new_mult(predicted_impr, actual_impr, number_gps, m)

    def _update_step_size_global_policy(self, number_gps):
        """
        Calculate new step sizes. This version uses the same step size for all conditions.
        """
        LOGGER = logging.getLogger('log%d' % number_gps)


        # Compute previous cost and previous expected cost.
        prev_M = len(self.prev[number_gps])  # May be different in future.
        prev_laplace = np.empty(prev_M)
        prev_mc = np.empty(prev_M)
        prev_predicted = np.empty(prev_M)
        for m in range(prev_M):
            prev_nn = self.prev[number_gps][m].pol_info.traj_distr()
            prev_lg = self.prev[number_gps][m].new_traj_distr

            # Compute values under Laplace approximation. This is the policy that the previous samples were actually
            # drawn from under the dynamics that were estimated from the previous samples.
            prev_laplace[m] = self.traj_opt[number_gps].estimate_cost(prev_nn, self.prev[number_gps][m].traj_info).sum()
            # This is the actual cost that we experienced.
            prev_mc[m] = self.prev[number_gps][m].cs.mean(axis=0).sum()
            # This is the policy that we just used under the dynamics that were estimated from the prev samples (so
            # this is the cost we thought we would have).
            prev_predicted[m] = self.traj_opt[number_gps].estimate_cost(prev_lg, self.prev[number_gps][m].traj_info).sum()

        # Compute current cost.
        cur_laplace = np.empty(self.M)
        cur_mc = np.empty(self.M)
        for m in range(self.M):
            cur_nn = self.cur[number_gps][m].pol_info.traj_distr()
            # This is the actual cost we have under the current trajectory based on the latest samples.
            cur_laplace[m] = self.traj_opt[number_gps].estimate_cost(cur_nn, self.cur[number_gps][m].traj_info).sum()
            cur_mc[m] = self.cur[number_gps][m].cs.mean(axis=0).sum()

        # Compute predicted and actual improvement.
        prev_laplace = prev_laplace.mean()
        prev_mc = prev_mc.mean()
        prev_predicted = prev_predicted.mean()
        cur_laplace = cur_laplace.mean()
        cur_mc = cur_mc.mean()
        if self._hyperparams['gps_algo_hyperparams'][number_gps]['step_rule'] == 'laplace':
            predicted_impr = prev_laplace - prev_predicted
            actual_impr = prev_laplace - cur_laplace
        elif self._hyperparams['gps_algo_hyperparams'][number_gps]['step_rule'] == 'mc':
            predicted_impr = prev_mc - prev_predicted
            actual_impr = prev_mc - cur_mc
        LOGGER.debug('GPS:02d | Previous cost: Laplace: %f, MC: %f', number_gps, prev_laplace, prev_mc)
        LOGGER.debug('GPS:02d | Predicted cost: Laplace: %f', number_gps, prev_predicted)
        LOGGER.debug('GPS:02d | Actual cost: Laplace: %f, MC: %f', number_gps, cur_laplace, cur_mc)

        for m in range(self.M):
            self._set_new_mult(predicted_impr, actual_impr, number_gps, m)

    def update_policy_fit(self, number_gps, cond):
        """
        Re-estimate the local policy values in the neighborhood of the trajectory.
        :param cond: Condition
        :return: None
        """
        dX, dU, T = self.dX, self.dU, self.T
        # Choose samples to use.
        samples = self.cur[number_gps][cond].sample_list
        N = len(samples)
        pol_info = self.cur[number_gps][cond].pol_info
        X = samples.get_states().copy()
        obs = samples.get_obs().copy()
        pol_mu, pol_sig = self.policy_opt[number_gps].prob(obs)[:2]
        pol_info.pol_mu, pol_info.pol_sig = pol_mu, pol_sig

        # Update policy prior.
        policy_prior = pol_info.policy_prior
        samples = SampleList(self.cur[number_gps][cond].sample_list)
        mode = self._hyperparams['gps_algo_hyperparams'][number_gps]['policy_sample_mode']
        policy_prior.update(samples, self.policy_opt[number_gps], mode)

        # Fit linearization and store in pol_info.
        pol_info.pol_K, pol_info.pol_k, pol_info.pol_S = policy_prior.fit(X, pol_mu, pol_sig)
        for t in range(T):
            pol_info.chol_pol_S[t, :, :] = sp.linalg.cholesky(pol_info.pol_S[t, :, :])

    def update_policy(self, number_gps):
        """
        Computes(updates) a new global policy.
        :return: 
        """
        LOGGER = logging.getLogger('log%d' % number_gps)

        LOGGER.info('-->Updating Global policy...')
        dU, dO, T = self.dU, self.dO, self.T
        # Compute target mean, cov(precision), and weight for each sample; and concatenate them.
        obs_data, tgt_mu = np.zeros((0, T, dO)), np.zeros((0, T, dU))
        tgt_prc, tgt_wt = np.zeros((0, T, dU, dU)), np.zeros((0, T))
        for m in range(self.M):
            samples = self.cur[number_gps][m].sample_list
            X = samples.get_states()
            N = len(samples)
            traj = self.new_traj_distr[number_gps][m]
            pol_info = self.cur[number_gps][m].pol_info
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

        logger = logging.getLogger('log%d' % number_gps)

        self.policy_opt[number_gps].update(obs_data, tgt_mu, tgt_prc, tgt_wt, LOGGER=logger)



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

    def _log_data(self, number_gps, itr, traj_sample_lists, pol_sample_lists=None, pol_sample_lists_costs=None,
                  pol_sample_lists_cost_compositions=None):
        """
        Log data and algorithm.
        :param itr: Iteration number.
        :param traj_sample_lists: trajectory (local policies) samples as SampleList object.
        :param pol_sample_lists: global policy samples as SampleList object.
        :return: None
        """
        LOGGER = logging.getLogger('log%d' % number_gps)
        LOGGER.info("Logging Agent... ")
        self.data_logger.pickle(
            ('gps%02d_agent_itr_%02d.pkl' % (number_gps, itr)),
            # copy.copy(temp_dict)
            copy.copy(self.agents[number_gps])
        )
        if self.use_global_policy:
            LOGGER.info("Logging Policy_Opt... ")
            self.data_logger.pickle(
                ('gps%02d_policy_opt_itr_%02d.pkl' % (number_gps, itr)),
                self.agents[number_gps].policy_opt
            )
            print("TODO: NOT LOGGING POLICY!!!")
            #LOGGER.info("Logging Policy... ")
            #self.agents[number_gps].policy_opt.policy.pickle_policy(self.dO, self.dU,
            #                                           self.data_logger.dir_path + '/' + ('gps%02d_policy_itr_%02d' % (number_gps, itr)),
            #                                           goal_state=None,
            #                                           should_hash=False)

        print("TODO: CHECK HOW TO SOLVE LOGGING MULTIGPS ALGO")
        # print("Logging GPS algorithm state... ")
        # self.data_logger.pickle(
        #     ('%s_algorithm_itr_%02d.pkl' % (self.gps_algo.upper(), itr)),
        #     copy.copy(self)
        # )

        LOGGER.info("Logging GPS iteration data... ")
        self.data_logger.pickle(
            ('gps%02d_%s_iteration_data_itr_%02d.pkl' % (number_gps, self.gps_algo[number_gps].upper(), itr)),
            copy.copy(self.prev[number_gps])  # prev instead of cur
        )

        LOGGER.info("Logging Trajectory samples... ")
        self.data_logger.pickle(
            ('gps%02d_traj_sample_itr_%02d.pkl' % (number_gps, itr)),
            copy.copy(traj_sample_lists)
        )

        if pol_sample_lists is not None:
            LOGGER.info("Logging Global Policy samples... ")
            self.data_logger.pickle(
                ('gps%02d_pol_sample_itr_%02d.pkl' % (number_gps, itr)),
                copy.copy(pol_sample_lists)
            )

        if pol_sample_lists_costs is not None:
            LOGGER.info("Logging Global Policy samples costs... ")
            self.data_logger.pickle(
                ('gps%02d_pol_sample_cost_itr_%02d.pkl' % (number_gps, itr)),
                copy.copy(pol_sample_lists_costs)
            )

        if pol_sample_lists_cost_compositions is not None:
            LOGGER.info("Logging Global Policy samples cost compositions... ")
            self.data_logger.pickle(
                ('gps%02d_pol_sample_cost_composition_itr_%02d.pkl' % (number_gps, itr)),
                copy.copy(pol_sample_lists_cost_compositions)
            )

        LOGGER.info("Logging God/Bad duality data")
        self.data_logger.pickle(
            ('gps%02d_good_trajectories_info_itr_%02d.pkl' % (number_gps, itr)),
            copy.copy(self.good_trajectories_info[number_gps])
        )
        self.data_logger.pickle(
            ('gps%02d_bad_trajectories_info_itr_%02d.pkl' % (number_gps, itr)),
            copy.copy(self.bad_trajectories_info[number_gps])
        )
        self.data_logger.pickle(
            ('gps%02d_good_duality_info_itr_%02d.pkl' % (number_gps, itr)),
            copy.copy(self.good_duality_info[number_gps])
        )
        self.data_logger.pickle(
            ('gps%02d_bad_duality_info_itr_%02d.pkl' % (number_gps, itr)),
            copy.copy(self.bad_duality_info[number_gps])
        )

    def _update_dynamics(self, gps):
        """
        Instantiate dynamics objects and update prior. Fit dynamics to current samples.
        """
        for m in range(self.M):
            cur_data = self.cur[gps][m].sample_list
            X = cur_data.get_states()
            U = cur_data.get_actions()

            # Update prior and fit dynamics.
            self.cur[gps][m].traj_info.dynamics.update_prior(cur_data)
            self.cur[gps][m].traj_info.dynamics.fit(X, U)

            # Fit x0mu/x0sigma.
            x0 = X[:, 0, :]
            x0mu = np.mean(x0, axis=0)
            self.cur[gps][m].traj_info.x0mu = x0mu
            self.cur[gps][m].traj_info.x0sigma = np.diag(np.maximum(np.var(x0, axis=0),
                                                    self._hyperparams['initial_state_var']))

            prior = self.cur[gps][m].traj_info.dynamics.get_prior()
            if prior:
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = len(cur_data)
                self.cur[gps][m].traj_info.x0sigma += Phi + (N*priorm) / (N+priorm) * np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)

    def _update_trajectories(self, number_gps):
        """
        Compute new linear Gaussian controllers.
        """
        LOGGER = logging.getLogger('log%d' % number_gps)

        LOGGER.info('-->GPS:%02d | Updating trajectories (local policies)...' % number_gps)
        if self.new_traj_distr[number_gps] is None:
            self.new_traj_distr[number_gps] = [self.cur[number_gps][cond].traj_distr for cond in range(self.M)]
        for cond in range(self.M):
            traj_opt_outputs = self.traj_opt[number_gps].update(cond, self, number_gps=number_gps)
            self.new_traj_distr[number_gps][cond] = traj_opt_outputs[0]
            self.cur[number_gps][cond].eta = traj_opt_outputs[1]
            self.cur[number_gps][cond].omega = traj_opt_outputs[2]
            self.cur[number_gps][cond].nu = traj_opt_outputs[3]

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

    def _eval_cost(self, number_gps, cond):
        """
        Evaluate costs for all current samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        # Constants.
        T, dX, dU = self.T, self.dX, self.dU
        N = len(self.cur[number_gps][cond].sample_list)

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX+dU))
        Cm = np.zeros((N, T, dX+dU, dX+dU))
        for n in range(N):
            sample = self.cur[number_gps][cond].sample_list[n]
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
        self.cur[number_gps][cond].traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        self.cur[number_gps][cond].traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        self.cur[number_gps][cond].traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        self.cur[number_gps][cond].cs = cs  # True value of cost.

    def _advance_iteration_variables(self, number_gps):
        """
        Move all 'cur' variables to 'prev', and advance iteration counter.
        :return: None
        """
        self.iteration_count[number_gps] += 1
        self.prev[number_gps] = copy.deepcopy(self.cur[number_gps])
        # TODO: change IterationData to reflect new stuff better
        for m in range(self.M):
            self.prev[number_gps][m].new_traj_distr = self.new_traj_distr[number_gps][m]

        # NEW IterationData object, and remove new_traj_distr
        self.cur[number_gps] = [IterationData() for _ in range(self.M)]
        for m in range(self.M):
            self.cur[number_gps][m].traj_info = TrajectoryInfo()
            self.cur[number_gps][m].traj_info.dynamics = copy.deepcopy(self.prev[number_gps][m].traj_info.dynamics)
            self.cur[number_gps][m].step_mult = self.prev[number_gps][m].step_mult
            self.cur[number_gps][m].eta = self.prev[number_gps][m].eta
            self.cur[number_gps][m].traj_distr = self.new_traj_distr[number_gps][m]
        self.new_traj_distr[number_gps] = None

    def _set_new_mult(self, predicted_impr, actual_impr, number_gps, m):
        """
        Adjust step size multiplier according to the predicted versus actual improvement.
        """
        LOGGER = logging.getLogger('log%d' % number_gps)

        # Model improvement as I = predicted_dI * KL + penalty * KL^2,
        # where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2).
        # Optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL =>
        # KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL.
        # Therefore, the new multiplier is given by pred/2*(pred-act).
        new_mult = predicted_impr / (2.0 * max(1e-4, predicted_impr - actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(min(new_mult * self.cur[number_gps][m].step_mult, self._hyperparams['max_step_mult']),
                       self._hyperparams['min_step_mult'])
        self.cur[number_gps][m].step_mult = new_step

        if new_mult > 1:
            LOGGER.debug('GPS:%02d | Increasing step size multiplier to %f', number_gps, new_step)
        else:
            LOGGER.debug('GPS:%02d | Decreasing step size multiplier to %f', number_gps, new_step)

    def _measure_ent(self, number_gps, m):
        """ Measure the entropy of the current trajectory. """
        ent = 0
        for t in range(self.T):
            ent = ent + np.sum(
                np.log(np.diag(self.cur[number_gps][m].traj_distr.chol_pol_covar[t, :, :]))
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

        # print(pol_sig)
        # raw_input('perhkhjk')

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
    def setup_logger(logger_name, dir_path, log_file, level=logging.INFO, also_screen=False):
        l = logging.getLogger(logger_name)
        formatter = logging.Formatter('%(asctime)s : %(message)s')

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        fileHandler = logging.FileHandler(dir_path+log_file, mode='w')
        fileHandler.setFormatter(formatter)
        l.setLevel(level)
        l.addHandler(fileHandler)

        if also_screen:
            streamHandler = logging.StreamHandler()
            streamHandler.setFormatter(formatter)
            l.addHandler(streamHandler)


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
