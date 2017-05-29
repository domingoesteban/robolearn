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
        self._conditions = self._hyperparams['conditions']

        if 'train_conditions' in self._hyperparams:
            self._train_idx = self._hyperparams['train_conditions']
            self._test_idx = self._hyperparams['test_conditions']
        else:
            self._train_idx = self._test_idx = range(self._conditions)
            self._hyperparams['train_conditions'] = self._train_idx
            self._hyperparams['test_conditions'] = self._test_idx

        self.data_logger = DataLogger()
        self._data_files_dir = "gps_data_files"

        # ############################### #
        # Code used in original Algorithm #
        # ############################### #
        self.M = self._conditions  # TODO: Review code so only one is used
        self._cond_idx = self._train_idx  # TODO: Review code so only one is used

        self.iteration_count = 0

        # Get some values from the environment.
        self.dU = self._hyperparams['dU'] = env.get_action_dim()
        self.dX = self._hyperparams['dX'] = env.get_state_dim()
        self.dO = self._hyperparams['dO'] = env.get_obs_dim()

        # Get time values from the 'hyperparams'
        self.T = self._hyperparams['T']
        self.dt = self._hyperparams['dt']

        init_traj_distr = self._hyperparams['init_traj_distr']
        init_traj_distr['x0'] = env.get_x0()  # TODO: Check if it is better get_x0() or get_state()
        init_traj_distr['dX'] = self.dX
        init_traj_distr['dU'] = self.dU
        init_traj_distr['dt'] = self.dt
        init_traj_distr['T'] = self.T
        #TODO:Temporal for testing init_pd
        init_traj_distr['dQ'] = self.dU


        # IterationData objects for each condition.
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]

        # Add dynamics if the algorithm requires fit_dynamics
        if self._hyperparams['fit_dynamics']:
            dynamics = self._hyperparams['dynamics']

        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()

            if self._hyperparams['fit_dynamics']:
                self.cur[m].traj_info.dynamics = dynamics['type'](dynamics)

            init_traj_distr = extract_condition(
                self._hyperparams['init_traj_distr'], self._cond_idx[m]
            )

            # Trajectory Distribution: init_lqr or init_pd
            self.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr)

        # Trajectory Optimization: LQR, PI2
        self.traj_opt = self._hyperparams['traj_opt']['type'](
            self._hyperparams['traj_opt']
        )

        # Cost
        if self._hyperparams['cost'] is None:
            raise AttributeError("Cost function has not been defined")

        if isinstance(type(self._hyperparams['cost']), list):
            self.cost = [
                self._hyperparams['cost'][i]['type'](self._hyperparams['cost'][i])
                for i in range(self.M)
            ]
        else:
            # Same cost function for all conditions
            self.cost = [
                self._hyperparams['cost']['type'](self._hyperparams['cost'])
                for _ in range(self.M)
            ]

        # KL step
        self.base_kl_step = self._hyperparams['kl_step']

        # ############# #
        # GPS Algorithm #
        # ############# #
        self.gps_algo = self._hyperparams['gps_algo']

        if self.gps_algo == 'pigps':
            self._hyperparams.update(default_pigps_hyperparams)

        if self.gps_algo in ['pigps', 'mdgps']:
            self._hyperparams.update(default_mdgps_hyperparams)
            policy_prior = self._hyperparams['policy_prior']

            for m in range(self.M):
                self.cur[m].pol_info = PolicyInfo(self._hyperparams)
                self.cur[m].pol_info.policy_prior = \
                    policy_prior['type'](policy_prior)

            #self.policy_opt = self._hyperparams['policy_opt']['type'](
            #    self._hyperparams['policy_opt'], self.dO, self.dU
            #)
            self.policy_opt = self.agent.policy

    def run(self, itr_load=None):
        """
        Run GPS.
        If itr_load is especified, first loads the algorithm state from that iteration
         and resumes training at the next iteration
        :param itr_load: desired iteration to load algorithm from
        :return: 
        """
        try:
            itr_start = self._initialize(itr_load)

            for itr in range(itr_start, self._hyperparams['iterations']):
                # Collect samples
                for cond in self._train_idx:
                    for i in range(self._hyperparams['num_samples']):
                        print("")
                        print("#"*40)
                        print("Sample itr:%d/%d, cond:%d/%d, i:%d/%d" % (itr+1, self._hyperparams['iterations'],
                                                                         cond+1, len(self._train_idx),
                                                                         i+1, self._hyperparams['num_samples']))
                        print("#"*40)
                        self._take_sample(itr, cond, i)

                traj_sample_lists = [
                    self.agent.get_samples(cond, -self._hyperparams['num_samples'])
                    for cond in self._train_idx
                ]

                # Clear agent samples.
                self.agent.clear_samples()

                self._take_iteration(itr, traj_sample_lists)

                if self._hyperparams['test_after_iter']:
                    pol_sample_lists = self._take_policy_samples()

                # Log data
                #self._log_data(itr, traj_sample_lists, pol_sample_lists)

        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            print("#"*30)
            print("#"*30)
            print_skull()
            print("Panic: ERROR IN GPS!!!!")
            print("#"*30)
            print("#"*30)
        finally:
            self._end()

    def _end(self):
        """ Finish running and exit. """
        print("")
        print("GPS has finished!")
        # TODO: Check how to also stop/abort the environment in case of error
        self.env.reset(time=2, conf=0)

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
            raise NotImplementedError("Initialize from itr")
            #algorithm_file = self._data_files_dir + 'algorithm_itr_%02d.pkl' % itr_load
            #self.algorithm = self.data_logger.unpickle(algorithm_file)
            #if self.algorithm is None:
            #    print("Error: cannot find '%s.'" % algorithm_file)
            #    os._exit(1) # called instead of sys.exit(), since this is in a thread

            #if self.gui:
            #    traj_sample_lists = self.data_logger.unpickle(self._data_files_dir +
            #                                                  ('traj_sample_itr_%02d.pkl' % itr_load))
            #    if self.algorithm.cur[0].pol_info:
            #        pol_sample_lists = self.data_logger.unpickle(self._data_files_dir +
            #                                                     ('pol_sample_itr_%02d.pkl' % itr_load))
            #    else:
            #        pol_sample_lists = None
            #    self.gui.set_status_text(
            #        ('Resuming training from algorithm state at iteration %d.\n' +
            #         'Press \'go\' to begin.') % itr_load)
            #return itr_load + 1

    def _take_sample(self, itr, cond, i, verbose=True, save=True, noisy=True):
        """
        Collect a sample from the environment.
        Args:
            itr: Iteration number.
            cond: Condition number.
            i: Sample number.
        Returns: None
        """
        # Parameters
        self._hyperparams['smooth_noise'] = True
        self._hyperparams['smooth_noise_var'] = 2.0
        self._hyperparams['smooth_noise_renormalize'] = True

        # On-policy or Off-policy
        if self._hyperparams['sample_on_policy'] \
                and (self.iteration_count > 0 or
                     ('sample_pol_first_itr' in self._hyperparams and self._hyperparams['sample_pol_first_itr'])):
            policy = self.agent.policy.policy  # DOM: Instead self.opt_pol.policy
            print("On-policy sampling: %s!" % type(policy))
        else:
            policy = self.cur[cond].traj_distr
            print("Off-policy sampling: %s!" % type(policy))

        #self.agent.explore(pol, cond,
        #                   verbose=(i < self._hyperparams['verbose_trials']))

        #sample(self, policy, condition, verbose=True, save=True, noisy=True)


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
        self.env.reset(time=1)
        import rospy

        ros_rate = rospy.Rate(int(1/self.dt))  # hz
        # Collect history
        for t in range(self.T):
            if verbose:
                print("Sample itr:%d/%d, cond:%d/%d, i:%d/%d | t:%d/%d" % (itr+1, self._hyperparams['iterations'],
                                                                           cond+1, len(self._train_idx),
                                                                           i+1, self._hyperparams['num_samples'],
                                                                           t+1, self.T))
            obs = self.env.get_observation()
            state = self.env.get_state()
            #action = self.agent.act(obs=obs)
            #action = policy.act(state, obs, t, noise[t, :])
            action = policy.eval(state, obs, t, noise[t, :])
            self.env.send_action(action)
            #print("Episode %d/%d | Sample:%d/%d | t=%d/%d" % (episode+1, total_episodes,
            #                                                  n_sample+1, num_samples,
            #                                                  i+1, T))
            obs_hist[t] = (obs, action)
            history[t] = (state, action)
            #print(obs)
            #print("..")
            #print(state)
            #print("--")
            #print("obs_shape:(%s)" % obs.shape)
            #print("state_shape:(%s)" % state.shape)
            #print("obs active names: %s" % bigman_env.get_obs_info()['names'])
            #print("obs active dims: %s" % bigman_env.get_obs_info()['dimensions'])
            #print("state active names: %s" % bigman_env.get_state_info()['names'])
            #print("state active dims: %s" % bigman_env.get_state_info()['dimensions'])
            #print("")

            #sample.set_acts(action, t=i)  # Set action One by one
            #sample.set_obs(obs[:42], obs_name='joint_state', t=i)  # Set action One by one
            #sample.set_states(state[:7], state_name='link_position', t=i)  # Set action One by one

            ros_rate.sleep()

        all_actions = np.array([hist[1] for hist in history])
        all_states = np.array([hist[0] for hist in history])
        all_obs = np.array([hist[0] for hist in obs_hist])
        sample.set_acts(all_actions)  # Set all actions at the same time
        sample.set_obs(all_obs)  # Set all obs at the same time
        sample.set_states(all_states)  # Set all states at the same time

        if save:  # Save sample in agent
            self.agent._samples[cond].append(sample)  # TODO: Do this with a method

        print("Plotting sample %d" % (i+1))
        plot_sample(sample, data_to_plot='actions', block=False)
        plot_sample(sample, data_to_plot='states', block=True)

        return sample

    def _take_iteration(self, itr, sample_lists):
        print("")
        print("GPS iteration %d | Using %d samples" % (itr+1, len(sample_lists)))

        if self.gps_algo == 'pigps':
            self.iteration_pigps(sample_lists)

        elif self.gps_algo == 'mdgps':
            self.iteration_mdgps(sample_lists)

        else:
            raise NotImplementedError("GPS algorithm:%s NOT IMPLEMENTED!" % self.gps_algo)


    def _take_policy_samples(self, N=None, verbose=True):
        """
        Take samples from the policy to see how it's doing.
        Args:
            N  : number of policy samples to take per condition
        Returns: None
        """
        #policy = self.agent.policy.policy  # DOM: Instead self.opt_pol.policy

        pol_samples = [[None] for _ in range(len(self._test_idx))]

        # Since this isn't noisy, just take one sample.
        # TODO: Make this noisy? Add hyperparam?
        noise = np.zeros((self.T, self.dU))

        # TODO: Take at all conditions for GUI?

        for cond in range(len(self._test_idx)):
            # Create a sample class
            # TODO: In original GPS code this is done with self._init_sample(self, condition, feature_fn=None) in agent
            sample = Sample(self.env, self.T)
            history = [None] * self.T
            obs_hist = [None] * self.T

            #print("Resetting environment...")
            #self.env.reset(time=1)
            import rospy

            ros_rate = rospy.Rate(int(1/self.dt))  # hz

            for t in range(self.T):
                if verbose:
                    print("Sampling using Agent Policy | cond:%d/%d | t:%d/%d" % (cond+1, len(self._test_idx),
                                                                                  t+1, self.T))
                obs = self.env.get_observation()
                state = self.env.get_state()
                #action = self.agent.act(obs=obs)
                #action = policy.act(state, obs, t, noise[t, :])
                action = self.agent.act(x=state, obs=obs, t=t, noise=noise[t, :])
                self.env.send_action(action)
                #print("Episode %d/%d | Sample:%d/%d | t=%d/%d" % (episode+1, total_episodes,
                #                                                  n_sample+1, num_samples,
                #                                                  i+1, T))
                obs_hist[t] = (obs, action)
                history[t] = (state, action)

                ros_rate.sleep()

            all_actions = np.array([hist[1] for hist in history])
            all_states = np.array([hist[0] for hist in history])
            all_obs = np.array([hist[0] for hist in obs_hist])
            sample.set_acts(all_actions)  # Set all actions at the same time
            sample.set_obs(all_obs)  # Set all obs at the same time
            sample.set_states(all_states)  # Set all states at the same time

            pol_samples[cond] = sample

        return [SampleList(samples) for samples in pol_samples]

    def _log_data(self, itr, traj_sample_lists, pol_sample_lists=None):
        """
        Log data and algorithm
        Args:
            itr: Iteration number.
            traj_sample_lists: trajectory samples as SampleList object
            pol_sample_lists: policy samples as SampleList object
        Returns: None
        """
        #TODO DOMINGO: Assuming we always save
        #if 'no_sample_logging' in self._hyperparams['common']:
        #    return
        self.data_logger.pickle(
            self._data_files_dir + ('%sgps_algorithm_itr_%02d.pkl' % (self.gps_algo, itr)),
            #copy.copy(self.algorithm)
        )

        self.data_logger.pickle(
            self._data_files_dir + ('traj_sample_itr_%02d.pkl' % itr),
            copy.copy(traj_sample_lists)
        )

        if pol_sample_lists:
            self.data_logger.pickle(
                self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists)
            )


    # FROM ALGORITHM CLASS!!!!
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
            self.cur[m].traj_info.x0sigma = np.diag(
                np.maximum(np.var(x0, axis=0),
                           self._hyperparams['initial_state_var'])
            )

            prior = self.cur[m].traj_info.dynamics.get_prior()
            if prior:
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = len(cur_data)
                self.cur[m].traj_info.x0sigma += \
                    Phi + (N*priorm) / (N+priorm) * \
                          np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)

    def _update_trajectories(self):
        """
        Compute new linear Gaussian controllers.
        """
        if not hasattr(self, 'new_traj_distr'):
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M)
            ]
        for cond in range(self.M):
            self.new_traj_distr[cond], self.cur[cond].eta = \
                self.traj_opt.update(cond, self)

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
            l, lx, lu, lxx, luu, lux = self.cost[cond].eval(sample)
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
        Move all 'cur' variables to 'prev', and advance iteration
        counter.
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
        state['_random_state'] = random.getstate()
        state['_np_random_state'] = np.random.get_state()
        return state

    # For unpickling.
    def __setstate__(self, state):
        self.__dict__ = state
        random.setstate(state.pop('_random_state'))
        np.random.set_state(state.pop('_np_random_state'))


    # ################### #
    # ################### #
    # ###### PIGPS ###### #
    # ################### #
    # ################### #
    """ This file defines the PIGPS algorithm. 
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
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M)
            ]
            self._update_policy_mdgps()

        # Update policy linearizations.
        for m in range(self.M):
            self._update_policy_fit_mdgps(m)

        # C-step
        self._update_trajectories()

        # S-step
        self._update_policy_mdgps()

        # Prepare for next iteration
        self._advance_iteration_variables_mdgps()



    # ################### #
    # ################### #
    # ###### MSGPS ###### #
    # ################### #
    # ################### #
    def iteration_mdgps(self, sample_lists):
        """
        Run iteration of MDGPS-based guided policy search.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        # Store the samples and evaluate the costs.
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
            self._eval_cost(m)

        # Update dynamics linearizations.
        self._update_dynamics()

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.iteration_count == 0:
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M)
            ]
            self._update_policy_mdgps()

        # Update policy linearizations.
        for m in range(self.M):
            self._update_policy_fit_mdgps(m)

        # C-step
        if self.iteration_count > 0:
            self._stepadjust_mdgps()
        self._update_trajectories()

        # S-step
        self._update_policy_mdgps()

        # Prepare for next iteration
        self._advance_iteration_variables()

    def _update_policy_mdgps(self):
        """ Compute the new policy. """
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

    def _update_policy_fit_mdgps(self, m):
        """
        Re-estimate the local policy values in the neighborhood of the
        trajectory.
        Args:
            m: Condition
        """
        dX, dU, T = self.dX, self.dU, self.T
        # Choose samples to use.
        samples = self.cur[m].sample_list
        N = len(samples)
        pol_info = self.cur[m].pol_info
        X = samples.get_states()
        obs = samples.get_obs().copy()
        pol_mu, pol_sig = self.policy_opt.prob(obs)[:2]
        pol_info.pol_mu, pol_info.pol_sig = pol_mu, pol_sig

        # Update policy prior.
        policy_prior = pol_info.policy_prior
        samples = SampleList(self.cur[m].sample_list)
        mode = self._hyperparams['policy_sample_mode']
        policy_prior.update(samples, self.policy_opt, mode)

        # Fit linearization and store in pol_info.
        pol_info.pol_K, pol_info.pol_k, pol_info.pol_S = \
            policy_prior.fit(X, pol_mu, pol_sig)
        for t in range(T):
            pol_info.chol_pol_S[t, :, :] = \
                sp.linalg.cholesky(pol_info.pol_S[t, :, :])

    def _advance_iteration_variables_mdgps(self):
        """
        Move all 'cur' variables to 'prev', reinitialize 'cur'
        variables, and advance iteration counter.
        """
        self._advance_iteration_variables(self)
        for m in range(self.M):
            self.cur[m].traj_info.last_kl_step = \
                self.prev[m].traj_info.last_kl_step
            self.cur[m].pol_info = copy.deepcopy(self.prev[m].pol_info)

    def _stepadjust_mdgps(self):
        """
        Calculate new step sizes. This version uses the same step size
        for all conditions.
        """
        # Compute previous cost and previous expected cost.
        prev_M = len(self.prev) # May be different in future.
        prev_laplace = np.empty(prev_M)
        prev_mc = np.empty(prev_M)
        prev_predicted = np.empty(prev_M)
        for m in range(prev_M):
            prev_nn = self.prev[m].pol_info.traj_distr()
            prev_lg = self.prev[m].new_traj_distr

            # Compute values under Laplace approximation. This is the policy
            # that the previous samples were actually drawn from under the
            # dynamics that were estimated from the previous samples.
            prev_laplace[m] = self.traj_opt.estimate_cost(
                prev_nn, self.prev[m].traj_info
            ).sum()
            # This is the actual cost that we experienced.
            prev_mc[m] = self.prev[m].cs.mean(axis=0).sum()
            # This is the policy that we just used under the dynamics that
            # were estimated from the prev samples (so this is the cost
            # we thought we would have).
            prev_predicted[m] = self.traj_opt.estimate_cost(
                prev_lg, self.prev[m].traj_info
            ).sum()

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
            NotImplementedError("function not implemented for %s gps" % self.gps_algo)

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
