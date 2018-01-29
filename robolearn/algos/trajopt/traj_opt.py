import numpy as np
import datetime
import copy
from robolearn.algos.algorithm import Algorithm

from robolearn.utils.sample.sample import Sample
from robolearn.utils.sample.sample_list import SampleList

from robolearn.algos.gps.gps_utils import IterationData, extract_condition
from robolearn.algos.gps.gps_utils import TrajectoryInfo
from robolearn.agents.agent_utils import generate_noise
from robolearn.utils.data_logger import DataLogger


class TrajOpt(Algorithm):
    def __init__(self, agent, env, default_hyperparams, **kwargs):
        super(TrajOpt, self).__init__(default_hyperparams, kwargs)
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
        else:
            self._train_cond_idx = self._test_cond_idx = list(range(self.M))
            self._hyperparams['train_conditions'] = self._train_cond_idx

        # Number of initial conditions
        self.M = len(self._train_cond_idx)

        # Log and Data files
        if 'data_files_dir' in self._hyperparams:
            self._data_files_dir = self._hyperparams['data_files_dir']
        else:
            self._data_files_dir = type(self).__name__ + '_' + \
                                   str(datetime.datetime.now().
                                       strftime("%Y-%m-%d_%H:%M:%S"))
        self.data_logger = DataLogger(self._data_files_dir)
        self.logger = self._setup_logger('log_dualtrajopt',
                                         self._data_files_dir,
                                         '/log_dualtrajopt.log',
                                         also_screen=True)

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
        total_conditions = self._train_cond_idx
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


    def _iteration(self, **kwargs):
        raise NotImplementedError

    def _take_sample(self, itr):
        """
        Collect a sample from the environment.
        :param itr: Current TrajOpt iteration
        :return:
        """
        # If 'pol' sampling, do it with zero noise
        zero_noise = np.zeros((self.T, self.dU))

        self.logger.info("Sampling with trajectory distribution")

        conditions = self._train_cond_idx
        total_samples = self._hyperparams['num_samples']

        save = False  # Add sample to agent sample list

        sample_lists = list()

        for cond in range(len(conditions)):
            samples = list()

            policy = self.cur[cond].traj_distr
            self.logger.info("Sampling with %s policy!" % type(policy).__name__)
            for i in range(total_samples):
                noise = self.noise_data[itr, cond, i, :, :]

                self.env.reset(condition=cond)
                sample_text = "TrajOpt sampling | itr:%d/%d, " \
                              "cond:%d/%d, s:%d/%d" \
                              % (itr+1, self.max_iterations,
                                 cond+1, len(conditions),
                                 i+1, total_samples)

                self.logger.info(sample_text)
                sample = self.agent.sample(self.env, cond, self.T,
                                           self.dt, noise, policy=policy,
                                           save=save)
                samples.append(sample)

            sample_lists.append(SampleList(samples))

        return sample_lists

    def _update_trajectories(self):
        raise NotImplementedError

    def _eval_iter_samples_cost(self, cond):
        """
        Evaluate costs for all current samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        sample_list = self.cur[cond].sample_list
        cost_fcn = self.cost_function[cond]

        true_cost, cost_estimate, _ = self._eval_sample_list_cost(sample_list,
                                                                  cost_fcn)
        self.cur[cond].cs = true_cost  # True value of cost.

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

    def _update_step_size(self):
        """ Adjust the step size. """
        # Adjust step size relative to the previous iteration.
        for m in range(self.M):
            if self.iteration_count >= 1 and self.prev[m].sample_list:
                self._stepadjust(m)

    def _stepadjust(self, m):
        """
        Calculate new step sizes.
        Args:
            m: Condition
        """
        # traj_dist to obtain Ks, traj_info to obtain dynamics.

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

        self.logger.info('Trajectory step: ent: %f cost: %f -> %f',
                         ent, previous_mc_obj, new_mc_obj)

        # Compute predicted and actual improvement.
        predicted_impr = np.sum(previous_laplace_obj) - \
                         np.sum(new_predicted_laplace_obj)
        actual_impr = np.sum(previous_laplace_obj) - \
                      np.sum(new_actual_laplace_obj)

        # Print improvement details.
        self.logger.info('Previous cost: Laplace: %f MC: %f',
                         np.sum(previous_laplace_obj), previous_mc_obj)
        self.logger.info('Predicted new cost: Laplace: %f MC: %f',
                         np.sum(new_predicted_laplace_obj), new_mc_obj)
        self.logger.info('Actual new cost: Laplace: %f MC: %f',
                         np.sum(new_actual_laplace_obj), new_mc_obj)
        self.logger.info('Predicted/actual improvement: %f / %f',
                         predicted_impr, actual_impr)

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
            self.logger.info('Increasing step size multiplier to %f', new_step)
        else:
            self.logger.info('Decreasing step size multiplier to %f', new_step)

    def _measure_ent(self, m):
        """ Measure the entropy of the current trajectory. """
        ent = 0
        for t in range(self.T):
            ent = ent + np.sum(
                np.log(np.diag(self.cur[m].traj_distr.chol_pol_covar[t, :, :]))
            )
        return ent

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
        self.new_traj_distr = None

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
        LOGGER = self.logger
        LOGGER.warning('*'*20)
        LOGGER.warning('NO LOGGING AGENT, POL AND ALGO DATAAAAAAAAAAAAAAAA')
        LOGGER.warning('*'*20)

        dir_path = self.data_logger.dir_path + ('/itr_%02d' % itr)

        # LOGGER.info("Logging Agent... ")
        # self.data_logger.pickle(
        #     ('dualgps_itr_%02d.pkl' % itr),
        #     # copy.copy(temp_dict)
        #     copy.copy(self.agent)
        # )

        # print("TODO: CHECK HOW TO SOLVE LOGGING DUAL ALGO")
        # # print("Logging TrajOpt algorithm state... ")
        # # self.data_logger.pickle(
        # #     ('%s_algorithm_itr_%02d.pkl' % (type(self).__name__, itr)),
        # #     copy.copy(self)
        # # )

        LOGGER.info("Logging TrajOpt iteration data... ")
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

        if sample_lists_costs is not None:
            LOGGER.info("Logging Samples costs... ")
            self.data_logger.pickle(
                ('sample_cost_itr_%02d.pkl' % itr),
                copy.copy(sample_lists_costs),
                dir_path=dir_path
            )

        if sample_lists_cost_compositions is not None:
            LOGGER.info("Logging Samples cost compositions... ")
            self.data_logger.pickle(
                ('sample_cost_composition_itr_%02d.pkl' % itr),
                copy.copy(sample_lists_cost_compositions),
                dir_path=dir_path
            )
