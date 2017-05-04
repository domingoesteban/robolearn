"""
GPS
Authors: Finn et al
Adapted by robolearn collaborators
"""

import os
import sys
import traceback
import numpy as np

from robolearn.agents.agent import Agent
from robolearn.envs.environment import Environment

from robolearn.algos.algorithm import RLAlgorithm
from robolearn.policies.lin_gauss_init import init_lqr, init_pd
from robolearn.utils.gps_utils import IterationData, TrajectoryInfo, extract_condition
from robolearn.utils.dynamics_lr_prior import DynamicsLRPrior
from robolearn.utils.dynamics_prior_gmm import DynamicsPriorGMM

# Algorithm
default_gps_hyperparams = {
    'conditions': 1,  # Number of initial conditions

    #'init_traj_distr': None,  # A list of initial LinearGaussianPolicy objects for each condition.
    'init_traj_distr': {
        'type': init_lqr,  # init_pd
        #'init_gains':  1.0 / PR2_GAINS,
        #'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
        'init_var': 1.0,
        'stiffness': 0.5,
        'stiffness_vel': 0.25,
        'final_weight': 50,
        #'dt': agent['dt'],
        #'T': agent['T']
        },

    # Dynamics hyperaparams.
    #'dynamics': None,
    'dynamics': {
        'type': DynamicsLRPrior,
        'regularization': 1e-6,
        'prior': {
            'type': DynamicsPriorGMM,
            'max_clusters': 20,
            'min_samples_per_cluster': 40,
            'max_samples': 20,
            },
    },


    # DEFAULT FROM ORIGINAL CODE
    'inner_iterations': 1,  # Number of iterations.
    'min_eta': 1e-5,  # Minimum initial lagrange multiplier in DGD for
    # trajectory optimization.
    'kl_step': 0.2,
    'min_step_mult': 0.01,
    'max_step_mult': 10.0,
    'min_mult': 0.1,
    'max_mult': 5.0,
    # Trajectory settings.
    'initial_state_var': 1e-6,
    # Trajectory optimization.
    'traj_opt': None,
    # Weight of maximum entropy term in trajectory optimization.
    'max_ent_traj': 0.0,
    # Costs.
    'cost': None,  # A list of Cost objects for each condition.
    # Whether or not to sample with neural net policy (only for badmm/mdgps).
    'sample_on_policy': False,
    # Inidicates if the algorithm requires fitting of the dynamics.
    'fit_dynamics': True,

}


class GPS(RLAlgorithm):
    def __init__(self, agent=None, env=None, **kwargs):
        super(RLAlgorithm, self).__init__(default_gps_hyperparams, kwargs)

        if not issubclass(type(agent), Agent):
            raise TypeError("Wrong Agent type for agent argument.")
        self.agent = agent

        if not issubclass(type(env), Environment):
            raise TypeError("Wrong Environment type for environment argument")
        self.env = env

        # ###########################
        # From original GPS-Main code
        self._conditions = self._hyperparams['conditions']
        if 'train_conditions' in self._hyperparams:
            self._train_idx = self._hyperparams['train_conditions']
            self._test_idx = self._hyperparams['test_conditions']
        else:
            self._train_idx = range(self._conditions)
            self._hyperparams['train_conditions'] = self._train_idx
            self._test_idx = self._train_idx

        # ################################
        # From original GPS-Algorithm code
        if 'train_conditions' in self._hyperparams:
            self._cond_idx = self._hyperparams['train_conditions']
            self.M = len(self._cond_idx)
        else:
            self.M = self._hyperparams['conditions']
            self._cond_idx = range(self.M)
            self._hyperparams['train_conditions'] = self._cond_idx
            self._hyperparams['test_conditions'] = self._cond_idx
        self.iteration_count = 0

        # Grab a few values from the agent.
        self.T = self._hyperparams['T']  # Instead agent.T
        self.dU = self._hyperparams['dU'] = agent.act_dim
        self.dX = self._hyperparams['dX'] = agent.state_dim
        self.dO = self._hyperparams['dO'] = agent.obs_dim

        init_traj_distr = self._hyperparams['init_traj_distr']
        init_traj_distr['x0'] = env.get_x0  # Instead agent.x0
        init_traj_distr['dX'] = self.dX
        init_traj_distr['dU'] = self.dU

        # IterationData objects for each condition.
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]

        if self._hyperparams['fit_dynamics']:
            dynamics = self._hyperparams['dynamics']

        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            if self._hyperparams['fit_dynamics']:
                self.cur[m].traj_info.dynamics = dynamics['type'](dynamics)
            init_traj_distr = extract_condition(
                self._hyperparams['init_traj_distr'], self._cond_idx[m]
            )
            self.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr)

        self.traj_opt = self._hyperparams['traj_opt']['type'](
            self._hyperparams['traj_opt']
        )
        if type(self._hyperparams['cost']) == list:
            self.cost = [
                self._hyperparams['cost'][i]['type'](self._hyperparams['cost'][i])
                for i in range(self.M)
            ]
        else:
            self.cost = [
                self._hyperparams['cost']['type'](self._hyperparams['cost'])
                for _ in range(self.M)
            ]
        self.base_kl_step = self._hyperparams['kl_step']


    def run(self, itr_load=None):
        """
        Run GPS.
        If itr_load is especified, first loads the algorithm state from that iteration
         and resumes training at the next iteration
        :param itr_load: desired iteration to load algorithm from
        :return: 
        """
        itr_start = self._initialize(itr_load)

        for itr in range(itr_start, self._hyperparams['iterations']):
            # Collect samples
            for cond in self._train_idx:
                for i in range(self._hyperparams['num_samples']):
                    print("Sample itr:%d, cond:%d, i:%d" % (itr, cond, i))
                    self._take_sample(itr, cond, i)

            traj_sample_lists = [
                self.agent.get_samples(cond, -self._hyperparams['num_samples'])
                for cond in self._train_idx
            ]

            # Clear agent samples.
            self.agent.clear_samples()

            self._take_iteration(itr, traj_sample_lists)
            pol_sample_lists = self._take_policy_samples()
            self._log_data(itr, traj_sample_lists, pol_sample_lists)

        self._end()
        #finally:
        #    self._end()

    @staticmethod
    def _end():
        """ Finish running and exit. """
        print("")
        print("Training complete.")

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
            print('Starting from zero!')
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

    def _take_sample(self, itr, cond, i):
        """
        Collect a sample from the agent.
        Args:
            itr: Iteration number.
            cond: Condition number.
            i: Sample number.
        Returns: None
        """

        # On-policy or Off-policy
        if self._hyperparams['sample_on_policy'] \
                and (self.iteration_count > 0 or
                     ('sample_pol_first_itr' in self._hyperparams and self._hyperparams['sample_pol_first_itr'])):
            pol = self.agent.policy  # DOM: Instead self.opt_pol.policy
        else:
            pol = self.cur[cond].traj_distr

        self.agent.explore(pol, cond,
                           verbose=(i < self._hyperparams['verbose_trials']))




    # FROM ALGORITHM CLASS!!!!



