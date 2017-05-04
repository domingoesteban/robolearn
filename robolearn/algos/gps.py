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

# Algorithm
default_gps_hyperparams = {
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
    'init_traj_distr': None,  # A list of initial LinearGaussianPolicy
    # objects for each condition.
    # Trajectory optimization.
    'traj_opt': None,
    # Weight of maximum entropy term in trajectory optimization.
    'max_ent_traj': 0.0,
    # Dynamics hyperaparams.
    'dynamics': None,
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

        # From original GPS code
        self._conditions = self._hyperparams['conditions']
        if 'train_conditions' in self._hyperparams['common']:
            self._train_idx = self._hyperparams['train_conditions']
            self._test_idx = self._hyperparams['test_conditions']
        else:
            self._train_idx = range(self._conditions)
            self._hyperparams['train_conditions'] = self._hyperparams['conditions']
            self._test_idx = self._train_idx


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

        except Exception as e:
            traceback.print_exception(*sys.exc_info())

        finally:
            self._end()

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
