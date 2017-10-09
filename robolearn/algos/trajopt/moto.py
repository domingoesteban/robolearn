"""
This file defines the MOTO trajectory optimization method.
"""
import copy
import numpy as np

from robolearn.algos.gps.gps import GPS
from robolearn.algos.trajopt.trajopt_config import DEFAULT_MOTO_HYPERPARAMS


class MOTO(GPS):
    """ Sample-based trajectory optimization with MOTO. """
    def __init__(self, agent, env, **kwargs):
        super(MOTO, self).__init__(agent, env, **kwargs)
        gps_algo_hyperparams = DEFAULT_MOTO_HYPERPARAMS.copy()
        gps_algo_hyperparams.update(self._hyperparams['gps_algo_hyperparams'])
        self._hyperparams.update(gps_algo_hyperparams)

        self.gps_algo = 'pi2'

    def iteration(self, sample_lists):
        """
        Run iteration of PI2.
        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        # Copy samples for all conditions.
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
    
        # Evaluate cost function for all conditions and samples.
        for m in range(self.M):
            self._eval_cost(m)            

        # Run inner loop to compute new policies.
        for _ in range(self._hyperparams['inner_iterations']):
            self._update_trajectories()

        self._advance_iteration_variables()
