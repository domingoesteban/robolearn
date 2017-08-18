"""
This file defines the DREPS-based trajectory optimization method.
"""
import copy
import numpy as np

from robolearn.algos.gps.temp_gps import GPS
from robolearn.algos.trajopt.trajopt_config import default_dreps_hyperparams


class DREPS(GPS):
    """ Sample-based trajectory optimization with DREPS. """
    def __init__(self, agent, env, **kwargs):
        super(DREPS, self).__init__(agent, env, **kwargs)
        gps_algo_hyperparams = default_dreps_hyperparams.copy()
        gps_algo_hyperparams.update(self._hyperparams['gps_algo_hyperparams'])
        self._hyperparams.update(gps_algo_hyperparams)

        self.gps_algo = 'dreps'

    def iteration(self, sample_lists):
        """
        Run iteration of DREPS.
        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        # Copy samples for all conditions.
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
    
        # Evaluate cost function for all conditions and samples.
        print("->Evaluating trajectories costs...")
        for m in range(self.M):
            self._eval_cost(m)

        # Re-use last L iterations
        print("->Re-using last L iterations (TODO)...")

        print("->Updating dynamics linearization (time-varying models) ...")
        self._update_dynamics()

        print("->Generate 'virtual' roll-outs (TODO)...")

            # Run inner loop to compute new policies.
        print("->Updating trajectories...")
        for ii in range(self._hyperparams['inner_iterations']):
            print("-->Inner iteration %d/%d" % (ii+1, self._hyperparams['inner_iterations']))
            self._update_trajectories()

        self._advance_iteration_variables()
