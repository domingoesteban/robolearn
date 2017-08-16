"""
# ################### #
# ################### #
# ###### PIGPS ###### #
# ################### #
# ################### #
PIGPS algorithm. 
Authors: C.Finn et al
Reference:
Y. Chebotar, M. Kalakrishnan, A. Yahya, A. Li, S. Schaal, S. Levine. 
Path Integral Guided Policy Search. 2016. https://arxiv.org/abs/1610.00529.
"""

from robolearn.algos.gps.mdgps import MDGPS
from robolearn.algos.gps.gps_config import default_pigps_hyperparams


class PIGPS(MDGPS):
    def __init__(self, agent, env, **kwargs):
        super(PIGPS, self).__init__(agent, env, **kwargs)

        gps_algo_hyperparams = default_pigps_hyperparams.copy()
        gps_algo_hyperparams.update(self._hyperparams['gps_algo_hyperparams'])
        self._hyperparams.update(gps_algo_hyperparams)

        self.gps_algo = 'pi2gps'

    def iteration(self, sample_lists):
        """
        Run iteration of PI-based guided policy search.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        # Store the samples and evaluate the costs.
        print('->Evaluating samples costs...')
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
            self._eval_cost(m)

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.iteration_count == 0:
            self.new_traj_distr = [self.cur[cond].traj_distr for cond in range(self.M)]
            self.update_policy()

        # Update global policy linearizations.
        print('->Updating global policy linearization...')
        for m in range(self.M):
            self.update_policy_fit(m)

        # C-step
        print('->| C-step |<-')
        self._update_trajectories()

        # S-step
        print('->| S-step |<-')
        self.update_policy()

        # Prepare for next iteration
        self.advance_iteration_variables()
