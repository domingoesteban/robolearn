"""
This file defines the PI2-based trajectory optimization method.
Author: Finn et al
Adapted by: robolearn collaborators
"""
import copy
import numpy as np

from robolearn.algos.trajopt.traj_opt import TrajOpt
from robolearn.algos.trajopt.trajopt_config import DEFAULT_PI2_HYPERPARAMS
from robolearn.utils.traj_opt.traj_opt_pi2 import TrajOptPI2


class PI2TrajOpt(TrajOpt):
    """ Sample-based trajectory optimization with PI2. """
    def __init__(self, agent, env, **kwargs):
        TrajOpt.__init__(self, agent, env, DEFAULT_PI2_HYPERPARAMS,
                         **kwargs)

        # Traj Opt (Local policy opt) method #
        # ---------------------------------- #
        if self._hyperparams['traj_opt']['type'].__name__ != 'TrajOptPI2':
            raise ValueError("The selected traj_opt method is not %s!"
                             % TrajOptPI2.__name__)
        self.traj_opt = \
            self._hyperparams['traj_opt']['type'](self._hyperparams['traj_opt'])
        self.traj_opt.set_logger(self.logger)

    def _iteration(self, itr):
        """
        Run iteration of PI2.
        """
        logger = self.logger

        # Sample from environment using current trajectory distributions
        logger.info('')
        logger.info('PI2TrajOpt: itr:%02d | Sampling from local trajectories..'
                    % (itr+1))

        traj_sample_lists = self._take_sample(itr)

        # Copy samples for all conditions.
        for m, m_train in enumerate(self._train_cond_idx):
            self.cur[m_train].sample_list = traj_sample_lists[m]

        # Evaluate cost function for all conditions and samples.
        logger.info('')
        logger.info('PI2Opt: itr:%02d | '
                    'Evaluating samples costs...' % (itr+1))
        for m in range(self.M):
            self._eval_iter_samples_cost(m)

        # Run inner loop to compute new policies.
        logger.info('')
        logger.info('PI2TrajOpt: itr:%02d | '
                    'Updating trajectories...' % (itr+1))
        for ii in range(self._hyperparams['algo_hyperparams']
                        ['inner_iterations']):
            logger.info('-PI2TrajOpt: itr:%02d | Inner iteration %d/%d'
                        % (itr+1, ii+1,
                           self._hyperparams['algo_hyperparams']
                           ['inner_iterations']))
            self._update_trajectories()

        sample_lists_costs = [None for _ in traj_sample_lists]
        sample_lists_cost_compositions = [None for _ in traj_sample_lists]
        for cc, sample_list in enumerate(traj_sample_lists):
            cost_fcn = self.cost_function[cc]
            costs = self._eval_sample_list_cost(sample_list, cost_fcn)
            sample_lists_costs[cc] = costs[0]
            sample_lists_cost_compositions[cc] = costs[2]

        for m, cond in enumerate(self._train_cond_idx):
            print('&'*10)
            print('Average Cost')
            print('Condition:%02d' % cond)
            print('Avg cost: %f'
                  % sample_lists_costs[m].sum(axis=1).mean())
            print('&'*10)

        # Log data
        self._log_iter_data(itr, traj_sample_lists, sample_lists_costs,
                            sample_lists_cost_compositions)

        # Prepare everything for next iteration
        self._advance_iteration_variables()

    def _update_trajectories(self):
        """
        Compute new linear Gaussian controllers using the TrajOpt algorithm.
        """
        LOGGER = self.logger

        LOGGER.info('-->TrajOpt: Updating trajectories (local policies)...')
        if self.new_traj_distr is None:
            self.new_traj_distr = [self.cur[cond].traj_distr
                                   for cond in range(self.M)]
        for cond in range(self.M):
            traj_opt_outputs = self.traj_opt.update(cond, self)
            self.new_traj_distr[cond] = traj_opt_outputs[0]
            self.cur[cond].eta = traj_opt_outputs[1]

    def _log_iter_data(self, itr, traj_sample_lists,
                       sample_lists_costs=None,
                       sample_lists_cost_compositions=None):
        TrajOpt._log_iter_data(self, itr, traj_sample_lists,
                               sample_lists_costs=sample_lists_costs,
                               sample_lists_cost_compositions=sample_lists_cost_compositions)
