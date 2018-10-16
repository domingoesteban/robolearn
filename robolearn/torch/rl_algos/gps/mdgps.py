"""
Based on Fins GPS implementation
TODO: ALSO MONTGOMORY

https://github.com/cbfinn/gps
"""

import numpy as np
import torch
import torch.optim as optim
# from torch.autograd import Variable
import copy
import gtimer as gt

from collections import OrderedDict

import robolearn.torch.pytorch_util as ptu
from robolearn.torch.rl_algos.torch_iterative_rl_algorithm \
    import TorchIterativeRLAlgorithm
from robolearn.core import logger, eval_util
from robolearn.policies import MakeDeterministic
from robolearn.utils.samplers.exploration_rollout import exploration_rollout
from robolearn.utils.data_management import PathBuilder
from robolearn.policies.base import ExplorationPolicy
from robolearn.utils.exploration_strategies import SmoothNoiseStrategy

EPS = 1e-6


def assert_shape(tensor, expected_shape):
    tensor_shape = list(tensor.shape)
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])


class MDGPS(TorchIterativeRLAlgorithm):
    """MDGPS Algorithm

    """
    def __init__(self,
                 local_policies,
                 global_policy,
                 *args,
                 **kwargs):
        """
        MDGPS
        """
        env = kwargs['env']
        self.local_policies_wrapper = LocalPolWrapper(local_policies, env)
        self.global_policy = global_policy

        # MDGPS hyperparameters
        self._traj_opt_inner_iters = kwargs.pop('traj_opt_inner_iters', 1)
        self._train_cond_idxs = kwargs.pop('train_cond_idxs', [0])
        self._test_cond_idxs = kwargs.pop('test_cond_idxs', [0])

        super(MDGPS, self).__init__(
            exploration_policy=self.local_policies_wrapper,
            eval_policy=self.global_policy,
            *args,
            **kwargs
        )

    def train(self, start_epoch=0):
        # Get snapshot of initial stuff
        if start_epoch == 0:
            self.training_mode(False)
            params = self.get_epoch_snapshot(-1)
            logger.save_itr_params(-1, params)

        self._n_env_steps_total = start_epoch * self.num_train_steps_per_epoch

        gt.reset()
        gt.set_def_unique(False)

        for epoch in gt.timed_for(
                range(start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self._start_epoch(epoch)

            n_policies = self.exploration_policy.n_policies
            for cond in self._train_cond_idxs:
                for _ in range(int(self.rollouts_per_epoch/n_policies)):
                    self._current_path_builder = PathBuilder()

                    path = exploration_rollout(self.env,
                                               self.exploration_policy,
                                               max_path_length=self.max_path_length,
                                               animated=self._render,
                                               deterministic=None,
                                               condition=cond)
                    self._handle_path(path)
                    self._n_env_steps_total += len(path['observations'])

            # Iterative learning step
            gt.stamp('sample')
            self._try_to_train()
            gt.stamp('train')

            # Evaluate if requirements are met
            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch()

    def _do_training(self):

        print("Getting exploration paths...")
        exploration_paths = self.get_exploration_paths()

        self._update_dynamics_models()

        self._compute_samples_cost()

        if self._n_train_steps_total == 0:
            print("Updating the policy for the first time")
            self._update_policy()

        self._update_policy_linearization()

        if self._n_train_steps_total > 0:
            self._update_kl_step_size()

        # C-step
        for ii in range(self._traj_opt_inner_iters):
            print("TrajOpt inner_iter %02d" % ii)
            self._update_trajectories()

        # S-step
        self._update_policy()

    def _update_dynamics_models(self):
        print("Update dynamics model")
        pass

    def _compute_samples_cost(self):
        print("Evaluate samples costs")
        pass

    def _update_policy(self):
        print("Updating the policy")
        pass

    def _update_policy_linearization(self):
        print("Update policy linearizations")
        pass

    def _update_kl_step_size(self):
        print("Update KL step size")
        pass

    def _update_trajectories(self):
        print("Update trajectories")
        pass

    @property
    def torch_models(self):
        return [
            self.global_policy,
        ]

    def evaluate(self, epoch):
        # Create a new eval_statistics
        statistics = OrderedDict()

        # Update from previous eval_statisics
        if self.eval_statistics is not None:
            statistics.update(self.eval_statistics)
        self.eval_statistics = None

        logger.log("Collecting samples for evaluation")
        test_paths = self.eval_sampler.obtain_samples()

        statistics.update(eval_util.get_generic_path_information(
            test_paths, stat_prefix="Test",
        ))
        statistics.update(eval_util.get_generic_path_information(
            self._exploration_paths, stat_prefix="Exploration",
        ))
        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths)

        average_returns = eval_util.get_average_returns(test_paths)
        statistics['AverageReturn'] = average_returns
        for key, value in statistics.items():
            logger.record_tabular(key, value)


class LocalPolWrapper(ExplorationPolicy):
    def __init__(self, local_policies, env, noisy=True, sigma=5.0,
                 sigma_scale=1.0):
        self._local_policies = local_policies
        self._current_pol_idx = None

        action_dim = self._local_policies[-1].action_dim
        ExplorationPolicy.__init__(self,
                                   action_dim=action_dim)

        self._T = self._local_policies[-1].H

        self._noisy = noisy
        self.es = SmoothNoiseStrategy(env.action_space,
                                      horizon=self._T,
                                      smooth=True,
                                      renormalize=True,
                                      sigma=sigma,
                                      sigma_scale=[sigma_scale]*action_dim)

        self._noise = torch.zeros((self.n_policies, self._T, self.action_dim))

    def reset(self, condition=None):
        self._current_pol_idx = condition
        self._current_time = 0

        self.es.reset()

    def get_action(self, *args, **kwargs):
        local_policy = self._local_policies[self._current_pol_idx]
        kwargs['t'] = self._current_time
        self._current_time += 1
        if self._noisy:
            return self.es.get_action(local_policy, *args, **kwargs)
        else:
            return local_policy.get_action(*args, **kwargs)

    @property
    def n_policies(self):
        return len(self._local_policies)

    @property
    def horizon(self):
        return self._T
