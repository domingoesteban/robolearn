"""
Based on Haarnoja's TensorFlow SQL implementation

https://github.com/haarnoja/softqlearning
"""

import numpy as np
import torch
import torch.optim as optim
# from torch.autograd import Variable

from collections import OrderedDict

import robolearn.torch.pytorch_util as ptu
from robolearn.torch import TorchIterativeRLAlgorithm
from robolearn.core import logger, eval_util
from robolearn.policies import MakeDeterministic

EPS = 1e-6


def assert_shape(tensor, expected_shape):
    tensor_shape = list(tensor.shape)
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])


class Reinforce(TorchIterativeRLAlgorithm):
    """Reinforce Algorithm

    """
    def __init__(self,
                 env,
                 policy,

                 policy_lr=1e-3,
                 optimizer_class=optim.Adam,

                 causality=True,
                 discounted=False,

                 plotter=None,
                 eval_deterministic=True,
                 **kwargs):
        """

        Args:
            env:
            qf (`robolearn.PyTorchModule`): Q-function approximator.
            policy (`robolearn.PyTorchModule`):
            policy_lr (`float`): Learning rate used for the Policy approximator.
            plotter (`MultiQFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            eval_deterministic: Evaluate with deterministic version of current
                _i_policy.
            **kwargs:
        """
        if eval_deterministic:
            eval_policy = MakeDeterministic(policy)
        else:
            eval_policy = policy
        super(Reinforce, self).__init__(
            env=env,
            exploration_policy=policy,
            eval_policy=eval_policy,
            **kwargs
        )
        self.policy = policy

        self.plotter = plotter

        # Env data
        self._action_dim = self.env.action_space.low.size
        self._obs_dim = self.env.observation_space.low.size

        # Optimize Policy
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

        # Return computation
        self.causality = causality
        self.discounted = discounted

    def pretrain(self):
        # Match target Qfcn with current one
        self._update_target_q_fcn()

    def _do_training(self):
        # batch = self.get_batch()
        paths = self.get_exploration_paths()

        # Update Networks

        # print('n_step', self._n_train_steps_total)
        # bellman_residual = self._update_softq_fcn(paths)
        surrogate_cost = self._update_policy(paths)
        # self._update_target_softq_fcn()

        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            # self.eval_statistics['Bellman Residual (QFcn)'] = \
            #     np.mean(ptu.get_numpy(bellman_residual))
            self.eval_statistics['Surrogate Cost (Policy)'] = \
                np.mean(ptu.get_numpy(surrogate_cost))

    def _update_q_fcn(self, batch):
        """
        Q-fcn update
        Args:
            batch:

        Returns:

        """
        # TODO: Implement for AC VERSION
        pass
        # obs = batch['observations']
        # actions = batch['actions']
        # next_obs = batch['next_observations']
        # rewards = batch['rewards']
        # terminals = batch['terminals']
        # n_batch = obs.shape[0]
        #
        # # \hat Q in Equation 11
        # # ys = (self.reward_scale * rewards.squeeze() +  # Current reward
        # ys = (rewards.squeeze() +  # IT IS NOT NECESSARY TO SCALE REWARDS (ALREADY DONE)
        #       (1 - terminals.squeeze()) * self.discount * next_value  # Future return
        #       ).detach()  # TODO: CHECK IF I AM DETACHING GRADIENT!!!
        # assert_shape(ys, [n_batch])
        #
        # # Equation 11:
        # bellman_residual = 0.5 * torch.mean((ys - q_values) ** 2)
        #
        # # Gradient descent on _i_policy parameters
        # self._i_qf_optimizer.zero_grad()  # Zero all model var grads
        # bellman_residual.backward()  # Compute gradient of surrogate_loss
        # self._i_qf_optimizer.step()  # Update model vars
        #
        # return bellman_residual

    def _update_policy(self, paths):
        """
        Policy update:
        Returns:

        """

        rewards = []
        obs = []
        log_probs = []
        qs = []

        for ii, path in enumerate(paths):
            rewards.append(path['rewards'])
            obs.append(path['observations'])
            log_probs.append(self.policy.log_action(path['actions'],
                                                    path['observations']))
            qs.append(self._accum_rewards(path['rewards']))

        log_probs = torch.cat([log_prob for log_prob in log_probs])
        qs = torch.cat([q for q in qs])

        weighted_log_probs = torch.mul(log_probs, qs)

        loss = -torch.mean(weighted_log_probs)

        # Gradient descent on _i_policy parameters
        self.policy_optimizer.zero_grad()  # Zero all model var grads
        loss.backward()  # Compute gradient of surrogate_loss
        self.policy_optimizer.step()  # Update model vars

        return loss

    def _update_target_q_fcn(self):
        # Implement for AC version
        pass
        # if self.use_hard_updates:
        #     # print(self._n_train_steps_total, self.hard_update_period)
        #     if self._n_train_steps_total % self.hard_update_period == 0:
        #         ptu.copy_model_params_from_to(self._i_qf, self.target_qf)
        # else:
        #     ptu.soft_update_from_to(self._i_qf, self.target_qf,
        #                             self.soft_target_tau)

    def _accum_rewards(self, rewards, normalize=False):
        """ take 1D float array of rewards and compute discounted reward """

        if self.causality:
            discounted_r = ptu.zeros_like(rewards)
            T = rewards.shape[0]
            running_add = 0
            for t in reversed(range(0, T)):
                if self.discounted:
                    gamma = self.discount
                else:
                    gamma = 1
                running_add = rewards[t, :] + running_add*gamma
                discounted_r[t, :] = running_add

            if normalize:
                discounted_r = self._normalize(discounted_r)

        else:
            discounted_r = torch.sum(rewards, dim=0, keepdim=True)

        return discounted_r

    @staticmethod
    def _normalize(data, mean=0.0, std=1.0):
        n_data = (data - torch.mean(data)) / (torch.std(data) + 1e-8)
        return n_data * (std + 1e-8) + mean

    @property
    def networks(self):
        return [
            self.policy,
        ]

    def get_epoch_snapshot(self, epoch):
        if self.plotter is not None:
            self.plotter.draw()

        snapshot = super(Reinforce, self).get_epoch_snapshot(epoch)
        snapshot.update(
            policy=self.eval_policy,
            trained_policy=self.policy,
        )
        return snapshot

    def evaluate(self, epoch):
        # TODO: AT THIS MOMENT THIS CODE IS THE SAME THAN SUPER
        statistics = OrderedDict()
        statistics.update(self.eval_statistics)
        self.eval_statistics = None

        logger.log("Collecting samples for evaluation")
        test_paths = self.eval_sampler.obtain_samples()

        statistics.update(eval_util.get_generic_path_information(
            test_paths, stat_prefix="Test"))
        statistics.update(eval_util.get_generic_path_information(
            self._exploration_paths, stat_prefix="Exploration",
        ))
        if hasattr(self.env, "log_diagnostics"):
            print('TODO: WE NEED LOG_DIAGNOSTICS IN ENV')
            self.env.log_diagnostics(test_paths)

        average_returns = eval_util.get_average_returns(test_paths)
        statistics['Average Test Return'] = average_returns

        # Record the data
        for key, value in statistics.items():
            logger.record_tabular(key, value)

        if self.render_eval_paths:
            self.env.render_paths(test_paths)

        if self.plotter is not None:
            self.plotter.draw()
