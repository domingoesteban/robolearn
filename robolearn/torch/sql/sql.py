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
from robolearn.torch.torch_rl_algorithm import TorchRLAlgorithm
from robolearn.core import logger, eval_util
from robolearn.utils.samplers import InPlacePathSampler
from robolearn.torch.sql.policies import MakeDeterministic
from robolearn.torch.sql.kernel import adaptive_isotropic_gaussian_kernel
from robolearn.torch.ops import log_sum_exp

EPS = 1e-6


def assert_shape(tensor, expected_shape):
    tensor_shape = list(tensor.shape)
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])


class SQL(TorchRLAlgorithm):
    """Soft Q-learning (SQL).


    Reference:
        [1] Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine,
        "Reinforcement Learning with Deep Energy-Based Policies," International
        Conference on Machine Learning, 2017. https://arxiv.org/abs/1702.08165
    """
    def __init__(self,
                 env,
                 qf,
                 policy,

                 qf_lr=1e-3,
                 policy_lr=1e-3,
                 optimizer_class=optim.Adam,
                 use_hard_updates=False,
                 hard_update_period=1000,
                 soft_target_tau=0.001,

                 value_n_particles=16,
                 kernel_fn=adaptive_isotropic_gaussian_kernel,
                 kernel_n_particles=16,
                 kernel_update_ratio=0.5,
                 plotter=None,
                 eval_deterministic=True,
                 **kwargs):
        """

        Args:
            env:
            qf (`robolearn.PyTorchModule`): Q-function approximator.
            policy (`robolearn.PyTorchModule`):
            qf_lr (`float`): Learning rate used for the Q-function approximator.
            use_hard_updates (`bool`): Use a hard rather than soft update.
            hard_update_period (`int`): How many gradient steps before copying
                the parameters over. Used if `use_hard_updates` is True.
            soft_target_tau (`float`): Soft target tau to update target QF.
                Used if `use_hard_updates` is False.
            value_n_particles (`int`): The number of action samples used for
                estimating the value of next state.
            kernel_fn (function object): A function object that represents
                a kernel function.
            kernel_n_particles (`int`): Total number of particles per state
                used in SVGD updates.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            eval_deterministic: Evaluate with deterministic version of current
                _i_policy.
            **kwargs:
        """
        if eval_deterministic:
            eval_policy = MakeDeterministic(policy)
        else:
            eval_policy = policy
        super().__init__(
            env=env,
            exploration_policy=policy,
            eval_policy=eval_policy,
            **kwargs
        )
        self.policy = policy
        self.qf = qf
        self.target_qf = self.qf.copy()
        # if ptu.gpu_enabled():
        #     self.target_qf.cuda()

        self.plotter = plotter

        # Env data
        self._action_dim = self.env.action_space.low.size
        self._obs_dim = self.env.observation_space.low.size

        # Optimize Q-fcn
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=qf_lr,
        )
        self._value_n_particles = value_n_particles

        # Optimize Policy
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

        self._kernel_n_particles = kernel_n_particles
        self._kernel_update_ratio = kernel_update_ratio
        self._kernel_fn = kernel_fn

        # Optimize target Q-fcn
        self.use_hard_updates = use_hard_updates
        self.hard_update_period = hard_update_period
        self.soft_target_tau = soft_target_tau

    def pretrain(self):
        # Math target Qfcn with current one
        self._update_target_q_fcn()

    def _do_training(self):
        batch = self.get_batch()

        # Update Networks

        # print('n_step', self._n_train_steps_total)
        bellman_residual = self._update_q_fcn(batch)
        surrogate_cost = self._update_policy(batch)
        self._update_target_q_fcn()

        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics['Bellman Residual (QFcn)'] = \
                np.mean(ptu.get_numpy(bellman_residual))
            self.eval_statistics['Surrogate Cost (Policy)'] = \
                np.mean(ptu.get_numpy(surrogate_cost))

    def _update_q_fcn(self, batch):
        """
        Q-fcn update
        Args:
            batch:

        Returns:

        """
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        rewards = batch['rewards']
        terminals = batch['terminals']
        n_batch = obs.shape[0]

        # The value of the next state is approximated with uniform samples.
        uniform_dist = torch.distributions.Uniform(ptu.FloatTensor([-1.0]),
                                                   ptu.FloatTensor([1.0]))
        target_actions = uniform_dist.sample((self._value_n_particles,
                                              self._action_dim)).squeeze()
        # target_actions = (-1 - 1) * ptu.Variable(torch.rand(self._value_n_particles,
        #                                        self._action_dim)) \
        #                  + 1

        q_value_targets = \
            self.target_qf(next_obs.unsqueeze(1).expand(n_batch,
                                                        self._value_n_particles,
                                                        self._obs_dim),
                           target_actions.unsqueeze(0).expand(n_batch,
                                                              self._value_n_particles,
                                                              self._action_dim)).squeeze()
        assert_shape(q_value_targets, [n_batch, self._value_n_particles])

        q_values = self.qf(obs, actions).squeeze()
        assert_shape(q_values, [n_batch])

        # Equation 10: 'Empirical' Vsoft
        next_value = log_sum_exp(q_value_targets.squeeze(), dim=1)
        assert_shape(next_value, [n_batch])

        # Importance _weights add just a constant to the value.
        next_value -= torch.log(ptu.FloatTensor([self._value_n_particles]))
        next_value += self._action_dim * np.log(2)

        # \hat Q in Equation 11
        # ys = (self.reward_scale * rewards.squeeze() +  # Current reward
        ys = (rewards.squeeze() +  # IT IS NOT NECESSARY TO SCALE REWARDS (ALREADY DONE)
              (1 - terminals.squeeze()) * self.discount * next_value  # Future return
              ).detach()  # TODO: CHECK IF I AM DETACHING GRADIENT!!!
        assert_shape(ys, [n_batch])

        # Equation 11:
        bellman_residual = 0.5 * torch.mean((ys - q_values) ** 2)

        # Gradient descent on _i_policy parameters
        self.qf_optimizer.zero_grad()  # Zero all model var grads
        bellman_residual.backward()  # Compute gradient of surrogate_loss
        self.qf_optimizer.step()  # Update model vars

        return bellman_residual

    def _update_policy(self, batch):
        """
        Policy update: SVGD
        Returns:

        """
        obs = batch['observations']
        next_obs = batch['next_observations']
        n_batch = obs.shape[0]

        actions = self.policy(obs.unsqueeze(1).expand(n_batch,
                                                      self._kernel_n_particles,
                                                      self._obs_dim))
        # actions = actions[0]  # For policies that return tuple
        assert_shape(actions,
                     [n_batch, self._kernel_n_particles, self._action_dim])

        # SVGD requires computing two empirical expectations over actions
        # (see Appendix C1.1.). To that end, we first sample a single set of
        # actions, and later split them into two sets: `fixed_actions` are used
        # to evaluate the expectation indexed by `j` and `updated_actions`
        # the expectation indexed by `i`.
        n_updated_actions = \
            int(self._kernel_n_particles*self._kernel_update_ratio)
        n_fixed_actions = self._kernel_n_particles - n_updated_actions

        fixed_actions, updated_actions \
            = torch.split(actions, [n_fixed_actions, n_updated_actions], dim=1)
        # Equiv: fixed_actions = tf.stop_gradient(fixed_actions)
        fixed_actions = ptu.Variable(fixed_actions.detach(), requires_grad=True)
        assert_shape(fixed_actions,
                     [n_batch, n_fixed_actions, self._action_dim])
        assert_shape(updated_actions,
                     [n_batch, n_updated_actions, self._action_dim])

        svgd_target_values = \
            (self.qf(next_obs.unsqueeze(1).expand(n_batch,
                                                  n_fixed_actions,
                                                  self._obs_dim),
                     fixed_actions)).squeeze()

        # Target log-density. Q_soft in Equation 13:
        squash_correction = torch.sum(torch.log(1 - fixed_actions**2 + EPS),
                                      dim=-1)
        log_p = svgd_target_values + squash_correction

        # Backward log_p
        grad_log_p = torch.autograd.grad(log_p,
                                         fixed_actions,
                                         grad_outputs=torch.ones_like(log_p),
                                         create_graph=False)[0]
        grad_log_p = torch.unsqueeze(grad_log_p, dim=2)
        assert_shape(grad_log_p,
                     [n_batch, n_fixed_actions, 1, self._action_dim])

        kernel_dict = self._kernel_fn(xs=fixed_actions,
                                      ys=updated_actions)

        # Kernel function in Eq. 13:
        kappa = torch.unsqueeze(kernel_dict['output'], dim=3)
        assert_shape(kappa,
                     [n_batch, n_fixed_actions, n_updated_actions, 1])

        # Stein Variational Gradient in Eq. 13:
        action_gradients = \
            torch.mean(kappa * grad_log_p + kernel_dict['gradient'], dim=1)
        assert_shape(action_gradients,
                     [n_batch, n_updated_actions, self._action_dim])

        # Propagate the gradient through the _i_policy network (Equation 14).
        gradients = torch.autograd.grad(updated_actions,
                                        self.policy.parameters(),
                                        grad_outputs=action_gradients,
                                        create_graph=False)

        # TODO: Check a better way to do this
        for pp, (w, g) in enumerate(zip(self.policy.parameters(), gradients)):
            if pp == 0:
                surrogate_loss = torch.sum(w*g)
            else:
                surrogate_loss += torch.sum(w*g)

        # Gradient descent on _i_policy parameters
        self.policy_optimizer.zero_grad()  # Zero all model var grads
        (-surrogate_loss).backward()  # Compute gradient of surrogate_loss
        self.policy_optimizer.step()  # Update model vars

        return -surrogate_loss

    def _update_target_q_fcn(self):
        if self.use_hard_updates:
            # print(self._n_train_steps_total, self.hard_update_period)
            if self._n_train_steps_total % self.hard_update_period == 0:
                ptu.copy_model_params_from_to(self.qf, self.target_qf)
        else:
            ptu.soft_update_from_to(self.qf, self.target_qf,
                                    self.soft_target_tau)

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.target_qf,
        ]

    def get_epoch_snapshot(self, epoch):
        if self.plotter is not None:
            self.plotter.draw()

        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            qf=self.qf,
            policy=self.eval_policy,
            trained_policy=self.policy,
            target_qf=self.target_qf
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
