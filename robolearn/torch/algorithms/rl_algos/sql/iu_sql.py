"""
Based on Haarnoja's TensorFlow SQL implementation

https://github.com/haarnoja/softqlearning
"""

import numpy as np
import torch
import torch.optim as optim

from collections import OrderedDict

import robolearn.torch.utils.pytorch_util as ptu
from robolearn.torch.algorithms.rl_algos.torch_incremental_rl_algorithm import TorchIncrementalRLAlgorithm
from robolearn.utils import eval_util
from robolearn.utils.logging import logger
from robolearn.utils.samplers import InPlacePathSampler
from robolearn.torch.sql.policies import MakeDeterministic
from robolearn.torch.sql.kernel import adaptive_isotropic_gaussian_kernel
from robolearn.torch.utils.ops import log_sum_exp

EPS = 1e-6


def assert_shape(tensor, expected_shape):
    tensor_shape = list(tensor.shape)
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])


class IUSQL(TorchIncrementalRLAlgorithm):
    """Intentional-Unintentional Soft Q-learning (IU-SQL).

    """
    def __init__(self,
                 env,
                 u_qfs,
                 u_policies,
                 i_qf=None,
                 i_policy=None,
                 exploration_pol_id=0,
                 iu_mode='composition',

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
            plotter (`MultiQFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            eval_deterministic: Evaluate with deterministic version of current
                _i_policy.
            **kwargs:
        """
        self._n_unintentional = len(u_qfs)

        if i_policy is None:
            self._i_policy = u_policies[exploration_pol_id]
        else:
            self._i_policy = i_policy

        if eval_deterministic:
            eval_policy = MakeDeterministic(self._i_policy)
        else:
            eval_policy = self._i_policy

        if i_qf is None:
            self._i_qf = u_qfs[exploration_pol_id]
        else:
            self._i_qf = i_qf

        self._iu_mode = iu_mode
        if iu_mode == 'composition':
            self._i_target_qf = None
        else:
            self._i_target_qf = self._i_qf.copy()

        super(IUSQL, self).__init__(
            env=env,
            exploration_policy=self._i_policy,
            eval_policy=eval_policy,
            **kwargs
        )

        # Unintentional Tasks
        self._u_policies = u_policies
        self._u_qfs = u_qfs
        self._u_target_qfs = [qf.copy() for qf in self._u_qfs]

        # Plotter
        self._epoch_plotter = plotter

        # Env data
        self._action_dim = self.env.action_space.low.size
        self._obs_dim = self.env.observation_space.low.size

        # Optimize Q-fcn
        self._u_qf_optimizers = [optimizer_class(qf.parameters(), lr=qf_lr, )
                                 for qf in self._u_qfs]
        self._value_n_particles = value_n_particles

        if iu_mode == 'composition':
            self._i_qf_optimizer = None
        else:
            self._i_qf_optimizer = optimizer_class(self._i_qf.parameters(),
                                                   lr=qf_lr,)

        # Optimize Sampling Policy
        self._u_policy_optimizers = [optimizer_class(policy.parameters(),
                                                     lr=policy_lr, )
                                     for policy in self._u_policies]
        if iu_mode == 'composition':
            self._i_policy_optimizer = \
                optimizer_class(self._i_policy.parameters(),
                                lr=policy_lr, )
        else:
            self._i_policy_optimizer = None

        self._kernel_n_particles = kernel_n_particles
        self._kernel_update_ratio = kernel_update_ratio
        self._kernel_fn = kernel_fn

        # Optimize target Q-fcn
        self.use_hard_updates = use_hard_updates
        self.hard_update_period = hard_update_period
        self.soft_target_tau = soft_target_tau

        # Evaluation Sampler (One for each unintentional
        self.eval_samplers = [
            InPlacePathSampler(env=env, policy=eval_policy,
                               total_samples=self.num_steps_per_eval + self.max_path_length,
                               max_path_length=self.max_path_length, )
            for eval_policy in self._u_policies
        ]

    def pretrain(self):
        # Match target Qfcn with current one
        for unint_idx in range(self._n_unintentional):
            self._update_target_softq_fcn(unint_idx=unint_idx)

        if self._iu_mode == 'composition':
            pass
        else:
            self._update_target_softq_fcn(unint_idx=None)

    def _do_training(self):
        batch = self.get_batch()

        # Update Unintentional Networks
        for unint_idx in range(self._n_unintentional):
            bellman_residual = self._update_softq_fcn(batch, unint_idx)
            surrogate_cost = self._update_sampling_policy(batch, unint_idx)
            self._update_target_softq_fcn(unint_idx=unint_idx)

            if self.eval_statistics is None:
                """
                Eval should set this to None.
                This way, these statistics are only computed for one batch.
                """
                self.eval_statistics = OrderedDict()
            self.eval_statistics['[%d] Bellman Residual (QFcn)' % unint_idx] = \
                np.mean(ptu.get_numpy(bellman_residual))
            self.eval_statistics['[%d] Surrogate Reward (Policy)' % unint_idx] = \
                np.mean(ptu.get_numpy(surrogate_cost))

        # Update Intentional Networks
        if self._iu_mode == 'composition':
            pass
        else:
            bellman_residual = self._update_softq_fcn(batch, unint_idx=None)
            self.eval_statistics['Bellman Residual (QFcn)'] = \
                np.mean(ptu.get_numpy(bellman_residual))

        if self._iu_mode == 'composition':
            surrogate_cost = self._update_sampling_policy(batch, unint_idx=None)
            self.eval_statistics['Surrogate Reward (Intentional Policy)'] = \
                np.mean(ptu.get_numpy(surrogate_cost))
        else:
            pass

        if self._iu_mode == 'composition':
            pass
        else:
            self._update_target_softq_fcn(unint_idx=None)

    # SoftQ Functions
    def _update_softq_fcn(self, batch, unint_idx=None):
        """
        Q-fcn update
        Args:
            batch:

        Returns:

        """

        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        if unint_idx is None:
            rewards = batch['rewards']
        else:
            rewards = batch['reward_vectors'][:, unint_idx].unsqueeze(-1) \
                      * self.reward_scale
        terminals = batch['terminals']
        n_batch = obs.shape[0]

        if unint_idx is None:
            target_q_fcn = self._i_target_qf
            q_fcn = self._i_qf
            q_fcn_optimizer = self._i_qf_optimizer
        else:
            target_q_fcn = self._u_target_qfs[unint_idx]
            q_fcn = self._u_qfs[unint_idx]
            q_fcn_optimizer = self._u_qf_optimizers[unint_idx]

        # The value of the next state is approximated with uniform act. samples.
        uniform_dist = torch.distributions.Uniform(ptu.FloatTensor([-1.0]),
                                                   ptu.FloatTensor([1.0]))
        target_actions = uniform_dist.sample((self._value_n_particles,
                                              self._action_dim)).squeeze()
        q_value_targets = \
            target_q_fcn(
                next_obs.unsqueeze(1).expand(n_batch,
                                             self._value_n_particles,
                                             self._obs_dim),
                target_actions.unsqueeze(0).expand(n_batch,
                                                   self._value_n_particles,
                                                   self._action_dim)
            ).squeeze()
        assert_shape(q_value_targets, [n_batch, self._value_n_particles])

        q_values = q_fcn(obs, actions).squeeze()
        assert_shape(q_values, [n_batch])

        # Equation 10: Vsoft: 'Empirical' mean from q_vals_tgts particles
        next_value = log_sum_exp(q_value_targets.squeeze(), dim=1)
        assert_shape(next_value, [n_batch])

        # Importance _weights add just a constant to the value.
        next_value -= torch.log(ptu.FloatTensor([self._value_n_particles]))
        next_value += self._action_dim * np.log(2)

        # \hat Q in Equation 11
        # ys = (self.reward_scale * rewards.squeeze() +  # Current reward
        ys = (rewards.squeeze() +  # Scale reward is already done by base class
              (1 - terminals.squeeze()) * self.discount * next_value
              ).detach()  # TODO: CHECK IF I AM DETACHING GRADIENT!!!
        assert_shape(ys, [n_batch])

        # Equation 11: Soft-Bellman error
        bellman_residual = 0.5 * torch.mean((ys - q_values) ** 2)

        # Gradient descent on _i_policy parameters
        q_fcn_optimizer.zero_grad()  # Zero all model var grads
        bellman_residual.backward()  # Compute gradient of surrogate_loss
        q_fcn_optimizer.step()  # Update model vars

        return bellman_residual

    # Sampling Policy
    def _update_sampling_policy(self, batch, unint_idx=None):
        """
        Policy update: SVGD
        Returns:

        """
        obs = batch['observations']
        next_obs = batch['next_observations']
        n_batch = obs.shape[0]

        if unint_idx is None:
            policy = self._i_policy
            q_fcn = self._i_qf
            pol_optimizer = self._i_policy_optimizer
        else:
            policy = self._u_policies[unint_idx]
            q_fcn = self._u_qfs[unint_idx]
            pol_optimizer = self._u_policy_optimizers[unint_idx]

        actions = policy(
            obs.unsqueeze(1).expand(n_batch,
                                    self._kernel_n_particles,
                                    self._obs_dim)
        )
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
            (q_fcn(next_obs.unsqueeze(1).expand(n_batch,
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
                                        policy.parameters(),
                                        grad_outputs=action_gradients,
                                        create_graph=False)

        # TODO: Check a better way to do this
        for pp, (w, g) in enumerate(zip(policy.parameters(),
                                        gradients)):
            if pp == 0:
                surrogate_loss = torch.sum(w*g)
            else:
                surrogate_loss += torch.sum(w*g)

        # Gradient descent on _i_policy parameters
        pol_optimizer.zero_grad()  # Zero all model var grads
        (-surrogate_loss).backward()  # Compute gradient of surrogate_loss
        pol_optimizer.step()  # Update model vars

        return -surrogate_loss

    # Target Q-Functions
    def _update_target_softq_fcn(self, unint_idx=None):
        """
        Update the Target SoftQ function
        Args:
            unint_idx: ID of the unintentional task.
                      None updates for the intentional one.

        Returns: None

        """
        if unint_idx is None:
            target_q_fcn = self._i_target_qf
            q_fcn = self._i_qf
        else:
            target_q_fcn = self._u_target_qfs[unint_idx]
            q_fcn = self._u_qfs[unint_idx]

        if self.use_hard_updates:
            # print(self._n_train_steps_total, self.hard_update_period)
            if self._n_train_steps_total % self.hard_update_period == 0:
                ptu.copy_model_params_from_to(q_fcn,
                                              target_q_fcn)
        else:
            ptu.soft_update_from_to(q_fcn, target_q_fcn,
                                    self.soft_target_tau)

    @property
    def torch_models(self):
        if self._i_target_qf is None:
            target_i_q_fcn = []
        else:
            target_i_q_fcn = [self._i_target_qf]

        return [self._i_policy] + self._u_policies + \
               [self._i_qf] + self._u_qfs + \
               target_i_q_fcn + self._u_target_qfs

    def get_epoch_snapshot(self, epoch):
        if self._epoch_plotter is not None:
            self._epoch_plotter.draw()
            self._epoch_plotter.save_figure(epoch)

        snapshot = super(IUSQL, self).get_epoch_snapshot(epoch)
        snapshot.update(
            policy=self._i_policy,
            qf=self._i_qf,
            target_qf=self._i_target_qf,
            u_policies=self._u_policies,
            u_qfs=self._u_qfs,
            target_uqfs=self._u_target_qfs,
        )
        return snapshot

    def evaluate(self, epoch):
        statistics = OrderedDict()
        statistics.update(self.eval_statistics)
        self.eval_statistics = None

        test_paths = [None for _ in range(self._n_unintentional)]
        for demon in range(self._n_unintentional):
            logger.log("[U-%02d] Collecting samples for evaluation" % demon)
            test_paths[demon] = self.eval_samplers[demon].obtain_samples()

            statistics.update(eval_util.get_generic_path_information(
                test_paths[demon], stat_prefix="[U-%02d] Test" % demon,
            ))
            average_returns = eval_util.get_average_returns(test_paths[demon])
            statistics['[U-%02d] AverageReturn' % demon] = average_returns

        logger.log("[I] Collecting samples for evaluation")
        i_test_path = self.eval_sampler.obtain_samples()
        statistics.update(eval_util.get_generic_path_information(
            i_test_path, stat_prefix="[I] Test",
        ))
        average_return = eval_util.get_average_returns(i_test_path)
        statistics['[I] AverageReturn'] = average_return

        statistics.update(eval_util.get_generic_path_information(
            self._exploration_paths, stat_prefix="Exploration",
        ))
        if hasattr(self.env, "log_diagnostics"):
            # TODO: CHECK ENV LOG_DIAGNOSTICS
            print('TODO: WE NEED LOG_DIAGNOSTICS IN ENV')
            # self.env.log_diagnostics(test_paths[demon])

        # Record the data
        for key, value in statistics.items():
            logger.record_tabular(key, value)

        for demon in range(self._n_unintentional):
            if self.render_eval_paths:
                # TODO: CHECK ENV RENDER_PATHS
                print('TODO: RENDER_PATHS')
                pass
                # self.env.render_paths(test_paths[demon])

        if self._epoch_plotter is not None:
            self._epoch_plotter.draw()
