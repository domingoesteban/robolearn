import gtimer as gt
import numpy as np
import scipy as sp
import torch
import math
import copy
import logging

from robolearn.core.iterative_rl_algorithm import IterativeRLAlgorithm
from robolearn.core import logger
import robolearn.torch.pytorch_util as ptu

# from robolearn.utils.plots.core import subplots

from collections import OrderedDict

from robolearn.algorithms.rl_algos import ConstantPolicyPrior

from robolearn.algorithms.rl_algos import generate_noise
from robolearn.algorithms.rl_algos import IterationData
from robolearn.algorithms.rl_algos import TrajectoryInfo
from robolearn.algorithms.rl_algos import PolicyInfo


from robolearn.algorithms.rl_algos import DynamicsLRPrior
from robolearn.algorithms.rl_algos import DynamicsPriorGMM

from robolearn.algorithms.rl_algos import TrajOptLQR


class MDGPS(IterativeRLAlgorithm):
    def __init__(self,
                 env,
                 local_policies,
                 global_policy,
                 cost_fcn,
                 eval_env=None,
                 train_cond_idxs=None,
                 test_cond_idxs=None,
                 num_samples=1,
                 test_samples=1,
                 noisy_samples=True,
                 noise_hyperparams=None,
                 seed=10,
                 base_kl_step=0.1,
                 global_opt_iters=5000,
                 global_opt_batch_size=64,
                 global_opt_lr=1e-5,
                 traj_opt_prev='nn_pol',
                 traj_opt_iters=1,
                 traj_opt_min_eta=1e-8,
                 traj_opt_max_eta=1e16,
                 **kwargs):

        # TO DEFINE
        self._fit_dynamics = True
        self._initial_state_var = 1.0e-2

        self._global_opt_batch_size = global_opt_batch_size
        self._global_opt_iters = global_opt_iters
        self._global_opt_ent_reg = 0.0  # For update pol variance
        self._global_pol_sample_mode = 'add'
        self._global_opt_lr = global_opt_lr
        self._global_samples_counter = 0
        self._first_global_eval = False

        self.base_kl_step = base_kl_step
        self._max_step_mult = 3.0
        self._min_step_mult = 0.5
        self._kl_step_rule = 'laplace'

        self._traj_opt_iters = traj_opt_iters
        self._max_ent_traj = 0.0
        self._traj_opt_prev = traj_opt_prev

        self.T = kwargs['max_path_length']
        self._num_samples = num_samples
        self._test_samples = test_samples

        self._train_cond_idxs = train_cond_idxs
        self._test_cond_idxs = test_cond_idxs

        # Get dimensions from the environment
        self.dU = env.action_dim
        self.dX = env.obs_dim  # TODO: DOING THIS TEMPORALLY
        self.dO = env.obs_dim

        # Number of initial conditions
        self.M = len(local_policies)

        exploration_policy = global_policy

        IterativeRLAlgorithm.__init__(
            self,
            env=env,
            exploration_policy=exploration_policy,
            eval_env=eval_env,
            eval_policy=global_policy,
            eval_sampler=self.sample_global_pol,
            **kwargs
        )

        # Rename for GPS
        self.global_policy = self.eval_policy
        self.local_policies = local_policies

        # Noise to be used with trajectory distributions
        self.noise_data = np.zeros((self.num_epochs, self.M,
                                    self._num_samples,
                                    self.T, self.dU))
        self._noisy_samples = noisy_samples
        if self._noisy_samples:
            for ii in range(self.num_epochs):
                for cond in range(self.M):
                    for n in range(self._num_samples):
                        self.noise_data[ii, cond, n, :, :] = \
                            generate_noise(self.T, self.dU, noise_hyperparams)

        # IterationData objects for each condition.
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]

        # Trajectory Info
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()

            if self._fit_dynamics:
                sigma_regu = 1e-6
                prior = DynamicsPriorGMM(
                    min_samples_per_cluster=40,
                    max_clusters=20,
                    max_samples=20,
                    strength=1.,
                )

                self.cur[m].traj_info.dynamics = \
                    DynamicsLRPrior(prior=prior, sigma_regu=sigma_regu)

                self.cur[m].traj_distr = local_policies[m]

        # Cost Fcn
        self._cost_fcn = cost_fcn

        # Global Policy Optimization
        self.global_pol_optimizer = torch.optim.Adam(
            self.global_policy.parameters(),
            lr=self._global_opt_lr,
            betas=(0.9, 0.999),
            eps=1e-08,  # Term added to the denominator for numerical stability
            # weight_decay=0.005,
            weight_decay=0.5,
            amsgrad=True,
        )

        # Local Trajectory Information
        self._local_pol_optimizer = TrajOptLQR(
            cons_per_step=False,
            use_prev_distr=False,
            update_in_bwd_pass=True,
            min_eta=traj_opt_min_eta,
            max_eta=traj_opt_max_eta,
        )

        level = logging.INFO
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        console = logging.StreamHandler()
        self.logger.addHandler(console)
        for handler in self.logger.handlers:
            handler.setLevel(level)

        self.eval_statistics = None

        self._return_fig = None
        self._return_axs = None
        self._return_lines = [None for _ in range(self.n_test_conds)]

        # MDGPS data #
        # ---------- #
        for m in range(self.M):
            # Same policy prior type for all conditions
            self.cur[m].pol_info = PolicyInfo(
                T=self.T,
                dU=self.dU,
                dX=self.dX,
                init_pol_wt=0.01,
            )
            self.cur[m].pol_info.policy_prior = ConstantPolicyPrior()

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

            # self._current_path_builder = PathBuilder()

            # Sample from environment using current trajectory distributions
            noise = self.noise_data[epoch]
            self.logger.info('')
            self.logger.info('%s: itr:%02d | '
                             'Sampling from local trajectories...'
                             % (type(self).__name__, epoch))
            paths = self.sample_local_pol(noise=noise)
            self._exploration_paths = paths
            # self._handle_path(paths)
            self._n_env_steps_total += int(self.n_train_conds*self._num_samples*self.T)

            # Iterative learning step
            gt.stamp('sample')
            self._try_to_train()
            gt.stamp('train')

            # Evaluate if requirements are met
            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch()

    def _do_training(self):
        epoch = self._n_epochs
        # batch = self.get_batch()
        paths = self.get_exploration_paths()
        self.logger.info('')

        self.logger.info('')
        self.logger.info('%s: itr:%02d | '
                         'Creating Sample List...'
                         % (type(self).__name__, epoch))
        for m, m_train in enumerate(self._train_cond_idxs):
            self.cur[m_train].sample_list = SampleList(paths[m])

        # Update dynamics model using all samples.
        self.logger.info('')
        self.logger.info('%s: itr:%02d | '
                         'Updating dynamics linearization...'
                         % (type(self).__name__, epoch))
        self._update_dynamic_model()

        # Evaluate sample costs
        self.logger.info('')
        self.logger.info('%s: itr:%02d | '
                         'Evaluating samples costs...'
                         % (type(self).__name__, epoch))
        self._eval_iter_samples_costs()

        # Update Networks
        # On the first iteration, need to catch policy up to init_traj_distr.
        if self._n_epochs == 1:
            self.logger.info("\n"*2)
            self.logger.info('%s: itr:%02d | '
                             'S-step for init_traj_distribution (iter=0)...'
                             % (type(self).__name__, epoch))
            self.new_traj_distr = [self.cur[cond].traj_distr
                                   for cond in range(self.M)]
            self._update_global_policy()

            # TODO:
            self.sample_global_pol()

        # Update global policy linearizations.
        self.logger.info('')
        self.logger.info('%s: itr:%02d | '
                         'Updating global policy linearization...'
                         % (type(self).__name__, epoch))
        self._update_local_policies_fit()

        # Update KL step
        if self._n_epochs > 1:
            self.logger.info('')
            self.logger.info('%s: itr:%02d | '
                             'Updating KL step size with GLOBAL policy...'
                             % (type(self).__name__, epoch))
            self._update_kl_step_size()

        # C-step
        self.logger.info('')
        self.logger.info('%s: itr:%02d | '
                         'Updating trajectories...'
                         % (type(self).__name__, epoch))
        for ii in range(self._traj_opt_iters):
            self.logger.info('-%s: itr:%02d | Inner iteration %d/%d'
                             % (type(self).__name__, epoch, ii+1,
                                self._traj_opt_iters))
            self._update_local_policies()

        # S-step
        self.logger.info('')
        self.logger.info('%s:itr:%02d | ->| S-step |<-'
                         % (type(self).__name__, epoch))
        self._update_global_policy()

        # if self.eval_statistics is None:
        #     """
        #     Eval should set this to None.
        #     This way, these statistics are only computed for one batch.
        #     """
        #     self.eval_statistics = OrderedDict()
        #     # self.eval_statistics['Bellman Residual (QFcn)'] = \
        #     #     np.mean(ptu.get_numpy(bellman_residual))
        #     self.eval_statistics['Surrogate Reward (Policy)'] = \
        #         np.mean(ptu.get_numpy(surrogate_cost))

    def _can_evaluate(self):
        return True

    def evaluate(self, epoch):
        statistics = OrderedDict()
        self._update_logging_data()
        statistics.update(self.eval_statistics)
        self.eval_statistics = None
        paths = self.sample_global_pol()

        if paths is None:
            print("NO LOGGING LAST SAMPLING")
            return

        cond_returns_mean = np.zeros(len(paths))
        cond_returns_std = np.zeros(len(paths))

        for cc, cond_path in enumerate(paths):
            sample_list = SampleList(cond_path)

            true_cost, cost_estimate, cost_compo = \
                self._eval_sample_list_cost(sample_list, self._cost_fcn)

            cond_returns_mean[cc] = np.mean(np.sum(true_cost, axis=-1))
            cond_returns_std[cc] = np.std(np.sum(true_cost, axis=-1))

            stat_txt = '[Cond-%02d] Global Mean Return' % cc
            statistics[stat_txt] = cond_returns_mean[cc]

            stat_txt = '[Cond-%02d] Global Std Return' % cc
            statistics[stat_txt] = cond_returns_std[cc]

            stat_txt = '[Cond-%02d] Eta' % cc
            statistics[stat_txt] = self.cur[cc].eta

        # stat_txt = 'Mean Return'
        # statistics[stat_txt] = np.mean(cond_returns_mean)

        # Record the data
        for key, value in statistics.items():
            logger.record_tabular(key, value)

        self._update_plot(statistics)

    def _update_plot(self, statistics):
        # if self._return_fig is None:
        #     # self._return_fig, self._return_axs = subplots(1, self.n_test_conds+1)
        #     self._return_fig, self._return_axs = plt.subplots(1, self.n_test_conds+1)
        #     for aa, ax in enumerate(self._return_axs[:-1]):
        #         self._return_lines = \
        #             ax.plot(self._n_epochs,
        #                     statistics['[Cond-%02d] Mean Return' % aa],
        #                     color='b',
        #                     marker='o',
        #                     markersize=2
        #                     )
        #     # plt.show(block=False)
        # else:
        #     for aa, line in enumerate(self._return_lines[:-1]):
        #         line.set_xdata(
        #             np.append(line.get_xdata(),
        #                       self._n_epochs)
        #         )
        #         line.set_ydata(
        #             np.append(line.get_ydata(),
        #                       statistics['[Cond-%02d] Mean Return' % aa])
        #         )
        #     self._return_fig.canvas.draw()
        #     plt_pause(0.01)

        # self._return_fig, self._return_axs = plt.subplots(1, self.n_test_conds+1)
        # for aa, ax in enumerate(self._return_axs[:-1]):
        #     self._return_lines = \
        #         ax.plot(self._n_epochs,
        #                 statistics['[Cond-%02d] Mean Return' % aa],
        #                 color='b',
        #                 marker='o',
        #                 markersize=2
        #                 )
        # self._return_fig.savefig('tempo/fig%02d.png' % self._n_epochs)
        #
        # del self._return_fig
        # del self._return_axs
        # del self._return_lines
        pass

    def _update_logging_data(self):
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()

    def _end_epoch(self):
        # TODO: change IterationData to reflect new stuff better

        del self.prev
        self.prev = copy.deepcopy(self.cur)

        for m in range(self.M):
            self.prev[m].new_traj_distr = self.new_traj_distr[m]

        # NEW IterationData object, and remove new_traj_distr
        self.cur = [IterationData() for _ in range(self.M)]
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].traj_info.dynamics = \
                copy.deepcopy(self.prev[m].traj_info.dynamics)
            self.cur[m].step_mult = self.prev[m].step_mult
            self.cur[m].eta = self.prev[m].eta
            self.cur[m].traj_distr = self.new_traj_distr[m]
            self.cur[m].traj_info.last_kl_step = \
                self.prev[m].traj_info.last_kl_step
            # MDGPS
            self.cur[m].pol_info = copy.deepcopy(self.prev[m].pol_info)
        self.new_traj_distr = None

        IterativeRLAlgorithm._end_epoch(self)

    def _update_dynamic_model(self):
        """
        Instantiate dynamics objects and update prior.
        Fit dynamics to current samples.
        """
        for m in range(self.M):
            cur_data = self.cur[m].sample_list
            X = cur_data['observations']
            U = cur_data['actions']

            # Update prior and fit dynamics.
            self.cur[m].traj_info.dynamics.update_prior(X, U)
            self.cur[m].traj_info.dynamics.fit(X, U)

            # Fm = self.cur[m].traj_info.dynamics.Fm
            # fv = self.cur[m].traj_info.dynamics.fv
            # T = -2
            # N = 0
            # oo = X[N, T, :]
            # uu = U[N, T, :]
            # oo_uu = np.concatenate((oo, uu), axis=0)
            # oop1 = Fm[T].dot(oo_uu) + fv[T]
            # print('real', X[N, T+1, :])
            # print('pred', oop1)
            # input('fds')

            # Fit x0mu/x0sigma.
            x0 = X[:, 0, :]
            x0mu = np.mean(x0, axis=0)
            self.cur[m].traj_info.x0mu = x0mu
            self.cur[m].traj_info.x0sigma = \
                np.diag(np.maximum(np.var(x0, axis=0),
                                   self._initial_state_var))

            prior = self.cur[m].traj_info.dynamics.get_prior()
            if prior:
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = len(cur_data)
                self.cur[m].traj_info.x0sigma += \
                    Phi + (N*priorm) / (N+priorm) * \
                    np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)

    def _eval_iter_samples_costs(self):
        for cond in range(self.M):
            sample_list = self.cur[cond].sample_list

            true_cost, cost_estimate, cost_compo = \
                self._eval_sample_list_cost(sample_list, self._cost_fcn)

            # Cost sample
            self.cur[cond].cs = true_cost  # True value of cost.

            # Cost composition
            self.cur[cond].cost_compo = cost_compo  # Cost 'composition'.

            # Cost estimate.
            self.cur[cond].traj_info.Cm = cost_estimate[0]  # Quadratic term (matrix).
            self.cur[cond].traj_info.cv = cost_estimate[1]  # Linear term (vector).
            self.cur[cond].traj_info.cc = cost_estimate[2]  # Constant term (scalar).

    def _eval_sample_list_cost(self, sample_list, cost_fcn):
        """
        Evaluate costs for a sample_list using a specific cost function.
        Args:
            cost: self.cost_function[cond]
            cond: Condition to evaluate cost on.
        """
        # Constants.
        T, dX, dU = self.T, self.dX, self.dU
        N = len(sample_list)

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX+dU))
        Cm = np.zeros((N, T, dX+dU, dX+dU))
        cost_composition = [None for _ in range(N)]
        for n in range(N):
            sample = sample_list[n]
            # Get costs.
            l, lx, lu, lxx, luu, lux, cost_composition[n] = cost_fcn.eval(sample)

            print('XX | cost_compo', [np.sum(co) for co in cost_composition[n]])

            # True value of cost
            cs[n, :] = l

            # Constant term
            cc[n, :] = l

            # Assemble matrix and vector.
            cv[n, :, :] = np.c_[lx, lu]
            Cm[n, :, :, :] = np.concatenate(
                (np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]),
                axis=1
            )

            # Adjust for expanding cost around a sample.
            X = sample['observations']
            U = sample['actions']
            yhat = np.c_[X, U]
            rdiff = -yhat
            rdiff_expand = np.expand_dims(rdiff, axis=2)
            cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)
            cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) \
                        + 0.5 * np.sum(rdiff * cv_update, axis=1)
            cv[n, :, :] += cv_update

        # Expected Costs
        cc = np.mean(cc, axis=0)  # Constant term (scalar).
        cv = np.mean(cv, axis=0)  # Linear term (vector).
        Cm = np.mean(Cm, axis=0)  # Quadratic term (matrix).

        return cs, (Cm, cv, cc), cost_composition

    def _update_global_policy(self):
        """
        Computes(updates) a new global policy.
        :return:
        """
        dU, dO, T = self.dU, self.dO, self.T
        # Compute target mean, cov(precision), and weight for each sample;
        # and concatenate them.
        obs_data, tgt_mu = ptu.zeros((0, T, dO)), ptu.zeros((0, T, dU))
        tgt_prc, tgt_wt = ptu.zeros((0, T, dU, dU)), ptu.zeros((0, T))
        for m in range(self.M):
            samples = self.cur[m].sample_list
            X = samples['observations']
            N = len(samples)
            traj = self.new_traj_distr[m]
            pol_info = self.cur[m].pol_info
            mu = ptu.zeros((N, T, dU))
            prc = ptu.zeros((N, T, dU, dU))
            wt = ptu.zeros((N, T))
            obs = ptu.FloatTensor(samples['observations'])
            # Get time-indexed actions.
            for t in range(T):
                # Compute actions along this trajectory.
                prc[:, t, :, :] = ptu.FloatTensor(
                    np.tile(traj.inv_pol_covar[t, :, :], [N, 1, 1])
                )
                for i in range(N):
                    mu[i, t, :] = ptu.FloatTensor(
                        traj.K[t, :, :].dot(X[i, t, :]) + traj.k[t, :]
                    )
                wt[:, t] = pol_info.pol_wt[t]

            tgt_mu = torch.cat((tgt_mu, mu))
            tgt_prc = torch.cat((tgt_prc, prc))
            tgt_wt = torch.cat((tgt_wt, wt))
            obs_data = torch.cat((obs_data, obs))

        self.global_policy_optimization(obs_data, tgt_mu, tgt_prc, tgt_wt)

    def global_policy_optimization(self, obs, tgt_mu, tgt_prc, tgt_wt):
        """
        Update policy.
        :param obs: Numpy array of observations, N x T x dO.
        :param tgt_mu: Numpy array of mean controller outputs, N x T x dU.
        :param tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
        :param tgt_wt: Numpy array of weights, N x T.
        """
        N, T = obs.shape[:2]
        dU = self.dU
        dO = self.dO

        # Save original tgt_prc.
        tgt_prc_orig = torch.reshape(tgt_prc, [N*T, dU, dU])

        # Renormalize weights.
        tgt_wt *= (float(N * T) / torch.sum(tgt_wt))
        # Allow ights to be at most twice the robust median.
        mn = torch.median(tgt_wt[tgt_wt > 1e-2])
        tgt_wt = torch.clamp(tgt_wt, max=2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        obs = torch.reshape(obs, (N*T, dO))
        tgt_mu = torch.reshape(tgt_mu, (N*T, dU))
        tgt_prc = torch.reshape(tgt_prc, (N*T, dU, dU))
        tgt_wt = torch.reshape(tgt_wt, (N*T, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        # TODO: DO THIS MORE THAN ONCE!!
        if not hasattr(self.global_policy, 'scale') or not hasattr(self.global_policy, 'bias'):
            # 1e-3 to avoid infs if some state dimensions don't change in the
            # first batch of samples
            self.global_policy.scale = ptu.zeros(self.env.obs_dim)
            self.global_policy.bias = ptu.zeros(self.env.obs_dim)

        m = self._global_samples_counter
        n = m + N*T

        scale_obs = torch.diag(1.0 / torch.clamp(torch.std(obs, dim=0),
                                                 min=1e-3))
        var_obs = scale_obs**2
        var_prev = self.global_policy.scale**2

        bias_obs = -torch.mean(obs.matmul(scale_obs), dim=0)
        bias_prev = self.global_policy.bias
        bias_new = float(n/(m+n))*bias_obs + float(m/(m+n))*bias_prev

        var_new = float(n/(m+n))*var_obs + float(m/(m+n))*var_prev - \
                  float((m*n)/(m+n)**2)*(bias_prev - bias_new)**2
        self.global_policy.scale = torch.sqrt(var_new)
        self.global_policy.bias = bias_new

        # self.global_policy.scale = ptu.eye(self.env.obs_dim)
        # self.global_policy.bias = ptu.zeros(self.env.obs_dim)

        # Normalize Inputs
        obs = obs.matmul(self.global_policy.scale) + self.global_policy.bias

        # # Global Policy Optimization
        # self.global_pol_optimizer = torch.optim.Adam(
        #     self.global_policy.parameters(),
        #     lr=self._global_opt_lr,
        #     betas=(0.9, 0.999),
        #     eps=1e-08,  # Term added to the denominator for numerical stability
        #     # weight_decay=0.005,
        #     weight_decay=0.5,
        #     amsgrad=True,
        # )

        # Assuming that N*T >= self.batch_size.
        batches_per_epoch = math.floor(N*T / self._global_opt_batch_size)
        idx = list(range(N*T))
        average_loss = 0
        np.random.shuffle(idx)

        if torch.any(torch.isnan(obs)):
            raise ValueError('GIVING NaN OBSERVATIONS to PYTORCH')
        if torch.any(torch.isnan(tgt_mu)):
            raise ValueError('GIVING NaN ACTIONS to PYTORCH')
        if torch.any(torch.isnan(tgt_prc)):
            raise ValueError('GIVING NaN PRECISION to PYTORCH')

        for oo in range(1):
            print('$$$$\n'*2)
            print('GLOBAL_OPT %02d' % oo)
            print('$$$$\n'*2)
            # # Global Policy Optimization
            # self.global_pol_optimizer = torch.optim.Adam(
            #     self.global_policy.parameters(),
            #     lr=self._global_opt_lr,
            #     betas=(0.9, 0.999),
            #     eps=1e-08,  # Term added to the denominator for numerical stability
            #     # weight_decay=0.005,
            #     weight_decay=0.5,
            #     amsgrad=True,
            # )

            for ii in range(self._global_opt_iters):
                # # Load in data for this batch.
                # start_idx = int(ii * self._global_opt_batch_size %
                #                 (batches_per_epoch * self._global_opt_batch_size))
                # idx_i = idx[start_idx:start_idx+self._global_opt_batch_size]

                # Load in data for this batch.
                idx_i = np.random.choice(N*T, self._global_opt_batch_size)

                self.global_pol_optimizer.zero_grad()

                pol_output = self.global_policy(obs[idx_i], deterministic=True)[0]

                train_loss = euclidean_loss(mlp_out=pol_output,
                                            action=tgt_mu[idx_i],
                                            precision=tgt_prc[idx_i],
                                            batch_size=self._global_opt_batch_size)

                train_loss.backward()
                self.global_pol_optimizer.step()

                average_loss += train_loss.item()

                # del pol_output
                # del train_loss
                loss_tolerance = 5e-10

                if (ii+1) % 50 == 0:
                    print('PolOpt iteration %d, average loss %f'
                          % (ii+1, average_loss/50))
                    average_loss = 0

                if train_loss <= loss_tolerance:
                    print("It converged! loss:", train_loss)
                    break

            if train_loss <= loss_tolerance:
                break

        # Optimize variance.
        A = torch.sum(tgt_prc_orig, dim=0) \
            + 2 * N * T * self._global_opt_ent_reg * ptu.ones((dU, dU))
        A = A / torch.sum(tgt_wt)

        # TODO - Use dense covariance?
        self.global_policy.std = torch.diag(torch.sqrt(A))

    def _global_pol_prob(self, obs):
        dU = self.dU
        N, T = obs.shape[:2]

        # Normalize obs.
        if hasattr(self.global_policy, 'scale'):
            # TODO: Should prob be called before update?
            obs_scale = ptu.get_numpy(self.global_policy.scale)
            obs_bias = ptu.get_numpy(self.global_policy.bias)
            for n in range(N):
                obs[n, :] = obs[n, :].dot(obs_scale) + obs_bias
        else:
            raise AssertionError('WE ARE NOT NORMALIZING THE OBS!!!')

        output = np.zeros((N, T, dU))

        # for i in range(N):
        #     for t in range(T):
        #         # Feed in data.
        #         feed_dict = {self.obs_tensor: np.expand_dims(obs[i, t], axis=0)}
        #         with tf.device(self.device_string):
        #             output[i, t, :] = self.sess.run(self.act_op,
        #                                             feed_dict=feed_dict)
        output = ptu.get_numpy(self.global_policy(ptu.from_numpy(obs),
                                                  deterministic=True)[0]
                               )

        pol_var = ptu.get_numpy(self.global_policy.std) ** 2

        # Same variance for all time steps
        pol_sigma = np.tile(np.diag(pol_var), [N, T, 1, 1])
        pol_prec = np.tile(np.diag(1.0 / pol_var), [N, T, 1, 1])
        pol_det_sigma = np.tile(np.prod(pol_var), [N, T])

        return output, pol_sigma, pol_prec, pol_det_sigma

    def _update_kl_step_size(self):
        estimate_cost_fcn = self._local_pol_optimizer.estimate_cost

        # Compute previous cost and previous expected cost.
        prev_M = len(self.prev)  # May be different in future.
        prev_laplace = np.empty(prev_M)
        prev_mc = np.empty(prev_M)
        prev_predicted = np.empty(prev_M)
        for m in range(prev_M):
            prev_nn = self.prev[m].pol_info.traj_distr()
            prev_lg = self.prev[m].new_traj_distr

            # Compute values under Laplace approximation. This is the policy
            # that the previous samples were actually drawn from under the
            # dynamics that were estimated from the previous samples.
            prev_laplace[m] = estimate_cost_fcn(prev_nn,
                                                self.prev[m].traj_info).sum()
            # This is the actual cost that we experienced.
            prev_mc[m] = self.prev[m].cs.mean(axis=0).sum()
            # This is the policy that we just used under the dynamics that
            # were estimated from the prev samples (so this is the cost
            # we thought we would have).
            prev_predicted[m] = estimate_cost_fcn(prev_lg,
                                                  self.prev[m].traj_info).sum()

        # Compute current cost.
        cur_laplace = np.empty(self.M)
        cur_mc = np.empty(self.M)
        for m in range(self.M):
            cur_nn = self.cur[m].pol_info.traj_distr()
            # This is the actual cost we have under the current trajectory
            # based on the latest samples.
            cur_laplace[m] = estimate_cost_fcn(cur_nn,
                                               self.cur[m].traj_info).sum()
            cur_mc[m] = self.cur[m].cs.mean(axis=0).sum()

        # Compute predicted and actual improvement.
        prev_laplace = prev_laplace.mean()
        prev_mc = prev_mc.mean()
        prev_predicted = prev_predicted.mean()
        cur_laplace = cur_laplace.mean()
        cur_mc = cur_mc.mean()

        if self._kl_step_rule == 'laplace':
            predicted_impr = prev_laplace - prev_predicted
            actual_impr = prev_laplace - cur_laplace
        elif self._kl_step_rule == 'mc':
            predicted_impr = prev_mc - prev_predicted
            actual_impr = prev_mc - cur_mc
        else:
            raise AttributeError('Wrong kl_step_rule')

        for m in range(self.M):
            self._set_new_mult(predicted_impr, actual_impr, m)

    def _set_new_mult(self, predicted_impr, actual_impr, m):
        """
        Adjust step size multiplier according to the predicted versus
        actual improvement.
        """
        # Model improvement as I = predicted_dI * KL + penalty * KL^2,
        # where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2).
        # Optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL =>
        # KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL.
        # Therefore, the new multiplier is given by pred/2*(pred-act).
        new_mult = predicted_impr / (2.0 * max(1e-4,
                                               predicted_impr - actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(min(new_mult * self.cur[m].step_mult,
                           self._max_step_mult),
                       self._min_step_mult
        )
        self.cur[m].step_mult = new_step

        if new_mult > 1:
            print('%s: Increasing step size multiplier to %f'
                  % (type(self).__name__, new_step))
        else:
            print('%s: Decreasing step size multiplier to %f'
                  % (type(self).__name__, new_step))

    def _update_local_policies(self):

        if self.new_traj_distr is None:
            self.new_traj_distr = [self.cur[cond].traj_distr
                                   for cond in range(self.M)]

        for cond in range(self.M):
            traj_opt_outputs = \
                self._local_pol_optimizer.update(cond, self,
                                                 prev_type=self._traj_opt_prev)
            self.new_traj_distr[cond] = traj_opt_outputs[0]
            self.local_policies[cond] = traj_opt_outputs[0]
            self.cur[cond].eta = traj_opt_outputs[1]

    def _update_local_policies_fit(self):
        """
        Re-estimate the local policy values in the neighborhood of the trajectory.
        :return: None
        """
        for cond in range(self.M):
            dX, dU, T = self.dX, self.dU, self.T
            # Choose samples to use.
            samples = self.cur[cond].sample_list
            N = len(samples)
            pol_info = self.cur[cond].pol_info
            X = samples['observations'].copy()
            obs = samples['observations'].copy()

            pol_mu, pol_sig = self._global_pol_prob(obs)[:2]
            pol_info.pol_mu, pol_info.pol_sig = pol_mu, pol_sig

            # Update policy prior.
            policy_prior = pol_info.policy_prior
            # TODO: THE FOLLOWING IS USELESS FOR CONSTANT PRIOR
            # samples = SampleList(self.cur[cond].sample_list)
            # mode = self._global_pol_sample_mode
            # policy_prior.update(samples, self._global_policy, mode)

            # Fit linearization and store in pol_info.
            # max_var = self.cur[cond].traj_distr.max_var
            max_var = None
            pol_info.pol_K, pol_info.pol_k, pol_info.pol_S = \
                policy_prior.fit(X, pol_mu, pol_sig, max_var=max_var)

            for t in range(T):
                pol_info.chol_pol_S[t, :, :] = \
                    sp.linalg.cholesky(pol_info.pol_S[t, :, :])

    def compute_traj_cost(self, cond, eta, augment=True):
        """
        Compute cost estimates used in the LQR backward pass.

        :param cond: Number of condition
        :param eta: Dual variable corresponding to KL divergence with
                    previous policy.
        :param augment: True if we want a KL constraint for all time-steps.
                        False otherwise. True for MDGPS
        :return: Cm and cv
        """

        traj_info = self.cur[cond].traj_info
        traj_distr = self.cur[cond].traj_distr  # We do not use it

        if not augment:  # Whether to augment cost with term to penalize KL
            return traj_info.Cm, traj_info.cv

        T = self.T
        dX = self.dX
        dU = self.dU

        Cm, cv = np.copy(traj_info.Cm), np.copy(traj_info.cv)

        # Pol_info
        pol_info = self.cur[cond].pol_info

        # Weight of maximum entropy term in trajectory optimization
        multiplier = self._max_ent_traj

        # Surrogate cost
        PKLm = np.zeros((T, dX+dU, dX+dU))
        PKLv = np.zeros((T, dX+dU))

        # TODO: 'WARN: adding a beta to divisor in compute_traj_cost')
        eps = 1e-8
        divisor = (eta + multiplier + eps)
        fCm = Cm / divisor
        fcv = cv / divisor

        # Add in the KL divergence with previous policy.
        for t in range(self.T):
            if self._traj_opt_prev == 'nn_pol':
                # Policy KL-divergence terms.
                inv_pol_S = np.linalg.solve(
                    pol_info.chol_pol_S[t, :, :],
                    np.linalg.solve(pol_info.chol_pol_S[t, :, :].T, np.eye(dU))
                )
                KB = pol_info.pol_K[t, :, :]
                kB = pol_info.pol_k[t, :]
            else:
                # Policy KL-divergence terms.
                inv_pol_S = self.cur[cond].traj_distr.inv_pol_covar[t, :, :]
                KB = self.cur[cond].traj_distr.K[t, :, :]
                kB = self.cur[cond].traj_distr.k[t, :]

            PKLm[t, :, :] = np.vstack([
                np.hstack([KB.T.dot(inv_pol_S).dot(KB), -KB.T.dot(inv_pol_S)]),
                np.hstack([-inv_pol_S.dot(KB), inv_pol_S])
            ])
            PKLv[t, :] = np.concatenate([
                KB.T.dot(inv_pol_S).dot(kB), -inv_pol_S.dot(kB)
            ])

            fCm[t, :, :] += PKLm[t, :, :] * eta / divisor
            fcv[t, :] += PKLv[t, :] * eta / divisor

        return fCm, fcv

    def sample_local_pol(self, noise):
        conditions = self._train_cond_idxs
        all_paths = list()
        for cc, cond in enumerate(conditions):
            paths = list()
            # policy = self.local_policies[cc]
            policy = self.cur[cc].traj_distr

            for ss in range(self._num_samples):
                observations = []
                actions = []
                rewards = []
                terminals = []
                agent_infos = []
                env_infos = []

                o = self.env.reset(condition=cond)
                next_o = None
                for t in range(self.T):
                    a, agent_info = \
                        policy.get_action(o, t, noise[cc, ss, t])

                    # Checking NAN
                    nan_number = np.isnan(a)
                    if np.any(nan_number):
                        print("\e[31mERROR ACTION: NAN!!!!!")
                    a[nan_number] = 0

                    next_o, r, d, env_info = self.env.step(a)

                    observations.append(o)
                    rewards.append(r)
                    terminals.append(d)
                    actions.append(a)
                    agent_infos.append(agent_info)
                    env_infos.append(env_info)
                    o = next_o

                actions = np.array(actions)
                if len(actions.shape) == 1:
                    actions = np.expand_dims(actions, 1)

                observations = np.array(observations)
                if len(observations.shape) == 1:
                    observations = np.expand_dims(observations, 1)
                    next_o = np.array([next_o])

                next_observations = np.vstack(
                    (
                        observations[1:, :],
                        np.expand_dims(next_o, 0)
                    )
                )

                path = dict(
                    observations=observations,
                    actions=actions,
                    rewards=np.array(rewards).reshape(-1, 1),
                    next_observations=next_observations,
                    terminals=np.array(terminals).reshape(-1, 1),
                    agent_infos=agent_infos,
                    env_infos=env_infos,
                )
                paths.append(path)
            all_paths.append(paths)

        return all_paths

    def sample_global_pol(self):

        conditions = self._test_cond_idxs
        all_paths = list()
        for cc, cond in enumerate(conditions):
            paths = list()
            policy = self.global_policy
            obs_scale = ptu.get_numpy(policy.scale)
            obs_bias = ptu.get_numpy(policy.bias)

            for ss in range(self._test_samples):
                observations = []
                actions = []
                rewards = []
                terminals = []
                agent_infos = []
                env_infos = []

                o = self.env.reset(condition=cond)
                next_o = None

                for t in range(self.T):
                    pol_input = o.dot(obs_scale) + obs_bias
                    # print(o)
                    # print(pol_input)
                    # print(obs_scale)
                    # print(obs_bias)
                    # print(pol_input)

                    a, agent_info = \
                        policy.get_action(pol_input, deterministic=True)

                    # local_pol = self.local_policies[cc]
                    # local_act = local_pol.get_action(o, t, np.zeros(7))[0]
                    # print(t, 'local', local_act)
                    # print(t, 'NN', a)
                    # if self.cur[cc].pol_info.pol_mu is not None:
                    #     pol_lin = self.cur[cc].pol_info.traj_distr()
                    #     pol_lin_act = pol_lin.get_action(o, t, np.zeros(7))[0]
                    #     print(t, 'lin', pol_lin_act)
                    #
                    #     new_local_pol = self.new_traj_distr[cc]
                    #     new_local_act = new_local_pol.get_action(o, t, np.zeros(7))[0]
                    #     print(t, 'new_local', new_local_act)
                    #
                    #     if self._traj_opt_prev == 'traj':
                    #         a = new_local_act
                    # print('--')

                    # Checking NAN
                    nan_number = np.isnan(a)
                    if np.any(nan_number):
                        print("\e[31mERROR ACTION: NAN!!!!!")
                    a[nan_number] = 0

                    next_o, r, d, env_info = self.env.step(a)

                    observations.append(o)
                    rewards.append(r)
                    terminals.append(d)
                    actions.append(a)
                    agent_infos.append(agent_info)
                    env_infos.append(env_info)
                    o = next_o

                actions = np.array(actions)
                if len(actions.shape) == 1:
                    actions = np.expand_dims(actions, 1)

                observations = np.array(observations)
                if len(observations.shape) == 1:
                    observations = np.expand_dims(observations, 1)
                    next_o = np.array([next_o])

                next_observations = np.vstack(
                    (
                        observations[1:, :],
                        np.expand_dims(next_o, 0)
                    )
                )

                path = dict(
                    observations=observations,
                    actions=actions,
                    rewards=np.array(rewards).reshape(-1, 1),
                    next_observations=next_observations,
                    terminals=np.array(terminals).reshape(-1, 1),
                    agent_infos=agent_infos,
                    env_infos=env_infos,
                )
                paths.append(path)
            all_paths.append(paths)

        return all_paths

    def get_epoch_snapshot(self, epoch):
        """
        Stuff to save in file.
        Args:
            epoch:

        Returns:

        """
        snapshot = super(MDGPS, self).get_epoch_snapshot(epoch)
        snapshot.update(
            global_policy=self.global_policy,
            local_policies=self.local_policies,
        )

        return snapshot

    @property
    def n_train_conds(self):
        return len(self._train_cond_idxs)

    @property
    def n_test_conds(self):
        return len(self._test_cond_idxs)

    @property
    def networks(self):
        networks_list = [
            self.global_policy
        ]

        return networks_list


class SampleList(object):
    def __init__(self, sample_list):
        self._sample_list = [dict(
            observations=sample['observations'],
            actions=sample['actions'],
        ) for sample in sample_list]

    def __getitem__(self, arg):
        if arg == 'observations':
            return np.asarray([data['observations']
                               for data in self._sample_list])
        elif arg == 'actions':
            return np.asarray([data['actions']
                               for data in self._sample_list])
        elif isinstance(arg, int):
            return self._sample_list[arg]
        else:
            raise AttributeError('Wrong argument')

    def __len__(self):
        return len(self._sample_list)


def euclidean_loss(mlp_out, action, precision, batch_size):
    scale_factor = 2.*batch_size

    u = action-mlp_out
    uP = torch.matmul(u.unsqueeze(1), precision).squeeze(1)
    # This last dot product is then summed, so we just the sum all at once.
    uPu = torch.sum(uP*u)
    return uPu/scale_factor

    # uPu = torch.sum(u**2)
    # return uPu/scale_factor

    # uPu = 0.5*torch.sum(mlp_out**2)
    # return uPu
