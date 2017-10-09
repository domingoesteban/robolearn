"""
This file defines the DREPS-based trajectory optimization method.
"""
import copy
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from robolearn.algos.gps.gps import GPS
from robolearn.algos.trajopt.trajopt_config import default_dreps_hyperparams
from robolearn.agents.agent_utils import generate_noise

from robolearn.policies.lin_gauss_policy import LinearGaussianPolicy
from robolearn.algos.gps.gps_utils import gauss_fit_joint_prior
from robolearn.utils.print_utils import *

from robolearn.utils.sample.sample import Sample
from robolearn.utils.sample.sample_list import SampleList


class DREPS(GPS):
    """ Sample-based trajectory optimization with DREPS. """
    def __init__(self, agent, env, **kwargs):
        super(DREPS, self).__init__(agent, env, **kwargs)
        gps_algo_hyperparams = default_dreps_hyperparams.copy()
        gps_algo_hyperparams.update(self._hyperparams['gps_algo_hyperparams'])
        self._hyperparams.update(gps_algo_hyperparams)

        self.gps_algo = 'dreps'

        self.good_samples = self._hyperparams['good_samples']
        self.bad_samples = self._hyperparams['bad_samples']
        self.good_traj_prob = None
        self.bad_traj_prob = None

        if self.good_traj_prob is None or self.bad_traj_prob is None:
            print("->Generating good/bad trajectories distributions...")
            self.clustering()

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

    def clustering(self):
        M = len(self.good_samples)
        print("%d conditions" % M)

        for cond in range(M):
            # print('Bad policy...')
            # bad_samples = self.bad_samples[cond]
            # bad_cs, bad_other_costs = self._eval_sample_list_cost(bad_samples, cond)
            # bad_policy = self.fit_lgc(bad_samples, sample_costs=bad_cs, maximize=True, cost_eta=100)
            # # Sampling
            # self.sample_from_policy(cond, bad_policy, noisy=False)

            print('Good policy...')
            good_samples = self.good_samples[cond]
            good_cs, good_other_costs = self._eval_sample_list_cost(good_samples, cond)
            good_policy = self.fit_lgc(good_samples, sample_costs=None, maximize=False, cost_eta=0.01)
            # Sampling
            self.sample_from_policy(cond, good_policy, noisy=False)


        print(np.mean(np.sum(bad_cs, 1), 0))
        print(np.mean(np.sum(good_cs, 1), 0))


        for t in range(self.T):
            pass




        raw_input('yacacuyto')


    def fit_lgc(self, samples, sample_costs=None, maximize=True, cost_eta=1):
        X = samples.get_states()
        obs = samples.get_obs()
        U = samples.get_actions()

        print(X[:, -1, -2])
        X[:, :, :] = X[1, :, :]
        U[:, :, :] = U[1, :, :]
        print(X[:, -1, -2])

        plt.plot(X[2, :, -3], label=['x'])
        plt.plot(X[2, :, -2], label=['y'])
        plt.plot(X[2, :, -1], label=['z'])
        plt.legend()
        plt.show()

        N = X.shape[0]
        if N == 1:
            raise ValueError("Cannot fit dual_policy on 1 sample")

        pol_mu = U
        pol_sig = np.zeros((N, self.T, self.dU, self.dU))

        for t in range(self.T):
            # Using only diagonal covariances
            pol_sig[:, t, :, :] = np.tile(np.diag(np.diag(np.cov(U[:, t, :].T))), (N, 1, 1))

        # Collapse policy covariances. (This is only correct because the policy doesn't depend on state).
        pol_sig = np.mean(pol_sig, axis=0)

        # Allocate.
        pol_K = np.zeros([self.T, self.dU, self.dX])
        pol_k = np.zeros([self.T, self.dU])
        pol_S = np.zeros([self.T, self.dU, self.dU])
        chol_pol_S = np.zeros([self.T, self.dU, self.dU])
        inv_pol_S = np.zeros([self.T, self.dU, self.dU])

        # Update policy prior.
        def eval_prior(Ts, Ps):
            #strength = 1e-4
            strength = 1e-10
            dX, dU = Ts.shape[-1], Ps.shape[-1]
            prior_fd = np.zeros((dU, dX))
            prior_cond = 1e-5 * np.eye(dU)
            sig = np.eye(dX)
            Phi = strength * np.vstack([np.hstack([sig, sig.dot(prior_fd.T)]),
                                        np.hstack([prior_fd.dot(sig), prior_fd.dot(sig).dot(prior_fd.T) + prior_cond])])
            return np.zeros(dX+dU), Phi, 0, strength

        for t in range(self.T):
            # Fit linearization with least squares regression
            if sample_costs is None:
                dwts = (1.0 / N) * np.ones(N)
            else:
                cost_to_go = np.sum(sample_costs[:, t:self.T], axis=1)
                print(cost_to_go)
                if maximize is True:
                    exponent = cost_to_go
                    exp_cost = np.exp((exponent - np.max(exponent)) / cost_eta)
                else:
                    exponent = -cost_to_go
                    exp_cost = np.exp((exponent - np.max(exponent)) / cost_eta)

                dwts = exp_cost / np.sum(exp_cost)
                print(dwts)

            Ts = X[:, t, :]
            Ps = pol_mu[:, t, :]
            Ys = np.concatenate([Ts, Ps], axis=1)
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = eval_prior(Ts, Ps)
            sig_reg = np.zeros((self.dX+self.dU, self.dX+self.dU))
            # Slightly regularize on first timestep.
            if t == 0:
                sig_reg[:self.dX, :self.dX] = 1e-8
            pol_K[t, :, :], pol_k[t, :], pol_S[t, :, :] = gauss_fit_joint_prior(Ys, mu0, Phi, mm, n0, dwts,
                                                                                self.dX, self.dU, sig_reg)
        pol_S += pol_sig  # Add policy covariances mean

        for t in range(self.T):
            chol_pol_S[t, :, :] = sp.linalg.cholesky(pol_S[t, :, :])
            inv_pol_S[t, :, :] = np.linalg.inv(pol_S[t, :, :])

        return LinearGaussianPolicy(pol_K, pol_k, pol_S, chol_pol_S, inv_pol_S)

    def sample_from_policy(self, cond, policy, noisy=True):
        sample = Sample(self.env, self.T)
        history = [None] * self.T
        obs_hist = [None] * self.T

        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        print("Resetting environment...")
        self.env.reset(time=2, cond=cond)
        import rospy

        ros_rate = rospy.Rate(int(1/self.dt))  # hz
        # Collect history
        sampling_bar = ProgressBar(self.T, bar_title='Sampling')
        for t in range(self.T):
            sampling_bar.update(t)
            obs = self.env.get_observation()
            state = self.env.get_state()
            print(state[-3:])
            action = policy.eval(state.copy(), obs.copy(), t, noise[t, :].copy())  # TODO: Avoid TF policy writes in obs
            self.env.send_action(action)
            obs_hist[t] = (obs, action)
            history[t] = (state, action)
            ros_rate.sleep()

        sampling_bar.end()

        # Stop environment
        self.env.stop()
        print("Generating sample data...")

        all_actions = np.array([hist[1] for hist in history])
        all_states = np.array([hist[0] for hist in history])
        all_obs = np.array([hist[0] for hist in obs_hist])
        sample.set_acts(all_actions)   # Set all actions at the same time
        sample.set_obs(all_obs)        # Set all obs at the same time
        sample.set_states(all_states)  # Set all states at the same time
        sample.set_noise(noise)        # Set all noise at the same time

        return sample
