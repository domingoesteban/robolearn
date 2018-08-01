import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from robolearn_gym_envs.pybullet import CentauroObstacleEnv

from robolearn.v010.costs.cost_action import CostAction
from robolearn.v010.costs.cost_state import CostState
from robolearn.v010.costs.cost_sum import CostSum

from robolearn.v010.utils.sample.sample import Sample
from robolearn.v010.utils.sample.sample_list import SampleList

from robolearn.v010.policies.lin_gauss_policy import LinearGaussianPolicy
from robolearn.v010.policies.lin_gauss_init import init_pd
from robolearn.v010.policies.lin_gauss_init import init_lqr

from robolearn.v010.costs.cost_utils import RAMP_FINAL_ONLY, RAMP_CONSTANT

from robolearn.v010.utils.dynamics.dynamics_lr_prior import DynamicsLRPrior
from robolearn.v010.utils.dynamics.dynamics_prior_gmm import DynamicsPriorGMM

from robolearn.v010.agents.agent_utils import generate_noise

np.set_printoptions(suppress=True)

seed = 10
np.random.seed(seed)

Niter = 30
Tend = 1.0
# Tend = 0.1
Ts = 0.01
T = int(Tend/Ts)
Nrollouts = 4

# noisy = False
noisy = True
eta = 0.5

max_initial_state_var = 1e-2

render = False
# render = True
goal_tolerance = 0.02
SIM_TIMESTEP = 0.01
frame_skip = int(Ts/SIM_TIMESTEP)

# Environment
env = CentauroObstacleEnv(
    is_render=render,
    obs_with_img=False,
    active_joints='RA',
    control_type='tasktorque',
    # control_type='velocity',
    sim_timestep=SIM_TIMESTEP,
    frame_skip=frame_skip,
    obs_distances=True,
    goal_tolerance=goal_tolerance,
    max_time=None,
)

# Initial condition
env_condition = np.array([0.02, 0.25, 00.0, 0.08, 0.6, 0.0])
env_condition[2] = np.deg2rad(env_condition[2])
env_condition[5] = np.deg2rad(env_condition[5])
env.add_tgt_obst_init_cond(tgt_state=env_condition[:3],
                           obst_state=env_condition[3:])
env.update_init_conds()
dX = env.obs_dim
dU = env.action_dim

# Noise
noise_hyperparams = dict(
    smooth_noise=True,
    smooth_noise_var=2.0e+0,
    smooth_noise_renormalize=True,
    # noise_var_scale=1.e-4*np.array([1., 1., 1., 1., 1., 1., 1.]),
    noise_var_scale=1.,
)

# Policy
TORQUE_GAINS = np.array([1.0, 1.0, 1.0, 0.5, 0.1, 0.2, 0.001])
init_policy_hyperparams = {
    'init_gains':  1.0 / TORQUE_GAINS,
    'init_acc': np.zeros(7),
    'init_var': 1.0,
    'stiffness': 1.0,
    'stiffness_vel': 0.5,
    'final_weight': 100.0,
    'dt': Ts,
    'T': T,
    'x0': np.array([0., 0.3, 0.8, 0.8, 0., 0.8, 0.,
                    0., 0.,  0.,  0.,  0., 0.,  0.,
                    0.2095,  0.3503, -0.0584, -0.,  0., -0.]),
    'dX': dX,
    'dU': dU,
}
policy = init_lqr(init_policy_hyperparams)

# Dynamics
dynamics_hyperparams = dict(
    regularization=1e-6,
    prior={
        'type': DynamicsPriorGMM,
        'max_clusters': 20,  # Maximum number of clusters to fit.
        'min_samples_per_cluster': 40,  # Minimum samples per cluster.
        'max_samples': 20,  # Max. number of trajectories to use for fitting the GMM at any given time.
        # 'strength': 1.0,  # Adjusts the strength of the prior.
        },
)
dynamics = DynamicsLRPrior(dynamics_hyperparams)


# Cost Fcn
action_cost = {
    'type': CostAction,
    'wu': 1e-3 / TORQUE_GAINS,
    'target': None,   # Target action value
}

l2_l1_weights = np.array([1.0, 0.1])
# target_state = np.array([0., 0.3, 0.8, 0.8, 0., 0.8, 0.])
# target_state += np.array([0.2, -0.2, 0.3, -0.3, 0.1, 0., 0.])
# # target_state = np.array([0., 0., 0., -0.3, 0., 0., 0.])
# state_cost_distance = {
#     'type': CostState,
#     'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
#     'l2': l2_l1_weights[0],  # Weight for l2 norm
#     'l1': l2_l1_weights[1],  # Weight for l1 norm
#     'alpha': 1.e-2,  # Constant added in square root in l1 norm
#     'wp_final_multiplier': 1.0e0,  # Weight multiplier on final time step.
#     'data_types': {
#         'position': {
#             'wp': np.array([1., 1., 1., 1., 1., 1., 1.]),  # State weights - must be set.
#             'target_state': target_state,  # Target state - must be set.
#             'average': None,
#             'data_idx': env.get_state_info(name='position')['idx']
#         },
#     },
# }
target_state = np.array([0., 0., 0., 0., 0., 0.])
# target_state = np.array([0., 0., 0., -0.3, 0., 0., 0.])
state_cost_distance = {
    'type': CostState,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'l2': l2_l1_weights[0],  # Weight for l2 norm
    'l1': l2_l1_weights[1],  # Weight for l1 norm
    'alpha': 1.e-2,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 1.0e0,  # Weight multiplier on final time step.
    'data_types': {
        'target': {
            'wp': np.array([1., 1., 1., 0., 0., 0.]),  # State weights - must be set.
            'target_state': target_state,  # Target state - must be set.
            'average': None,
            'data_idx': env.get_state_info(name='target')['idx']
        },
    },
}
final_state_cost_distance = {
    'type': CostState,
    'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'l2': l2_l1_weights[0],  # Weight for l2 norm
    'l1': l2_l1_weights[1],  # Weight for l1 norm
    'alpha': 1.e-2,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 1.0e0,  # Weight multiplier on final time step.
    'data_types': {
        'target': {
            'wp': np.array([1., 1., 1., 0.1, 0.1, 0.1]),  # State weights - must be set.
            'target_state': target_state,  # Target state - must be set.
            'average': None,
            'data_idx': env.get_state_info(name='target')['idx']
        },
    },
}

all_costs = [
    action_cost,
    state_cost_distance,
    final_state_cost_distance,
]
all_weights = [
1.e+3,
# 1.e+0,
3.e+1,
3.e+3,
]

cost_sum_hyperparams = dict(
    costs=all_costs,
    weights=all_weights,

)
cost_fcn = CostSum(cost_sum_hyperparams)

iter_costs = np.zeros((Niter+1))

# ############### #
# Evaluate Policy #
# ############### #
# Sample and show cost
noise = np.zeros((T, dU))
sample = Sample(env, T)
all_actions = np.zeros((T, dU))
all_states = np.zeros((T, dX))
all_obs = np.zeros((T, dX))
obs0 = env.reset()
for t in range(T):
    state = env.get_state()# - obs0
    obs = env.get_observation()# - obs0
    action = policy.eval(state.copy(), obs.copy(),
                         t, noise[t].copy())
    env.step(action)
    all_states[t, :] = state
    all_obs[t, :] = obs
    all_actions[t, :] = action
sample.set_acts(all_actions)
sample.set_obs(all_obs)
sample.set_states(all_states)
sample.set_noise(noise)
cost_output = cost_fcn.eval(sample)
iter_costs[0] = np.sum(cost_output[0])

for ii in range(Niter):

    # All samples in iteration
    interaction_samples = list()

    # Sample from environment
    for rr in range(Nrollouts):
        print('Iter %02d' % ii, ' | ', 'Rollout:%02d' % rr)
        sample = Sample(env, T)
        all_actions = np.zeros((T, dU))
        all_states = np.zeros((T, dX))
        all_obs = np.zeros((T, dX))

        if noisy:
            noise = generate_noise(T, dU, noise_hyperparams)
        else:
            noise = np.zeros((T, dU))

        # Reset
        obs0 = env.reset()
        for t in range(T):
            state = env.get_state()# - obs0
            obs = env.get_observation()# - obs0
            action = policy.eval(state, obs, t, noise[t])
            env.step(action)
            # if t == 0:
            #     print('****')
            #     print(noise[t])
            #     print('****')

            all_states[t, :] = state
            all_obs[t, :] = obs
            all_actions[t, :] = action

        sample.set_acts(all_actions)
        sample.set_obs(all_obs)
        sample.set_states(all_states)
        sample.set_noise(noise)
        interaction_samples.append(sample)

    # Samp
    sample_list = SampleList(interaction_samples)

    # Fit Dynamics
    print('****'*2)
    print('FITTING DYNAMICS...')
    print('****'*2)
    cur_data = sample_list
    X = cur_data.get_states()
    U = cur_data.get_actions()

    # Update prior and fit dynamics.
    dynamics.update_prior(cur_data)
    dynamics.fit(X, U)

    # Fit x0mu/x0sigma.
    x0 = X[:, 0, :]
    x0mu = np.mean(x0, axis=0)
    x0mu = x0mu
    x0sigma = \
        np.diag(np.maximum(np.var(x0, axis=0), max_initial_state_var))

    prior = dynamics.get_prior()
    if prior:
        mu0, Phi, priorm, n0 = prior.initial_state()
        N = len(cur_data)
        x0sigma += \
            Phi + (N*priorm) / (N+priorm) * \
            np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)

    # Eval Samples Cost
    print('****'*2)
    print('EVALUATING COST...')
    print('****'*2)
    cs = np.zeros((Nrollouts, T))
    cc = np.zeros((Nrollouts, T))
    cv = np.zeros((Nrollouts, T, dX+dU))
    Cm = np.zeros((Nrollouts, T, dX+dU, dX+dU))
    cost_composition = [None for _ in range(Nrollouts)]

    for n in range(Nrollouts):
        sample = sample_list[n]
        l, lx, lu, lxx, luu, lux, cost_composition[n] = cost_fcn.eval(sample)

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
        X = sample.get_states()
        U = sample.get_acts()
        yhat = np.c_[X, U]
        rdiff = -yhat
        rdiff_expand = np.expand_dims(rdiff, axis=2)
        cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)
        cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) \
                    + 0.5 * np.sum(rdiff * cv_update, axis=1)
        cv[n, :, :] += cv_update

    # Cost estimate.
    cc = np.mean(cc, axis=0)  # Constant term (scalar).
    cv = np.mean(cv, axis=0)  # Linear term (vector).
    Cm = np.mean(Cm, axis=0)  # Quadratic term (matrix).

    # Consider KL divergence
    multiplier = 0
    PKLm = np.zeros((T, dX+dU, dX+dU))
    PKLv = np.zeros((T, dX+dU))
    fCm = np.zeros(Cm.shape)
    fcv = np.zeros(cv.shape)
    for t in range(T):
        # Policy KL-divergence terms.
        inv_pol_S = np.linalg.solve(
            policy.chol_pol_covar[t, :, :],
            np.linalg.solve(policy.chol_pol_covar[t, :, :].T, np.eye(dU))
        )
        KB = policy.K[t, :, :]
        kB = policy.k[t, :]
        PKLm[t, :, :] = np.vstack([
            np.hstack([KB.T.dot(inv_pol_S).dot(KB), -KB.T.dot(inv_pol_S)]),
            np.hstack([-inv_pol_S.dot(KB), inv_pol_S])
        ])
        PKLv[t, :] = np.concatenate([
            KB.T.dot(inv_pol_S).dot(kB), -inv_pol_S.dot(kB)
        ])
        fCm[t, :, :] = (Cm[t, :, :] + PKLm[t, :, :] * eta) / (eta + multiplier)
        fcv[t, :] = (cv[t, :] + PKLv[t, :] * eta) / (eta + multiplier)

    # Backward Pass
    print('****'*2)
    print('UPDATING POLICY...')
    print('****'*2)

    # print(cv[:, :7])
    # input('fadsfds')

    idx_x = slice(dX)
    idx_u = slice(dX, dX+dU)

    Fm = dynamics.Fm
    fv = dynamics.fv

    # Allocate.
    Vxx = np.zeros((T, dX, dX))
    Vx = np.zeros((T, dX))
    Qtt = np.zeros((T, dX+dU, dX+dU))
    Qt = np.zeros((T, dX+dU))

    new_K = np.zeros((T, dU, dX))
    new_k = np.zeros((T, dU))
    new_pS = np.zeros((T, dU, dU))
    new_ipS = np.zeros((T, dU, dU))
    new_cpS = np.zeros((T, dU, dU))

    for t in range(T - 1, -1, -1):
        # Add in the cost.
        Qtt[t] = fCm[t, :, :]  # (X+U) x (X+U)
        Qt[t] = fcv[t, :]  # (X+U) x 1
        # print('Qt', Qt[t])
        # input("fjdsalfsdf")

        # Add in the value function from the next time step.
        if t < T - 1:
            multiplier = 1.0

            Qtt[t] += Fm[t, :, :].T.dot(Vxx[t+1, :, :]).dot(Fm[t, :, :])
            Qt[t] += Fm[t, :, :].T.dot(Vx[t+1, :] +
                                       Vxx[t+1, :, :].dot(fv[t, :]))

        # Symmetrize quadratic component.
        Qtt[t] = 0.5 * (Qtt[t] + Qtt[t].T)

        inv_term = Qtt[t, idx_u, idx_u]  # Quu
        k_term = Qt[t, idx_u]  # Qu
        K_term = Qtt[t, idx_u, idx_x]  # Qxu
        # # For cos_per_step
        # inv_term = Qtt[t, idx_u, idx_u] + policy.inv_pol_covar[t]
        # k_term = Qt[t, idx_u] - policy.inv_pol_covar[t].dot(policy.k[t])
        # K_term = Qtt[t, idx_u, idx_x] - policy.inv_pol_covar[t].dot(policy.K[t])

        # Compute Cholesky decomposition of Q function action component.
        U = sp.linalg.cholesky(inv_term)
        L = U.T

        # Update the Trajectory Distribution Parameters
        # Store conditional covariance, inverse, and Cholesky.
        new_ipS[t, :, :] = inv_term  # Quu
        new_pS[t, :, :] = sp.linalg.solve_triangular(
            U, sp.linalg.solve_triangular(L, np.eye(dU), lower=True)
        )  # Quu^-1
        new_cpS[t, :, :] = sp.linalg.cholesky(
            new_pS[t, :, :]
        )

        # Compute mean terms.
        # print(policy.k[t, :])
        new_k[t, :] = -sp.linalg.solve_triangular(
            U, sp.linalg.solve_triangular(L, k_term, lower=True)
        )
        new_K[t, :, :] = -sp.linalg.solve_triangular(
            U, sp.linalg.solve_triangular(L, K_term, lower=True)
        )
        # if t == 0:
        #     print(np.round(new_K[t, :7, :7], 2))
        #     print('--')
        #     print(np.round(new_k[t, :7], 2))
        #     input('dfasf')
        # else:
        #     print(np.round(new_k[t, :7], 2))

        # Compute value function.
        Vxx[t, :, :] = Qtt[t, idx_x, idx_x] + \
                       Qtt[t, idx_x, idx_u].dot(new_K[t, :, :])
        Vx[t, :] = Qt[t, idx_x] + \
                   Qtt[t, idx_x, idx_u].dot(new_k[t, :])

        # # Option: cons_per_step or not upd_bw
        # Vxx[t, :, :] = Qtt[t, idx_x, idx_x] + \
        #                policy.K[t].T.dot(Qtt[t, idx_u, idx_u]).dot(policy.K[t]) + \
        #                (2 * Qtt[t, idx_x, idx_u]).dot(policy.K[t, :, :])
        # Vx[t, :] = Qt[t, idx_x].T + \
        #            Qt[t, idx_u].T.dot(policy.K[t]) + \
        #            policy.k[t].T.dot(Qtt[t, idx_u, idx_u]).dot(policy.K[t]) + \
        #            Qtt[t, idx_x, idx_u].dot(policy.k[t, :])

        # Symmetrize quadratic component.
        Vxx[t, :, :] = 0.5 * (Vxx[t, :, :] + Vxx[t, :, :].T)

        # Now update
        policy.K = new_K
        policy.k = new_k
        policy.pol_covar = new_pS
        policy.inv_pol_covar = new_ipS
        policy.chol_pol_covar = new_cpS

    # ############### #
    # Evaluate Policy #
    # ############### #
    # Sample and show cost
    noise = np.zeros((T, dU))
    sample = Sample(env, T)
    all_actions = np.zeros((T, dU))
    all_states = np.zeros((T, dX))
    all_obs = np.zeros((T, dX))

    obs0 = env.reset()
    for t in range(T):
        state = env.get_state()# - obs0
        obs = env.get_observation()# - obs0
        action = policy.eval(state.copy(), obs.copy(),
                             t, noise[t].copy())
        env.step(action)

        all_states[t, :] = state
        all_obs[t, :] = obs
        all_actions[t, :] = action
    sample.set_acts(all_actions)
    sample.set_obs(all_obs)
    sample.set_states(all_states)
    sample.set_noise(noise)

    cost_output = cost_fcn.eval(sample)
    print('***\n' * 5)
    iter_costs[ii+1] = np.sum(cost_output[0])
    print('SAMPLE COST', np.sum(cost_output[0]))
    print('***\n' * 5)

plt.plot(iter_costs[1:])
plt.show(block=False)

# input("Press a key to start sampling")
# resample = True
# while resample:
#     env.set_rendering(True)
#     obs0 = env.reset()
#     for t in range(T):
#         state = env.get_state()# - obs0
#         obs = env.get_observation()# - obs0
#         action = policy.eval(state.copy(), obs.copy(),
#                              t, noise[t].copy())
#         env.step(action)
#     answer = input('Do you want to finish the script? (y/Y)')
#     if answer.lower == 'y':
#         resample = False
#
input('Press a key to close the script')
