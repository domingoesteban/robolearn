# Based on lin_gauss_init from github.com:cbfinn/gps.git
""" Initializations for linear Gaussian controllers. """
import copy
import numpy as np
import scipy as sp

from robolearn.utils.dynamics.dynamics_utils import guess_dynamics
from robolearn.policies.lin_gauss_policy import LinearGaussianPolicy
from robolearn.algos.gps.gps_utils import gauss_fit_joint_prior


# Initial Linear Gaussian Trajectory Distributions, PD-based initializer.
# Note, PD is the default initializer type.
INIT_LG_PD = {
    'init_var': 10.0,
    'pos_gains': 10.0,  # position gains
    'vel_gains_mult': 0.01,  # velocity gains multiplier on pos_gains
    'init_action_offset': None,
}
# Initial Linear Gaussian Trajectory distribution, LQR-based initializer.
INIT_LG_LQR = {
    'init_var': 1.0,
    'stiffness': 1.0,
    'stiffness_vel': 0.5,
    'final_weight': 1.0,
    # Parameters for guessing dynamics
    'init_acc': [],  # dU vector of accelerations, default zeros.
    'init_gains': [],  # dU vector of gains, default ones.
}


# Original code
def init_lqr(hyperparams):
    """
    Return initial gains for a time-varying linear Gaussian controller that tries to hold the initial position.
    """
    config = copy.deepcopy(INIT_LG_LQR)
    config.update(hyperparams)

    x0, dX, dU = config['x0'], config['dX'], config['dU']
    dt, T = config['dt'], config['T']

    #TODO: Use packing instead of assuming which indices are the joint angles.

    # Notation notes:
    # L = loss, Q = q-function (dX+dU dimensional),
    # V = value function (dX dimensional), F = dynamics
    # Vectors are lower-case, matrices are upper case.
    # Derivatives: x = state, u = action, t = state+action (trajectory).
    # The time index is denoted by _t after the above.
    # Ex. Ltt_t = Loss, 2nd derivative (w.r.t. trajectory), indexed by time t.

    # Constants.
    idx_x = slice(dX)  # Slices out state.
    idx_u = slice(dX, dX+dU)  # Slices out actions.

    if len(config['init_acc']) == 0:
        config['init_acc'] = np.zeros(dU)

    if len(config['init_gains']) == 0:
        config['init_gains'] = np.ones(dU)

    # Set up simple linear dynamics model.
    Fd, fc = guess_dynamics(config['init_gains'], config['init_acc'], dX, dU, dt)

    # Setup a cost function based on stiffness.
    # Ltt = (dX+dU) by (dX+dU) - Hessian of loss with respect to trajectory at a single timestep.
    Ltt = np.diag(np.hstack([config['stiffness'] * np.ones(dU),
                             config['stiffness'] * config['stiffness_vel'] * np.ones(dU),
                             np.zeros(dX - dU*2),
                             np.ones(dU)
                             ]))
    Ltt = Ltt / config['init_var']  # Cost function - quadratic term.
    lt = -Ltt.dot(np.r_[x0, np.zeros(dU)])  # Cost function - linear term.

    # Perform dynamic programming.
    K = np.zeros((T, dU, dX))  # Controller gains matrix.
    k = np.zeros((T, dU))  # Controller bias term.
    PSig = np.zeros((T, dU, dU))  # Covariance of noise.
    cholPSig = np.zeros((T, dU, dU))  # Cholesky decomposition.
    invPSig = np.zeros((T, dU, dU))  # Inverse of covariance.
    vx_t = np.zeros(dX)  # Vx = dV/dX. Derivative of value function wrt to X at time t.
    Vxx_t = np.zeros((dX, dX))  # Vxx = ddV/dXdX at time t.

    # LQR backward pass.
    for t in range(T - 1, -1, -1):
        # Compute Q function at this step.
        if t == (T - 1):
            Ltt_t = config['final_weight'] * Ltt
            lt_t = config['final_weight'] * lt
        else:
            Ltt_t = Ltt
            lt_t = lt
        # Qtt = (dX+dU) by (dX+dU) 2nd Derivative of Q-function with respect to trajectory (dX+dU).
        Qtt_t = Ltt_t + Fd.T.dot(Vxx_t).dot(Fd)
        # Qt = (dX+dU) 1st Derivative of Q-function with respect to trajectory (dX+dU).
        qt_t = lt_t + Fd.T.dot(vx_t + Vxx_t.dot(fc))

        # Compute preceding value function.
        U = sp.linalg.cholesky(Qtt_t[idx_u, idx_u])
        L = U.T

        invPSig[t, :, :] = Qtt_t[idx_u, idx_u]
        PSig[t, :, :] = sp.linalg.solve_triangular(U,
                                                   sp.linalg.solve_triangular(L, np.eye(dU), lower=True))
        cholPSig[t, :, :] = sp.linalg.cholesky(PSig[t, :, :])
        K[t, :, :] = -sp.linalg.solve_triangular(U,
                                                 sp.linalg.solve_triangular(L, Qtt_t[idx_u, idx_x], lower=True))
        k[t, :] = -sp.linalg.solve_triangular(U,
                                              sp.linalg.solve_triangular(L, qt_t[idx_u], lower=True))
        Vxx_t = Qtt_t[idx_x, idx_x] + Qtt_t[idx_x, idx_u].dot(K[t, :, :])
        vx_t = qt_t[idx_x] + Qtt_t[idx_x, idx_u].dot(k[t, :])
        Vxx_t = 0.5 * (Vxx_t + Vxx_t.T)

    return LinearGaussianPolicy(K, k, PSig, cholPSig, invPSig)


#TODO: Fix docstring
def init_pd(hyperparams):
    """
    This function initializes the linear-Gaussian controller as a
    proportional-derivative (PD) controller with Gaussian noise. The
    position gains are controlled by the variable pos_gains, velocity
    gains are controlled by pos_gains*vel_gans_mult.
    """
    config = copy.deepcopy(INIT_LG_PD)
    config.update(hyperparams)

    dU, dJoints, dX = config['dU'], config['dJoints'], config['dX']
    x0, T = config['x0'], config['T']
    dDistance = config['dDistance']

    if not issubclass(type(config['pos_gains']), list) and not issubclass(type(config['pos_gains']), np.ndarray):
        pos_gains = np.tile(config['pos_gains'], dU)
    elif len(config['pos_gains']) == dU:
        pos_gains = config['pos_gains']
    else:
        raise TypeError("noise_var_scale size (%d) does not match dU (%d)" % (len(config['pos_gains']), dU))

    # Choose initialization mode.
    Kp = 1.0
    Kv = config['vel_gains_mult']
    if dU < dJoints:
        K = -np.diag(pos_gains).dot(np.hstack([np.eye(dU) * Kp, np.zeros((dU, dJoints-dU)),
                                               np.eye(dU) * Kv, np.zeros((dU, dJoints-dU))]))
        K = np.tile(K, [T, 1, 1])
    else:
        K = -np.diag(pos_gains).dot(np.hstack([np.eye(dU) * Kp, np.eye(dU) * Kv,
                                               np.zeros((dU, dX - dU*2))]))
        K = np.tile(K, [T, 1, 1])
    if config['state_to_pd'] == 'distance':
        k = np.tile(-K[0, :, :].dot(x0), [T, 1])
    else:
        k = np.tile(-K[0, :, :].dot(x0), [T, 1])

    #k = np.tile(K[0, :, :].dot(x0), [T, 1])
    PSig = config['init_var'] * np.tile(np.eye(dU), [T, 1, 1])
    cholPSig = np.sqrt(config['init_var']) * np.tile(np.eye(dU), [T, 1, 1])
    invPSig = (1.0 / config['init_var']) * np.tile(np.eye(dU), [T, 1, 1])

    return LinearGaussianPolicy(K, k, PSig, cholPSig, invPSig)


def init_demos(hyperparams):
    """
    Initializes the linear-Gaussian controller from demonstrations.
    :param hyperparams: SampleList, 
    :return: 
    """
    config = copy.deepcopy(INIT_LG_PD)
    config.update(hyperparams)

    samples = config['sample_lists']

    X = samples.get_states()
    obs = samples.get_obs()
    U = samples.get_actions()

    N, T, dX = X.shape
    dU = U.shape[2]
    if N == 1:
        raise ValueError("Cannot fit dynamics on 1 sample")

    # import matplotlib.pyplot as plt
    # plt.plot(U[1, :])
    # plt.show(block=False)
    # raw_input(U.shape)

    pol_mu = U
    pol_sig = np.zeros((N, T, dU, dU))

    for t in range(T):
        # Using only diagonal covariances
        pol_sig[:, t, :, :] = np.tile(np.diag(np.diag(np.cov(U[:, t, :].T))), (N, 1, 1))

    # Collapse policy covariances. (This is only correct because the policy doesn't depend on state).
    pol_sig = np.mean(pol_sig, axis=0)

    # Allocate.
    pol_K = np.zeros([T, dU, dX])
    pol_k = np.zeros([T, dU])
    pol_S = np.zeros([T, dU, dU])
    chol_pol_S = np.zeros([T, dU, dU])
    inv_pol_S = np.zeros([T, dU, dU])

    # Update policy prior.
    def eval_prior(Ts, Ps):
        strength = 1e-4
        dX, dU = Ts.shape[-1], Ps.shape[-1]
        prior_fd = np.zeros((dU, dX))
        prior_cond = 1e-5 * np.eye(dU)
        sig = np.eye(dX)
        Phi = strength * np.vstack([np.hstack([sig, sig.dot(prior_fd.T)]),
                                    np.hstack([prior_fd.dot(sig), prior_fd.dot(sig).dot(prior_fd.T) + prior_cond])])
        return np.zeros(dX+dU), Phi, 0, strength


    # Fit linearization with least squares regression
    dwts = (1.0 / N) * np.ones(N)
    for t in range(T):
        Ts = X[:, t, :]
        Ps = pol_mu[:, t, :]
        Ys = np.concatenate([Ts, Ps], axis=1)
        # Obtain Normal-inverse-Wishart prior.
        mu0, Phi, mm, n0 = eval_prior(Ts, Ps)
        sig_reg = np.zeros((dX+dU, dX+dU))
        # Slightly regularize on first timestep.
        if t == 0:
            sig_reg[:dX, :dX] = 1e-8
        pol_K[t, :, :], pol_k[t, :], pol_S[t, :, :] = gauss_fit_joint_prior(Ys, mu0, Phi, mm, n0, dwts, dX, dU, sig_reg)
    pol_S += pol_sig  # Add policy covariances mean

    for t in range(T):
        chol_pol_S[t, :, :] = sp.linalg.cholesky(pol_S[t, :, :])
        inv_pol_S[t, :, :] = np.linalg.inv(pol_S[t, :, :])

    return LinearGaussianPolicy(pol_K, pol_k, pol_S, chol_pol_S, inv_pol_S)
