import numpy as np
import scipy as sp

from robolearn.algos.gps.gps_utils import gauss_fit_joint_prior
from robolearn.policies.lin_gauss_policy import LinearGaussianPolicy


def fit_linear_gaussian_policy(sample_list, min_variance, state_or_obs='state'):
    """
    Fits a Linear Gaussian Policy (trajectory distribution) with
    least squared regression and Normal-inverse-Wishart prior.
    :param sample_list: Sample list object
    :param min_variance: minimum variance of action commands
    :return:
    """
    samples = sample_list

    # Get information from sample list
    if state_or_obs == 'state':
        X = samples.get_states()
    elif state_or_obs == 'obs':
        X = samples.get_obs()
    else:
        raise ValueError("Wrong option. It should be 'state' or 'obs'")

    U = samples.get_actions()

    N, T, dX = X.shape
    dU = U.shape[2]
    # if N == 1:
    #     raise ValueError("Cannot fit traj_dist on 1 sample")

    pol_mu = U
    pol_sig = np.zeros((N, T, dU, dU))

    print("TODO: WE ARE GIVING MIN GOOD/BAD VARIANCE TO TVLGC")
    print("TODO: TVLGC FITTING ONLY FROM SAMPLE ACTIONS (NO PRIOR)")
    for t in range(T):
        # pol_sig[:, t, :, :] = np.tile(np.diag(np.diag(np.cov(U[:, t, :].T))), (N, 1, 1))
        if N == 1:
            new_diag = min_variance
        else:
            # Using only diagonal covariances
            current_diag = np.diag(np.cov(U[:, t, :].T))
            new_diag = np.max(np.vstack((current_diag, min_variance)), axis=0)
        pol_sig[:, t, :, :] = np.tile(np.diag(new_diag), (N, 1, 1))

    # Collapse policy covariances. (This is only correct because the policy
    # doesn't depend on state).
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
        pol_K[t, :, :], pol_k[t, :], pol_S[t, :, :] = \
            gauss_fit_joint_prior(Ys, mu0, Phi, mm, n0, dwts,
                                  dX, dU, sig_reg)
    pol_S += pol_sig  # Add policy covariances mean

    for t in range(T):
        chol_pol_S[t, :, :] = sp.linalg.cholesky(pol_S[t, :, :])
        inv_pol_S[t, :, :] = np.linalg.inv(pol_S[t, :, :])

    return LinearGaussianPolicy(pol_K, pol_k, pol_S, chol_pol_S, inv_pol_S)


def lqr_forward(traj_distr, traj_info):
    """
    Perform LQR forward pass. Computes state-action marginals from dynamics and
    policy.
    Args:
        traj_distr: A linear Gaussian policy object.
        traj_info: A TrajectoryInfo object.
    Returns:
        mu: A T x dX mean action vector.
        sigma: A T x dX x dX covariance matrix.
    """
    # Compute state-action marginals from specified conditional
    # parameters and current traj_info.
    T = traj_distr.T
    dU = traj_distr.dU
    dX = traj_distr.dX

    # Constants.
    idx_x = slice(dX)

    # Allocate space.
    sigma = np.zeros((T, dX+dU, dX+dU))
    mu = np.zeros((T, dX+dU))

    # Pull out dynamics.
    Fm = traj_info.dynamics.Fm
    fv = traj_info.dynamics.fv
    dyn_covar = traj_info.dynamics.dyn_covar

    # Set initial state covariance and mean
    sigma[0, idx_x, idx_x] = traj_info.x0sigma
    mu[0, idx_x] = traj_info.x0mu

    for t in range(T):
        sigma[t, :, :] = np.vstack([
            np.hstack([sigma[t, idx_x, idx_x],
                       sigma[t, idx_x, idx_x].dot(traj_distr.K[t, :, :].T)]),
            np.hstack([traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]),
                       traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]).dot(traj_distr.K[t, :, :].T)
                       + traj_distr.pol_covar[t, :, :]])])

        # u_t = p(u_t | x_t)
        mu[t, :] = np.hstack([mu[t, idx_x], traj_distr.K[t, :, :].dot(mu[t, idx_x]) + traj_distr.k[t, :]])

        if t < T - 1:
            # x_t+1 = p(x_t+1 | x_t, u_t)
            sigma[t+1, idx_x, idx_x] = Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T) + dyn_covar[t, :, :]
            mu[t+1, idx_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]
    return mu, sigma

