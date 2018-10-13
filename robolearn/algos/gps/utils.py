import numpy as np
import scipy.ndimage as sp_ndimage
from robolearn.algos.gps.policies.lin_gauss_policy import LinearGaussianPolicy


def gauss_fit_joint_prior(pts, mu0, Phi, m, n0, dwts, dX, dU, sig_reg,
                          max_var=None):
    """
    Perform Gaussian fit to data with a prior.
    :param pts:
    :param mu0:
    :param Phi:
    :param m:
    :param n0:
    :param dwts:
    :param dX:
    :param dU:
    :param sig_reg: Sigma regularization
    :return:
    """
    # Build weights matrix.
    D = np.diag(dwts)

    # Compute empirical mean and covariance.
    mun = np.sum((pts.T * dwts).T, axis=0)
    diff = pts - mun
    empsig = diff.T.dot(D).dot(diff)
    empsig = 0.5 * (empsig + empsig.T)

    # MAP estimate of joint distribution.
    N = dwts.shape[0]
    mu = mun
    sigma = (N * empsig + Phi + (N * m) / (N + m) * np.outer(mun - mu0, mun - mu0)) / (N + n0)
    sigma = 0.5 * (sigma + sigma.T)

    if max_var is not None:
        sigma = np.clip(sigma, -max_var, max_var)

    # Add sigma regularization.
    sigma += sig_reg

    # Conditioning to get dynamics.
    fd = np.linalg.solve(sigma[:dX, :dX], sigma[:dX, dX:dX+dU]).T
    fc = mu[dX:dX+dU] - fd.dot(mu[:dX])
    dynsig = sigma[dX:dX+dU, dX:dX+dU] - fd.dot(sigma[:dX, :dX]).dot(fd.T)
    dynsig = 0.5 * (dynsig + dynsig.T)

    return fd, fc, dynsig


def generate_noise(T, dU, hyperparams):
    """
    Generate a T x dU gaussian-distributed noise vector. This will
    approximately have mean 0 and variance noise_var_scale, ignoring smoothing.

    Args:
        T: Number of time steps.
        dU: Dimensionality of actions.
    Hyperparams:
        smooth: Whether or not to perform smoothing of noise.
        var : If smooth=True, applies a Gaussian filter with this
            variance.
        renorm : If smooth=True, renormalizes data to have variance 1
            after smoothing.
    """
    smooth, var = hyperparams['smooth_noise'], hyperparams['smooth_noise_var']
    renorm = hyperparams['smooth_noise_renormalize']

    if 'noise_var_scale' not in hyperparams:
        hyperparams['noise_var_scale'] = 1

    if not issubclass(type(hyperparams['noise_var_scale']), list) and \
            not issubclass(type(hyperparams['noise_var_scale']), np.ndarray):
        scale = np.tile(hyperparams['noise_var_scale'], dU)
    elif len(hyperparams['noise_var_scale']) == dU:
        scale = hyperparams['noise_var_scale']
    else:
        raise TypeError("noise_var_scale size (%d) does not match dU (%d)" % (len(hyperparams['noise_var_scale']), dU))

    # np.random.seed(5)

    # Generate noise and scale
    noise = np.random.randn(T, dU)

    temp_labels = list()
    temp_noise_list = list()
    temp_noise_list.append(noise.copy())
    temp_labels.append('Noise')

    if smooth:
        # Smooth noise. This violates the controller assumption, but
        # might produce smoother motions.
        for i in range(dU):
            noise[:, i] = sp_ndimage.filters.gaussian_filter(noise[:, i], var)
        temp_noise_list.append(noise.copy())
        temp_labels.append('Filtered')
        if renorm:
            variance = np.var(noise, axis=0)
            noise = noise * np.sqrt(scale) / np.sqrt(variance)

        temp_noise_list.append(noise.copy())
        temp_labels.append('Renorm')

    else:
        noise = noise*np.sqrt(scale)

    # plot_multi_info(temp_noise_list, block=True, cols=2, legend=True,
    #                 labels=temp_labels)

    return noise


class BundleType(object):
    """
    This class bundles many fields, similar to a record or a mutable
    namedtuple.
    """
    def __init__(self, variables):
        for var, val in variables.items():
            object.__setattr__(self, var, val)

    # Freeze fields so new ones cannot be set.
    def __setattr__(self, key, value):
        if not hasattr(self, key):
            raise AttributeError("%r has no attribute %s" % (self, key))
        object.__setattr__(self, key, value)


class IterationData(BundleType):
    """ Collection of iteration variables. """
    def __init__(self):
        variables = {
            'sample_list': None,     # List of samples for the current iteration.
            'traj_info': None,       # Current TrajectoryInfo object.
            'pol_info': None,        # Current PolicyInfo object.
            'traj_distr': None,      # Current trajectory distribution. \bar{p}_i(u_t|x_t)
            'new_traj_distr': None,  # Updated trajectory distribution. p_i(u_t|x_t)
            'cs': None,              # Sample costs of the current iteration.
            'cost_compo': None,       # Sample cost compositions of the current iteration.
            'eta': 1.0,              # Dual variable used in LQR backward pass.
            'omega': 0.0,            # Dual variable used in LQR backward pass (Dualism).
            'nu': 0.0,               # Dual variable used in LQR backward pass (Dualism).
            'step_mult': 1.0,        # KL step multiplier for the current iteration.
            'good_step_mult': 1.0,   # KL good multiplier for the current iteration (Dualism).
            'bad_step_mult': 1.0,    # KL bad multiplier for the current iteration (Dualism).
            'good_traj_distr': None,  # Good traj_distr (Dualism).
            'bad_traj_distr': None,  # Bad traj_distr (Dualism).
        }
        BundleType.__init__(self, variables)


class TrajectoryInfo(BundleType):
    """ Collection of trajectory-related variables. """
    def __init__(self):
        variables = {
            'dynamics': None,   # Dynamics object for the current iteration.
            'x0mu': None,       # Mean for the initial state, used by the dynamics.
            'x0sigma': None,    # Covariance for the initial state distribution.
            'cc': None,         # Cost estimate constant term.
            'cv': None,         # Cost estimate vector term.
            'Cm': None,         # Cost estimate matrix term.
            'last_kl_step': float('inf'),  # KL step of the previous iteration.
        }
        BundleType.__init__(self, variables)


class PolicyInfo(BundleType):
    """ Collection of policy-related variables. """
    def __init__(self, **hyperparams):
        T, dU, dX = hyperparams['T'], hyperparams['dU'], hyperparams['dX']
        variables = {
            'lambda_k': np.zeros((T, dU)),        # Dual variable (Lagrange multiplier vectors) for k.
            'lambda_K': np.zeros((T, dU, dX)),    # Dual variables (Lagrange multiplier vectors) for K.
            'pol_wt': hyperparams['init_pol_wt'] * np.ones(T),  # Policy weight.
            'pol_mu': None,                       # Mean of the current policy output.
            'pol_sig': None,                      # Covariance of the current policy output.
            'pol_K': np.zeros((T, dU, dX)),       # Policy linearization K matrix.
            'pol_k': np.zeros((T, dU)),           # Policy linearization k vector.
            'pol_S': np.zeros((T, dU, dU)),       # Policy linearization covariance.
            'chol_pol_S': np.zeros((T, dU, dU)),  # Policy linearization Cholesky decomp of covar.
            'prev_kl': None,                      # Previous KL divergence.
            'init_kl': None,                      # The initial KL divergence, before the iteration.
            'policy_samples': [],                 # List of current policy samples.
            'policy_prior': None,                 # Current prior for policy linearization.
        }
        BundleType.__init__(self, variables)

    def traj_distr(self):
        """
        Create a trajectory distribution object from policy info
        (Policy linearization)
        """
        T, dU, dX = self.pol_K.shape
        # Compute inverse policy covariances.
        inv_pol_S = np.empty_like(self.chol_pol_S)
        for t in range(T):
            inv_pol_S[t, :, :] = \
                np.linalg.solve(self.chol_pol_S[t, :, :],
                                np.linalg.solve(self.chol_pol_S[t, :, :].T,
                                                np.eye(dU)))
        return LinearGaussianPolicy(self.pol_K, self.pol_k, self.pol_S,
                                    self.chol_pol_S, inv_pol_S)


class DualityInfo(BundleType):
    """ Collection of duality-trajectory-related variables. """
    def __init__(self):
        variables = {
            'sample_list': None,
            'samples_cost': None,
            'traj_cost': None,
            'traj_dist': None,
            'pol_info': None,        # Policy-related PolicyInfo object.
            'experience_buffer': None,
        }
        BundleType.__init__(self, variables)


