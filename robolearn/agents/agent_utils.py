import numpy as np
import scipy.ndimage as sp_ndimage
from robolearn.utils.plot_utils import plot_multi_info


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

    temp_noise_list = list()
    temp_noise_list.append(noise.copy())

    if smooth:
        # Smooth noise. This violates the controller assumption, but
        # might produce smoother motions.
        for i in range(dU):
            noise[:, i] = sp_ndimage.filters.gaussian_filter(noise[:, i], var)
        temp_noise_list.append(noise.copy())
        if renorm:
            variance = np.var(noise, axis=0)
            noise = noise * np.sqrt(scale) / np.sqrt(variance)

        temp_noise_list.append(noise.copy())

    else:
        noise = noise*np.sqrt(scale)

    # plot_multi_info(temp_noise_list, block=True, cols=2, legend=True)

    return noise
