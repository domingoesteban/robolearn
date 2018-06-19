"""
This script helps to tune the noise hyperparameters.
"""

import numpy as np
import scipy.ndimage as sp_ndimage
from robolearn.old_utils.plot_utils import plot_multi_info

# Noise hyperparams
#noise_var_scale = 1.0e-0         # Scale to Gaussian noise: N(0,1)*sqrt(noise_var_scale)
#noise_var_scale = np.array([5.0e-1, 5.0e-1, 5.0e-1, 5.0e-1, 5.0e-2, 5.0e-2, 5.0e-2])
#init_noise_var_scale = np.array([1.5e+0, 1.5e+0, 1.5e+0, 1.5e+0, 5.0e-1, 5.0e-1, 5.0e-1])
init_noise_var_scale = np.ones(7)
final_noise_var_scale = None  # np.array([1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1])
smooth_noise = True              # Apply Gaussian filter to noise generated
smooth_noise_var = 5.0e-0        # Variance to apply to Gaussian Filter
smooth_noise_renormalize = True  # Renormalize smooth noise to have variance=1
smoth_noise_renormalize_val = init_noise_var_scale
T = 500
dU = 7

if not issubclass(type(init_noise_var_scale), list) and not issubclass(type(init_noise_var_scale), np.ndarray):
    initial_noise = np.tile(init_noise_var_scale, dU)
elif len(init_noise_var_scale) == dU:
    initial_noise = init_noise_var_scale
else:
    raise TypeError("init_noise_var_scale size (%d) does not match dU (%d)" % (len(init_noise_var_scale), dU))

if final_noise_var_scale is not None:
    if not issubclass(type(final_noise_var_scale), list) and not issubclass(type(final_noise_var_scale), np.ndarray):
        final_noise = np.tile(final_noise_var_scale, dU)
    elif len(final_noise_var_scale) == dU:
        final_noise = final_noise_var_scale
    else:
        raise TypeError("final_noise_var_scale size (%d) does not match dU (%d)" % (len(final_noise_var_scale), dU))

    scale = np.zeros([T, dU])

    for u in range(dU):
        scale[:, u] = np.linspace(initial_noise[u], final_noise[u], T)

else:
    scale = initial_noise


# Generate noise and scale
#noise = np.random.randn(T, dU)*np.sqrt(scale)
noise = np.random.randn(T, dU)

temp_noise_list = list()
labels = list()
temp_noise_list.append(noise.copy())
labels.append('Noise')

print("*"*30)
print("Noise Max:%s" % np.max(noise, axis=0))
print("Noise Min:%s" % np.min(noise, axis=0))
if smooth_noise:
    # Smooth noise. This violates the controller assumption, but
    # might produce smoother motions.
    for i in range(dU):
        noise[:, i] = sp_ndimage.filters.gaussian_filter(noise[:, i], sigma=smooth_noise_var)
    temp_noise_list.append(noise.copy())
    labels.append('Smooth noise')
    print('')
    print("*"*20)
    print("Smooth noise Max:%s" % np.max(noise, axis=0))
    print("Smooth noise Min:%s" % np.min(noise, axis=0))

    if smooth_noise_renormalize:
        variance = np.var(noise, axis=0)
        noise = noise * smoth_noise_renormalize_val / np.sqrt(variance)

        temp_noise_list.append(noise.copy())
        labels.append('Smooth noise renormalized')

        print('')
        print("*"*20)
        print("Smooth noise renormalized Max:%s" % np.max(noise, axis=0))
        print("Smooth noise renormalized Min:%s" % np.min(noise, axis=0))


plot_multi_info(temp_noise_list, block=True, cols=3, legend=True, labels=labels)
