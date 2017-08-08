"""
This script helps to tune the noise hyperparameters.
"""

import numpy as np
import scipy.ndimage as sp_ndimage
from robolearn.utils.plot_utils import plot_multi_info

# Noise hyperparams
noise_var_scale = 1.0e-0         # Scale to Gaussian noise: N(0,1)*sqrt(noise_var_scale)
smooth_noise = True              # Apply Gaussian filter to noise generated
smooth_noise_var = 5.0e-0        # Variance to apply to Gaussian Filter
smooth_noise_renormalize = True  # Renormalize smooth noise to have variance=1
T = 500
dU = 7

if not issubclass(type(noise_var_scale), list) or not issubclass(type(noise_var_scale), np.ndarray):
    scale = np.tile(noise_var_scale, dU)
elif len(noise_var_scale) == dU:
    scale = noise_var_scale
else:
    raise TypeError("noise_var_scale size (%d) does not match dU (%d)" % (len(noise_var_scale), dU))

# Generate noise and scale
noise = np.random.randn(T, dU)*np.sqrt(scale)

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
        noise[:, i] = sp_ndimage.filters.gaussian_filter(noise[:, i], smooth_noise_var)
    temp_noise_list.append(noise.copy())
    labels.append('Smooth noise')
    print('')
    print("*"*20)
    print("Smooth noise Max:%s" % np.max(noise, axis=0))
    print("Smooth noise Min:%s" % np.min(noise, axis=0))

    if smooth_noise_renormalize:
        variance = np.var(noise, axis=0)
        noise = noise / np.sqrt(variance)

        temp_noise_list.append(noise.copy())
        labels.append('Smooth noise renormalized')

        print('')
        print("*"*20)
        print("Smooth noise renormalized Max:%s" % np.max(noise, axis=0))
        print("Smooth noise renormalized Min:%s" % np.min(noise, axis=0))

plot_multi_info(temp_noise_list, block=True, cols=3, legend=True, labels=labels)
