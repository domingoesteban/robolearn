import numpy as np
import numpy.random as nr
import scipy.ndimage as sp_ndimage

from robolearn.utils.exploration_strategies.base import ExplorationStrategy
from robolearn.utils.serializable import Serializable


class SmoothNoiseStrategy(ExplorationStrategy, Serializable):
    """
    Based on Finn gps implementation.
    """

    def __init__(
            self,
            action_space,
            horizon,
            smooth=True,
            renormalize=True,
            sigma=10.0,
            sigma_scale=None,
    ):
        Serializable.quick_init(self, locals())

        self._action_space = action_space
        self.low = action_space.low
        self.high = action_space.high

        self._horizon = horizon

        self._smooth = smooth
        self._renormalize = renormalize
        self._sigma = sigma

        if sigma_scale is None:
            self._sigma_scale = np.ones(self.action_dim)
        else:
            # Check if iterable
            try:
                iter(sigma_scale)
                if len(sigma_scale) != self.action_dim:
                    raise ValueError("Sigma scale different than action dim"
                                     "(%02d != %02d)" % (sigma_scale,
                                                         self.action_dim))
                self._sigma_scale = sigma_scale
            except TypeError as te:
                self._sigma_scale = np.repeat(sigma_scale, self.action_dim)

        self.noise = None
        self.reset()

    @property
    def action_dim(self):
        return np.prod(self._action_space.shape)

    def reset(self):
        noise = nr.randn(self._horizon, self.action_dim)

        noises = list()
        noises.append(noise.copy())

        # Smooth noise
        if self._smooth:
            for i in range(self.action_dim):
                noise[:, i] = \
                    sp_ndimage.filters.gaussian_filter(noise[:, i], self._sigma)

            noises.append(noise.copy())

            # Renormalize
            if self._renormalize:
                variance = np.var(noise, axis=0)
                noise = noise * np.sqrt(self._sigma_scale) / np.sqrt(variance)

                noises.append(noise.copy())
        else:
            noise = noise*np.sqrt(self._sigma_scale)

        self.noise = noise

    def get_action(self, policy, *args, **kwargs):
        t = kwargs['t']
        kwargs['noise'] = self.noise[t, :]
        action, pol_info = policy.get_action(*args, **kwargs)

        return np.clip(action, self.low, self.high), pol_info

    def get_actions(self, t, observation, policy, **kwargs):
        raise NotImplementedError

