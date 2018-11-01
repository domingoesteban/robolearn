import numpy as np
from gym.spaces import Box

from robolearn.core.serializable import Serializable
from robolearn.envs.proxy_env import ProxyEnv


class NormalizedBoxEnv(ProxyEnv, Serializable):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """
    def __init__(
            self,
            env,
            reward_scale=1.,
            normalize_obs=False,
            online_normalization=False,
            obs_mean=None,
            obs_var=None,
            obs_alpha=0.001,
    ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self._wrapped_env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

        # Observation Space
        if normalize_obs is True and online_normalization is True:
            raise AttributeError

        self._normalize_obs = normalize_obs
        self._online_normalize_obs = online_normalization

        if self._normalize_obs or self._online_normalize_obs:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_var is None:
                obs_var = np.ones_like(env.observation_space.low)
            else:
                obs_var = np.array(obs_var)

        self._obs_mean = obs_mean
        self._obs_var = obs_var
        self._obs_alpha = obs_alpha

        self._obs_mean_diff = np.zeros_like(env.observation_space.low)
        self._obs_n = 0

        # Action Space
        if isinstance(self._wrapped_env.action_space, Box):
            ub = np.ones(self._wrapped_env.action_space.shape)
            self.action_space = Box(-1 * ub, ub, dtype=np.float32)

        # Reward Scale
        self._reward_scale = reward_scale

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and variance already set. To "
                            "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_var = np.var(obs_batch, axis=0)

    def _update_obs_estimate(self, obs):
        flat_obs = obs.flatten()
        self._obs_mean = (1 - self._obs_alpha) * self._obs_mean + self._obs_alpha * flat_obs
        self._obs_var = (1 - self._obs_alpha) * self._obs_var + self._obs_alpha * np.square(flat_obs - self._obs_mean)

    def _apply_normalize_obs(self, obs):
        # return (obs - self._obs_mean) / (self._obs_std + 1e-8)
        return (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

    def _apply_online_normalize_obs(self, obs):
        self._obs_n += 1.
        last_mean = self._obs_mean
        self._obs_mean += (obs-self._obs_mean)/self._obs_n
        self._obs_mean_diff += (obs-last_mean)*(obs-self._obs_mean)
        self._obs_var = np.sqrt(np.clip(self._obs_mean_diff / self._obs_n,
                                        1.e-2, None))
        return self._apply_normalize_obs(obs)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        # Add these explicitly in case they were modified
        d["_obs_mean"] = self._obs_mean
        d["_obs_var"] = self._obs_var
        d["_reward_scale"] = self._reward_scale
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self._obs_mean = d["_obs_mean"]
        self._obs_var = d["_obs_var"]
        self._reward_scale = d["_reward_scale"]

    @property
    def obs_mean(self):
        return self._obs_mean

    @property
    def obs_var(self):
        return self._obs_var

    @property
    def reward_scale(self):
        return self._reward_scale

    def reset(self, *args, **kwargs):
        obs = self._wrapped_env.reset(*args, **kwargs)
        if self._normalize_obs:
            self._update_obs_estimate(obs)
            return self._apply_normalize_obs(obs)
        elif self._online_normalize_obs:
            return self._apply_online_normalize_obs(obs)
        else:
            return obs

    def step(self, action):
        if isinstance(self._wrapped_env.action_space, Box):
            # Scale Action
            lb = self._wrapped_env.action_space.low
            ub = self._wrapped_env.action_space.high
            scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
            scaled_action = np.clip(scaled_action, lb, ub)
        else:
            scaled_action = action

        # Interact with Environment
        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step

        # Normalize Observation
        if self._normalize_obs:
            self._update_obs_estimate(next_obs)
            next_obs = self._apply_normalize_obs(next_obs)
        elif self._online_normalize_obs:
            next_obs = self._apply_online_normalize_obs(next_obs)

        # Scale Reward
        reward = reward * self._reward_scale

        return next_obs, reward, done, info

    def seed(self, *args, **kwargs):
        self._wrapped_env.seed(*args, **kwargs)

    @property
    def online_normalization(self):
        return self._online_normalize_obs

    @property
    def normalize_obs(self):
        return self._normalize_obs

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

    def log_diagnostics(self, paths, **kwargs):
        if hasattr(self._wrapped_env, "log_diagnostics"):
            return self._wrapped_env.log_diagnostics(paths, **kwargs)
        else:
            return None

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)
