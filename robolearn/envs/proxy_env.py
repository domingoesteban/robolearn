import numpy as np
from gym import Env
from robolearn.utils.serializable import Serializable


class ProxyEnv(Serializable, Env):
    def __init__(self, wrapped_env):
        Serializable.quick_init(self, locals())
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, *args, **kwargs):
        return self._wrapped_env.reset(*args, **kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self._wrapped_env.close(*args, **kwargs)

    def seed(self, *args, **kwargs):
        return self._wrapped_env.seed(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        if hasattr(self._wrapped_env, 'log_diagnostics'):
            self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    @property
    def action_dim(self):
        return np.prod(self.action_space.shape)

    @property
    def obs_dim(self):
        return np.prod(self.observation_space.shape)
