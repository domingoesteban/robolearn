"""
Based on code from Marcin Andrychowicz and OpenAI baselines.
"""
import numpy as np


class RunningNormalizer(object):
    def __init__(self, epsilon=1e-4, shape=(), mean=0, var=1,
                 default_clip_range=np.inf):
        self.mean = mean + np.zeros(shape, np.float64)
        self.var = var*np.ones(shape, np.float64)
        self.count = epsilon
        self.default_clip_range = default_clip_range

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def normalize(self, x, clip_range=None):
        # Get clip range
        if clip_range is None:
            clip_range = self.default_clip_range
        # Get current mean and std

        mean, std = self.mean, self.std

        if x.ndim == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)

        # Return clipped values
        return np.clip((x - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        mean, std = self.mean, self.std
        if v.ndim == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return v * std + mean

    @property
    def std(self):
        return np.sqrt(self.var)


def update_mean_var_count_from_moments(mean, var, count,
                                       batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    # Calculate new mean
    new_mean = mean + delta*batch_count / tot_count

    # Calculate new variance
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count*count * batch_count / tot_count
    new_var = M2 / tot_count

    # Calculate new count
    new_count = tot_count

    return new_mean, new_var, new_count


class Normalizer(object):
    def __init__(
            self,
            size,
            eps=1e-8,
            default_clip_range=np.inf,
            mean=0,
            std=1,
    ):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        self.sum = np.zeros(self.size, np.float32)
        self.sumsq = np.zeros(self.size, np.float32)
        self.count = np.ones(1, np.float32)
        self.mean = mean + np.zeros(self.size, np.float32)
        self.std = std * np.ones(self.size, np.float32)
        self.synchronized = True

    def update(self, v):
        if v.ndim == 1:
            v = np.expand_dims(v, 0)
        assert v.ndim == 2
        assert v.shape[1] == self.size
        self.sum += v.sum(axis=0)
        self.sumsq += (np.square(v)).sum(axis=0)
        self.count[0] += v.shape[0]
        self.synchronized = False

    def normalize(self, v, clip_range=None):
        if not self.synchronized:
            self.synchronize()
        if clip_range is None:
            clip_range = self.default_clip_range
        mean, std = self.mean, self.std
        if v.ndim == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return np.clip((v - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        if not self.synchronized:
            self.synchronize()
        mean, std = self.mean, self.std
        if v.ndim == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return mean + v * std

    def synchronize(self):
        self.mean[...] = self.sum / self.count[0]
        self.std[...] = np.sqrt(
            np.maximum(
                np.square(self.eps),
                self.sumsq / self.count[0] - np.square(self.mean)
            )
        )
        self.synchronized = True


class IdentityNormalizer(object):
    def __init__(self, *args, **kwargs):
        pass

    def update(self, v):
        pass

    def normalize(self, v, clip_range=None):
        return v

    def denormalize(self, v):
        return v


class FixedNormalizer(object):
    def __init__(
            self,
            size,
            default_clip_range=np.inf,
            mean=0,
            std=1,
            eps=1e-8,
    ):
        assert std > 0
        std = std + eps
        self.size = size
        self.default_clip_range = default_clip_range
        self.mean = mean + np.zeros(self.size, np.float32)
        self.std = std + np.zeros(self.size, np.float32)
        self.eps = eps

    def set_mean(self, mean):
        self.mean = mean + np.zeros(self.size, np.float32)

    def set_std(self, std):
        std = std + self.eps
        self.std = std + np.zeros(self.size, np.float32)

    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        mean, std = self.mean, self.std
        if v.ndim == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return np.clip((v - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        mean, std = self.mean, self.std
        if v.ndim == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return mean + v * std

    def copy_stats(self, other):
        self.set_mean(other.mean)
        self.set_std(other.std)
