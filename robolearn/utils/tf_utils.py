"""  Original code: https://github.com/wojzaremba/trpo """

import numpy as np
import tensorflow as tf
import scipy.signal


def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def rollout(env, agent, max_pathlength, n_timesteps):
    paths = []
    timesteps_sofar = 0
    while timesteps_sofar < n_timesteps:
        obs, actions, rewards, action_dists = [], [], [], []
        ob = env.reset()
        agent.prev_action *= 0.0
        agent.prev_obs *= 0.0
        for _ in range(max_pathlength):
            action, action_dist, ob = agent.act(ob)
            obs.append(ob)
            actions.append(action)
            action_dists.append(action_dist)
            res = env.step(action)
            ob = res[0]
            rewards.append(res[1])
            if res[2]: # DONE
                path = {'obs': np.concatenate(np.expand_dims(obs, 0))}


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), "Shape function assumes that shape is fully known"
    return out


def numel(x):
    """ Returns number of elements in tensor """
    return np.prod(var_shape(x))


class GetFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat(0, [tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        return self.op.eval(session=self.session)


def slice_2d(x, inds0, inds1):
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(x), tf.int64)
    ncols = shape[1]
    x_flat = tf.reshape(x, [-1])
    return tf.gather(x_flat, inds0 * ncols + inds1)


def linesearch(f, x, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.5*np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return xnew
    return x


class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def explained_variance(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1  # They should be 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary
