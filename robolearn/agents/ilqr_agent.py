import numpy as np
from robolearn.agents.agent import Agent
from robolearn.policies.lin_gauss_policy import LinearGaussianPolicy


class ILQRAgent(Agent):
    def __init__(self, act_dim, obs_dim, T, agent_name=""):
        super(ILQRAgent, self).__init__(act_dim=act_dim, obs_dim=obs_dim, state_dim=None)

        self.agent_name = agent_name

        self.T = T
        self.t = 0  # Internal counter
        self.acum_reward = 0

        # TODO: IMPLEMENT THIS WITH lin_gauss_init.py functions
        self.init_var = 0.001  # Initial exploration noise
        K = np.tile(np.zeros((self.act_dim, self.obs_dim)), [T, 1, 1])
        k = np.random.random((T, self.act_dim))
        pol_covar = self.init_var * np.tile(np.eye(self.act_dim), [T, 1, 1])
        chol_pol_covar = np.sqrt(self.init_var) * np.tile(np.eye(self.act_dim), [T, 1, 1])
        inv_pol_covar = (1.0 / self.init_var) * np.tile(np.eye(self.act_dim), [T, 1, 1])

        self.policy = LinearGaussianPolicy(K, k, pol_covar, chol_pol_covar, inv_pol_covar)

    def act(self, obs, reward, done):
        self.acum_reward += reward
        noise = np.zeros(self.act_dim)

        action = self.policy.eval(state=obs, t=self.t, noise=noise)

        print(self.t, '/', self.T-1, 'action:', action)
        if not self.t < self.T-1:
            self.t = -1
            print('I %s (iLQR agent) OBTAINED A RETURN:' % self.agent_name, self.acum_reward)
            self.acum_reward = 0

        self.t += 1

        return action

    def seed(self, seed):
        # np_random(seed)
        rng = np.random.RandomState()
        rng.seed(seed)

def np_random(seed=None):
    if seed is not None and not (isinstance(seed, integer_types) and 0 <= seed):
        raise error.Error('Seed must be a non-negative integer or omitted, not {}'.format(seed))

    seed = _seed(seed)

    rng = np.random.RandomState()
    rng.seed(_int_list_from_bigint(hash_seed(seed)))
    return rng, seed

def hash_seed(seed=None, max_bytes=8):
    """Any given evaluation is likely to have many PRNG's active at
    once. (Most commonly, because the environment is running in
    multiple processes.) There's literature indicating that having
    linear correlations between seeds of multiple PRNG's can correlate
    the outputs:
    http://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers/
    http://stackoverflow.com/questions/1554958/how-different-do-random-seeds-need-to-be
    http://dl.acm.org/citation.cfm?id=1276928
    Thus, for sanity we hash the seeds before using them. (This scheme
    is likely not crypto-strength, but it should be good enough to get
    rid of simple correlations.)
    Args:
        seed (Optional[int]): None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the hashed seed.
    """
    if seed is None:
        seed = _seed(max_bytes=max_bytes)
    hash = hashlib.sha512(str(seed).encode('utf8')).digest()
    return _bigint_from_bytes(hash[:max_bytes])

def _seed(a=None, max_bytes=8):
    """Create a strong random seed. Otherwise, Python 2 would seed using
    the system time, which might be non-robust especially in the
    presence of concurrency.
    Args:
        a (Optional[int, str]): None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the seed.
    """
    # Adapted from https://svn.python.org/projects/python/tags/r32/Lib/random.py
    if a is None:
        a = _bigint_from_bytes(os.urandom(max_bytes))
    elif isinstance(a, str):
        a = a.encode('utf8')
        a += hashlib.sha512(a).digest()
        a = _bigint_from_bytes(a[:max_bytes])
    elif isinstance(a, integer_types):
        a = a % 2 ** (8 * max_bytes)
    else:
        raise error.Error('Invalid type for seed: {} ({})'.format(type(a), a))

    return