from robolearn.utils.samplers.rollout import rollout


class InPlacePathSampler(object):
    """
    A sampler that does not serialize for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, total_samples, max_path_length,
                 deterministic=None, obs_normalizer=None):
        """

        Args:
            env:
            policy:
            total_samples:
            max_path_length: Maximum interaction samples per path.
            deterministic:
        """
        self.env = env
        self.policy = policy
        self._max_path_length = max_path_length
        self._total_samples = total_samples
        if not total_samples >= max_path_length:
            raise ValueError("Need total_samples >= max_path_length (%d >=%d)"
                             % (total_samples, max_path_length))
        self.deterministic = deterministic
        self._obs_normalizer = obs_normalizer

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self):
        """

        Returns:
            List of paths (list): A list of all the paths obtained until
                max_samples is reached.
        """
        paths = []
        n_steps_total = 0
        while n_steps_total < self._total_samples:
            # Execute a single rollout
            max_length = min(self._total_samples - n_steps_total,
                             self._max_path_length)
            path = rollout(
                self.env, self.policy, max_path_length=max_length,
                deterministic=self.deterministic,
                obs_normalizer=self._obs_normalizer,
            )
            paths.append(path)
            n_steps_total += len(path['observations'])
        return paths
