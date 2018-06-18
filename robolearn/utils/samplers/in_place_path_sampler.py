from robolearn.utils.samplers.rollout import rollout


class InPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_samples, max_path_length,
                 deterministic=None):
        """

        Args:
            env:
            policy:
            max_samples:
            max_path_length:
            deterministic:
        """
        self.env = env
        self.policy = policy
        self.max_path_length = max_path_length
        self.max_samples = max_samples
        assert max_samples >= max_path_length, \
            "Need max_samples >= max_path_length"
        self.deterministic = deterministic

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
        while n_steps_total + self.max_path_length <= self.max_samples:
            # Execute a single rollout
            path = rollout(
                self.env, self.policy, max_path_length=self.max_path_length,
                deterministic=self.deterministic
            )
            paths.append(path)
            n_steps_total += len(path['observations'])
        return paths
