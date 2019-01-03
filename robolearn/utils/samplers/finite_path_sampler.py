from robolearn.utils.samplers.rollout import rollout


class FinitePathSampler(object):
    """

    """
    def __init__(self, env, policy, total_paths, max_path_length=1e20,
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
        self._total_paths =total_paths
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
        for nn in range(self._total_paths):
            path = rollout(
                self.env, self.policy, max_path_length=self._max_path_length,
                deterministic=self.deterministic,
                obs_normalizer=self._obs_normalizer,
            )
            paths.append(path)
        return paths
