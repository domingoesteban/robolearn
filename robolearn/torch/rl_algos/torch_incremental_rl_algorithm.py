from robolearn.core.incremental_rl_algorithm import IncrementalRLAlgorithm
from robolearn.torch.rl_algos.torch_algorithm import TorchAlgorithm
from robolearn.torch import pytorch_util as ptu


class TorchIncrementalRLAlgorithm(IncrementalRLAlgorithm, TorchAlgorithm):
    def __init__(self, *args, **kwargs):
        IncrementalRLAlgorithm.__init__(self, *args, **kwargs)

    def get_exploration_paths(self):
        """
        Get the current exploration paths.
        Returns:

        """
        paths = self._exploration_paths

        return [dict(observations=ptu.np_to_var(path['observations']),
                     actions=ptu.np_to_var(path['actions']),
                     rewards=ptu.np_to_var(path['rewards']),
                     next_observations=ptu.np_to_var(path["next_observations"]),
                     terminals=ptu.np_to_var(path["terminals"]),
                     agent_infos=path['agent_infos']
                     )
                for path in paths]
