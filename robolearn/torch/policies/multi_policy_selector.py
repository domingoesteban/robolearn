from robolearn.torch.core import PyTorchModule
from robolearn.policies.base import ExplorationPolicy


class MultiPolicySelector(PyTorchModule, ExplorationPolicy):
    def __init__(self, multipolicy, idx):
        self.save_init_params(locals())
        super(MultiPolicySelector, self).__init__()
        ExplorationPolicy.__init__(self, multipolicy.action_dim)

        self._multipolicy = multipolicy
        self.idx = [idx]

    def get_action(self, *args, **kwargs):
        kwargs['pol_idxs'] = self.idx
        actions, policy_infos = self._multipolicy.get_action(*args, **kwargs)

        action = actions[-1]

        for key, vals in policy_infos.items():
            policy_infos[key] = vals[-1]

        return action, policy_infos

    def get_actions(self, *args, **kwargs):
        kwargs['pol_idxs'] = self.idx
        actions, policy_infos = self._multipolicy.get_actions(*args, **kwargs)

        action = actions[-1]  # Only return actions

        for key, vals in policy_infos.items():
            policy_infos[key] = vals[-1]

        return action, policy_infos

    def forward(self, *nn_input, **kwargs):
        kwargs['pol_idxs'] = self.idx
        actions, policy_infos = self._multipolicy(*nn_input, **kwargs)
        action = actions[-1]
        for key, vals in policy_infos.items():
            policy_infos[key] = vals[-1]

        return action, policy_infos

