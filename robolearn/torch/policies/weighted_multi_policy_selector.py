from robolearn.torch.core import PyTorchModule
from robolearn.policies import ExplorationPolicy


class WeightedMultiPolicySelector(PyTorchModule, ExplorationPolicy):
    def __init__(self, multipolicy, idx):
        self.save_init_params(locals())
        super(WeightedMultiPolicySelector, self).__init__()
        ExplorationPolicy.__init__(self, multipolicy.action_dim)

        self._multipolicy = multipolicy
        self.idx = idx

    def get_action(self, *args, **kwargs):
        kwargs['pol_idx'] = self.idx
        action, policy_info = self._multipolicy.get_action(*args, **kwargs)

        return action, policy_info

    def get_actions(self, *args, **kwargs):
        kwargs['pol_idx'] = self.idx
        action, policy_info = self._multipolicy.get_actions(*args, **kwargs)

        return action, policy_info

    def forward(self, *nn_input, **kwargs):
        kwargs['pol_idx'] = self.idx
        action, policy_info = self._multipolicy(*nn_input, **kwargs)

        return action, policy_info

