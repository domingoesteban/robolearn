import torch.optim as optim

from robolearn.torch.rl_algos.torch_incremental_rl_algorithm \
    import TorchIncrementalRLAlgorithm
from robolearn.policies import MakeDeterministic


class PPO(TorchIncrementalRLAlgorithm):
    """
    Proximal Policy Optimization
    """

    def __init__(
            self,
            env,
            qf,
            policy,
            exploration_policy,

            policy_learning_rate=1e-4,
            qf_learning_rate=1e-3,
            qf_weight_decay=0,
            target_hard_update_period=1000,
            tau=1e-2,
            use_soft_update=False,
            qf_criterion=None,
            residual_gradient_weight=0,
            epoch_discount_schedule=None,
            eval_deterministic=True,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,

            plotter=None,
            render_eval_paths=False,

            **kwargs
    ):
        if eval_deterministic:
            eval_policy = MakeDeterministic(policy)
        else:
            eval_policy = policy
        super(PPO, self).__init__(
            env=env,
            exploration_policy=policy,
            eval_policy=eval_policy,
            **kwargs
        )
        self.policy = policy
        self.qf = qf

    def pretrain(self):
        pass

    def get_epoch_snapshot(self, epoch):
        pass

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

