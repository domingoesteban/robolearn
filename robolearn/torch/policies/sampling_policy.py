import numpy as np
import torch
from torch.distributions import Normal

from robolearn.torch.nn import Mlp
from robolearn.policies.base import ExplorationPolicy
import robolearn.torch.pytorch_util as ptu


class SamplingPolicy(Mlp, ExplorationPolicy):
    """Sampling NN policy."""
    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_sizes,
                 squash=True,
                 **kwargs
                 ):
        self.save_init_params(locals())  # For serialization

        # MLP Init
        super(SamplingPolicy, self).__init__(
            hidden_sizes,
            input_size=obs_dim + action_dim,  # +action_dim id for stochasticity
            output_size=action_dim,
            **kwargs
        )

        self._action_dim = action_dim
        self._latent_dist = Normal(torch.Tensor([0]), torch.Tensor([1]))

        self._squash = squash

        # # TODO: WE ARE INITIALIZING LAST LAYER WEIGHTS WITH XAVIER
        # nn_pol.init.xavier_normal_(self.last_pfcs.weight.data)
        # # self.last_pfcs.bias.data.zero_()

    def get_action(self, obs_np, deterministic=False):
        # TODO: CHECK IF INDEX 0
        actions, info_dict = self.get_actions(obs_np[None],
                                              deterministic=deterministic)
        for key, val in info_dict.items():
            if isinstance(val, np.ndarray):
                info_dict[key] = val[0, :]
        return actions[0, :], info_dict

    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)

    def forward(
            self,
            obs,
            deterministic=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        """
        # TODO: HOW TO DETERMINISTIC???
        latent_shape = (*list(obs.shape)[:-1], self._action_dim)
        if deterministic:
            latent = torch.zeros(latent_shape)
        else:
            latent = self._latent_dist.sample(latent_shape).squeeze(-1)

        if ptu.gpu_enabled():
            latent = latent.cuda()

        h = torch.cat([obs, latent], dim=-1)
        # print('--- INPUT ---')
        # print(torch.cat([obs, latent], dim=-1)[:5, :])
        for i, fc in enumerate(self.fcs):
            # h = self.hidden_activation(fc(h))
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)

        action = self.last_fc(h)

        if self._squash:
            action = torch.tanh(action)

        info_dict = dict()

        return action, info_dict
