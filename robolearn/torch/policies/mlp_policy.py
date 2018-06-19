import numpy as np
from robolearn.torch.nn import Mlp
from robolearn.policies import Policy
from robolearn.core.serializable import Serializable
import robolearn.torch.pytorch_util as ptu


class MlpPolicy(Mlp, Serializable, Policy):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_sizes=(100, 100),
                 hidden_w_init=ptu.xavier_init,
                 hidden_b_init_val=0,
                 output_w_init=ptu.xavier_init,
                 output_b_init_val=0,
                 **kwargs
                 ):

        Policy.__init__(self,
                        action_dim=action_dim)

        # self._serializable_initialized = False
        # Serializable.quick_init(self, locals())
        self.save_init_params(locals())
        Mlp.__init__(self,
                     hidden_sizes=hidden_sizes,
                     input_size=obs_dim,
                     output_size=action_dim,
                     hidden_w_init=hidden_w_init,
                     hidden_b_init_val=hidden_b_init_val,
                     output_w_init=output_w_init,
                     output_b_init_val=output_b_init_val,
                     **kwargs
                     )

    def get_action(self, obs_np, **kwargs):
        values, info_dict = \
            self.get_actions(obs_np[None], **kwargs)

        for key, val in info_dict.items():
            if isinstance(val, np.ndarray):
                info_dict[key] = val[0, :]

        return values[0, :], info_dict

    def get_actions(self, obs_np, **kwargs):
        return self.eval_np(obs_np, **kwargs)

    def forward(
            self,
            obs,
            return_preactivations=False,
    ):
        nn_ouput = Mlp.forward(self, obs,
                               return_preactivations=return_preactivations)

        if return_preactivations:
            action = nn_ouput[0]
            pre_activations = nn_ouput[1]
            info_dict = dict(
                pre_activations=pre_activations,
            )
        else:
            action = nn_ouput
            info_dict = dict()

        return action, info_dict

