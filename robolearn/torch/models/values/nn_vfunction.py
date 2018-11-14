import numpy as np
from robolearn.torch.utils.nn import Mlp
from robolearn.utils.serializable import Serializable
from robolearn.models import VFunction


class NNVFunction(Mlp, Serializable, VFunction):
    def __init__(self,
                 obs_dim,
                 hidden_sizes=(100, 100),
                 **kwargs):
        VFunction.__init__(self,
                           obs_dim=obs_dim)

        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        self.save_init_params(locals())
        Mlp.__init__(self,
                     hidden_sizes=hidden_sizes,
                     input_size=obs_dim,
                     output_size=1,
                     **kwargs
                     )

    def get_value(self, obs_np, **kwargs):
        values, info_dict = \
            self.get_values(obs_np[None], **kwargs)

        for key, val in info_dict.items():
            if isinstance(val, np.ndarray):
                info_dict[key] = val[0, :]

        return values[0, :], info_dict

    def get_values(self, obs_np, **kwargs):
        return self.eval_np(obs_np, **kwargs)

    def forward(self, obs, return_preactivations=False):
        nn_ouput = Mlp.forward(self, obs,
                               return_preactivations=return_preactivations)

        if return_preactivations:
            value = nn_ouput[0]
            pre_activations = nn_ouput[1]
            info_dict = dict(
                pre_activations=pre_activations,
            )
        else:
            value = nn_ouput
            info_dict = dict()

        return value, info_dict
