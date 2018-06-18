from robolearn.torch.nn import Mlp
from robolearn.core.serializable import Serializable
from robolearn.models import VFunction


class NNVFunction(Mlp, Serializable, VFunction):
    def __init__(self,
                 obs_dim,
                 hidden_sizes=(100, 100)):

        VFunction.__init__(self,
                           obs_dim=obs_dim)

        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        self.save_init_params(locals())
        Mlp.__init__(self,
                     hidden_sizes=hidden_sizes,
                     input_size=obs_dim,
                     output_size=1,
                     )

    def get_value(self, obs_np, deterministic=False):
        # TODO: CHECK IF INDEX 0
        values = self.get_values(obs_np[None], deterministic=deterministic)
        return values[0, :], {}

    def get_values(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)

