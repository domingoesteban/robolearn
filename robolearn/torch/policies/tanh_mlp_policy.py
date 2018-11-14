from robolearn.utils.serializable import Serializable
from robolearn.torch.policies.mlp_policy import MlpPolicy


class TanhMlpPolicy(MlpPolicy, Serializable):
    def __init__(self, *args, **kwargs):
        # self._serializable_initialized = False
        # Serializable.quick_init(self, locals())

        self.save_init_params(locals())
        super(TanhMlpPolicy, self).__init__(
            *args,
            output_activation='tanh',
            **kwargs)

