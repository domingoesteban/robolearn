from robolearn.torch.nn import FlattenMlp
from robolearn.core.serializable import Serializable
from robolearn.models import QFunction


class NNQFunction(FlattenMlp, Serializable, QFunction):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_sizes=(100, 100)):
        QFunction.__init__(self,
                           obs_dim=obs_dim,
                           action_dim=action_dim)

        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        self.save_init_params(locals())
        FlattenMlp.__init__(self,
                            hidden_sizes=hidden_sizes,
                            input_size=obs_dim+action_dim,
                            output_size=1,
                            )

    def get_value(self, obs_np, act_np, **kwargs):
        # TODO: CHECK IF INDEX 0
        values, val_info = self.get_values(obs_np[None], act_np[None], **kwargs)
        return values[0, :], val_info

    def get_values(self, obs_np, act_np, **kwargs):
        return self.eval_np(obs_np, act_np, **kwargs), dict()

