import torch
from robolearn.torch.utils.nn.networks.mlp import Mlp


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension -1 and then pass through MLP.
    """

    def forward(self, *nn_inputs, **kwargs):
        flat_inputs = torch.cat(nn_inputs, dim=-1)
        return super(FlattenMlp, self).forward(flat_inputs, **kwargs)
