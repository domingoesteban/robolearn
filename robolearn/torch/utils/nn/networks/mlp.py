import torch.nn as nn
from robolearn.torch.core import PyTorchModule
from robolearn.torch.utils.nn import LayerNorm
import robolearn.torch.utils.pytorch_util as ptu


class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            input_size,
            output_size,
            hidden_activation='relu',
            output_activation='linear',
            hidden_w_init='xavier_normal',
            hidden_b_init_val=0.1,
            output_w_init='xavier_normal',
            output_b_init_val=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super(Mlp, self).__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size

        self.hidden_activation = ptu.get_activation(hidden_activation)
        self.output_activation = ptu.get_activation(output_activation)

        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            ptu.layer_init(
                layer=fc,
                option=hidden_w_init,
                activation=hidden_activation,
                b=hidden_b_init_val
            )
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("shared_layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        ptu.layer_init(
            layer=self.last_fc,
            option=output_w_init,
            activation=output_activation,
            b=output_b_init_val
        )

    def forward(self, nn_input, return_preactivations=False):
        h = nn_input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        if return_preactivations:
            return output, preactivation
        else:
            return output
