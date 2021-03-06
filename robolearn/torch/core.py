import abc
from future.utils import with_metaclass
import numpy as np
from collections import OrderedDict

from torch import nn as nn

from robolearn.torch.utils import pytorch_util as ptu
from robolearn.utils.serializable import Serializable


class PyTorchModule(with_metaclass(abc.ABCMeta, nn.Module, Serializable)):

    def get_param_values(self):
        return self.state_dict()

    def set_param_values(self, param_values):
        self.load_state_dict(param_values)

    def get_param_values_np(self):
        state_dict = self.state_dict()
        np_dict = OrderedDict()
        for key, tensor in state_dict.items():
            np_dict[key] = ptu.get_numpy(tensor)
        return np_dict

    def set_param_values_np(self, param_values):
        torch_dict = OrderedDict()
        for key, tensor in param_values.items():
            torch_dict[key] = ptu.from_numpy(tensor)
        self.load_state_dict(torch_dict)

    def copy(self):
        copy = Serializable.clone(self)
        ptu.copy_model_params_from_to(self, copy)
        return copy

    def clamp_all_params(self, min=None, max=None):
        for param in self.parameters():
            if min is not None:
                param.data.clamp_(min=min)
            if max is not None:
                param.data.clamp_(max=max)

    def save_init_params(self, locals):
        """
        Should call this FIRST THING in the __init__ method if you ever want
        to serialize or clone this network.

        Usage:
        ```
        def __init__(self, ...):
            self.init_serialization(locals())
            ...
        ```
        :param locals:
        :return:
        """
        Serializable.quick_init(self, locals)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["params"] = self.get_param_values()
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.set_param_values(d["params"])

    def regularizable_parameters(self):
        """
        Return generator of regularizable parameters. Right now, all non-flat
        vectors are assumed to be regularizabled, presumably because only
        biases are flat.

        :return:
        """
        for param in self.parameters():
            if len(param.size()) > 1:
                yield param

    def eval_np(self, *args, **kwargs):
        """
        Eval this module with a numpy interface

        Same as a call to __call__ except all Variable input/outputs are
        replaced with numpy equivalents.

        Assumes the output is either a single object or a tuple of objects.
        """
        torch_args = tuple(ptu.torch_ify(x) for x in args)
        torch_kwargs = {k: ptu.torch_ify(v) for k, v in kwargs.items()}
        outputs = self.__call__(*torch_args, **torch_kwargs)
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            return tuple(ptu.np_ify(x) for x in outputs)
        else:
            return ptu.np_ify(outputs)

