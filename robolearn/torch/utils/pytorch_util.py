import torch
import numpy as np

from torch.autograd import Variable as TorchVariable
from torch.nn import functional as F
import torch.nn as nn


def seed(seed):
    torch.cuda.manual_seed(seed)
    r_generator = torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    return r_generator


def soft_update_from_to(source, target, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + source_param.data * tau
        )


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def identity(x):
    return x


def fill(tensor, value):
    with torch.no_grad():
        return tensor.fill_(value)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


"""
Tensor initialization
"""


def zeros_init(tensor):
    with torch.no_grad():
        return tensor.zero_()


def fanin_init(tensor):
    """
    Fan-in initialization.
    Args:
        tensor:

    Returns:

    """
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def fanin_init_weights_like(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    new_tensor = FloatTensor(tensor.size())
    new_tensor.data.uniform_(-bound, bound)
    return new_tensor


def xavier_initOLD(tensor, gain=1, uniform=True):
    if uniform:
        return xavier_uniform_init(tensor, gain)
    else:
        return xavier_norm_init(tensor, gain)


def xavier_norm_init(tensor, gain=1):
    return torch.nn.init.xavier_normal_(tensor, gain=gain)


def xavier_uniform_init(tensor, gain=1):
    return torch.nn.init.xavier_uniform_(tensor, gain=gain)


def layer_init(layer, option='xavier_normal', activation='relu', b=0.01):
    if option.lower().startswith('xavier'):
        init_weight_xavier(layer=layer, option=option, activation=activation)
    elif option.lower().startswith('uniform'):
        init_weight_uniform(layer=layer, activation=activation)
    elif option.lower().startswith('normal'):
        init_weight_normal(layer=layer, activation=activation)
    else:
        raise ValueError("Wrong init option")

    if hasattr(layer, 'bias'):
        fill(layer.bias, b)


def init_weight_uniform(layer, activation='1e-3'):
    if isinstance(activation, float):
        a = -activation
        b = activation
    else:
        a = -1.e-3
        b = 1.e-3
    nn.init.uniform_(layer.weight, a=a, b=b)


def init_weight_normal(layer, activation='1e2'):
    if isinstance(activation, float):
        std = activation
    else:
        std = 1.e-3
    nn.init.normal_(layer.weight, mean=0., std=std)


def init_weight_xavier(layer, option='xavier_normal', activation='relu'):
    if option == 'xavier_normal':
        xavier_fcn = nn.init.xavier_normal_
    elif option == 'xavier_uniform':
        xavier_fcn = nn.init.xavier_uniform_
    else:
        raise ValueError("Wrong init option")

    if activation.lower() in ['relu']:
        xavier_fcn(layer.weight,
                   gain=nn.init.calculate_gain('relu')
                   )
    elif activation in ['leaky_relu']:
        xavier_fcn(layer.weight,
                   gain=nn.init.calculate_gain('leaky_relu')
                   )
    elif activation.lower() in ['tanh']:
        xavier_fcn(layer.weight,
                   gain=nn.init.calculate_gain('tanh')
                   )
    elif activation.lower() in ['sigmoid']:
        xavier_fcn(layer.weight,
                   gain=nn.init.calculate_gain('sigmoid')
                   )
    elif activation.lower() in ['linear']:
        xavier_fcn(layer.weight,
                   gain=nn.init.calculate_gain('linear')
                   )
    elif activation.lower() in ['elu']:
        xavier_fcn(layer.weight,
                   gain=nn.init.calculate_gain('relu')
                   )
    elif activation.lower() in ['selu']:
        xavier_fcn(layer.weight,
                   gain=nn.init.calculate_gain('relu')
                   )
    elif activation.lower() in ['0.1']:
        xavier_fcn(layer.weight,
                   gain=0.1,
                   )
    elif activation.lower() in ['0.01']:
        xavier_fcn(layer.weight,
                   gain=0.01,
                   )
    elif activation.lower() in ['0.001']:
        xavier_fcn(layer.weight,
                   gain=0.001,
                   )
    elif activation.lower() in ['0.003']:
        xavier_fcn(layer.weight,
                   gain=0.001,
                   )
    else:
        raise AttributeError('Wrong option')


def get_activation(name):
    if name.lower() == 'relu':
        activation = torch.nn.functional.relu
    elif name.lower() == 'elu':
        activation = torch.nn.functional.elu
    elif name.lower() == 'leaky_relu':
        activation = torch.nn.functional.leaky_relu
    elif name.lower() == 'selu':
        activation = torch.nn.functional.selu
    elif name.lower() == 'sigmoid':
        activation = torch.nn.functional.sigmoid
    elif name.lower() == 'tanh':
        activation = torch.tanh
    elif name.lower() in ['linear', 'identity']:
        activation = identity
    else:
        raise AttributeError("Pytorch does not have activation '%s'",
                             name)
    return activation


"""
GPU wrappers
"""
_use_gpu = False
device = None


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")


def gpu_enabled():
    return _use_gpu


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    return torch.FloatTensor(*args, **kwargs).to(device)


# noinspection PyPep8Naming
def BinaryTensor(*args, **kwargs):
    return torch.ByteTensor(*args, **kwargs).to(device)


# noinspection PyPep8Naming
def LongTensor(*args, **kwargs):
    return torch.LongTensor(*args, **kwargs).to(device)


# noinspection PyPep8Naming
def IntTensor(*args, **kwargs):
    return torch.IntTensor(*args, **kwargs).to(device)


def Variable(tensor, **kwargs):
    if _use_gpu and not tensor.is_cuda:
        return TorchVariable(tensor.to(device), **kwargs)
    else:
        return TorchVariable(tensor, **kwargs)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.to('cpu').detach().numpy()
    else:
        return np.array(tensor)


def np_to_var(np_array, **kwargs):
    if np_array.dtype == np.bool:
        np_array = np_array.astype(int)
    return Variable(from_numpy(np_array), **kwargs)


def zeros(*sizes, **kwargs):
    return torch.zeros(*sizes, **kwargs).to(device)


def ones(*sizes, **kwargs):
    return torch.ones(*sizes, **kwargs).to(device)


def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs).to(device)


def ones_like(*args, **kwargs):
    return torch.ones_like(*args, **kwargs).to(device)


def eye(*sizes, **kwargs):
    return torch.eye(*sizes, **kwargs).to(device)


def rand(*args, **kwargs):
    return torch.rand(*args, **kwargs).to(device)


def randn(*args, **kwargs):
    return torch.randn(*args, **kwargs).to(device)


def arange(*args, **kwargs):
    return torch.arange(*args, **kwargs).to(device)


"""
Module functions
"""


def register_parameter(module, parameters_dict, name, param):
    r"""Adds a parameter to the module.

    The parameter can be accessed as an attribute using given name.

    Args:
        name (string): name of the parameter. The parameter can be accessed
            from this module using the given name
        parameter (Parameter): parameter to be added to the module.
    """
    if hasattr(module, name) and name not in parameters_dict:
        raise KeyError("attribute '{}' already exists".format(name))
    elif '.' in name:
        raise KeyError("parameter name can't contain \".\"")
    elif name == '':
        raise KeyError("parameter name can't be empty string \"\"")

    if param is None:
        parameters_dict[name] = None
    elif not isinstance(param, nn.Parameter):
        raise TypeError("cannot assign '{}' object to parameter '{}' "
                        "(torch.nn.Parameter or None required)"
                        .format(torch.typename(param), name))
    elif param.grad_fn:
        raise ValueError(
            "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
            "parameters must be created explicitly. To express '{0}' "
            "as a function of another Tensor, compute the value in "
            "the forward() method.".format(name))
    else:
        parameters_dict[name] = param


def named_children(modules_dict):
    r"""Returns an iterator over immediate children modules, yielding both
    the name of the module as well as the module itself.

    Yields:
        (string, Module): Tuple containing a name and child module

    Example::

        >>> for name, module in model.named_children():
        >>>     if name in ['conv4', 'conv5']:
        >>>         print(module)

    """
    memo = set()
    for name, module in modules_dict.items():
        if module is not None and module not in memo:
            memo.add(module)
            yield name, module


def named_parameters(modules_dict, parameters_dict, memo=None, prefix=''):
    r"""Returns an iterator over module parameters, yielding both the
    name of the parameter as well as the parameter itself

    Yields:
        (string, Parameter): Tuple containing the name and parameter

    Example::

        >>> for name, param in self.named_parameters(my_modules_dict, my_parameters_dict):
        >>>    if name in ['bias']:
        >>>        print(param.size())

    """
    if memo is None:
        memo = set()
    for name, p in parameters_dict.items():
        if p is not None and p not in memo:
            memo.add(p)
            yield prefix + ('.' if prefix else '') + name, p
    for mname, module in named_children(modules_dict):
        submodule_prefix = prefix + ('.' if prefix else '') + mname
        for name, p in module.named_parameters(memo, submodule_prefix):
            yield name, p


def add_module(modules_dict, name, module):
    r"""Adds a child module to the current module.

    The module can be accessed as an attribute using the given name.

    Args:
        name (string): name of the child module. The child module can be
            accessed from this module using the given name
        parameter (Module): child module to be added to the module.
    """
    if not isinstance(module, nn.Module) and module is not None:
        raise TypeError("{} is not a Module subclass".format(
            torch.typename(module)))
    elif '.' in name:
        raise KeyError("module name can't contain \".\"")
    elif name == '':
        raise KeyError("module name can't be empty string \"\"")
    modules_dict[name] = module


def np_to_pytorch_batch(np_batch):
    return {
        k: elem_or_tuple_to_variable(x)
        for k, x in filter_batch(np_batch)
        if isinstance(x, np.ndarray) and x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }


def elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return Variable(from_numpy(elem_or_tuple).float(), requires_grad=False)


def filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


