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
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def maximum_2d(t1, t2):
    # noinspection PyArgumentList
    return torch.max(
        torch.cat((t1.unsqueeze(2), t2.unsqueeze(2)), dim=2),
        dim=2,
    )[0].squeeze(2)


def identity(x):
    return x


def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    # TODO(vitchyr): see if you can use expand instead of repeat
    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2


def selu(
        x,
        alpha=1.6732632423543772848170429916717,
        scale=1.0507009873554804934193349852946,
):
    """
    Based on https://github.com/dannysdeng/selu/blob/master/selu.py
    """
    return scale * (
        F.relu(x) + alpha * (F.elu(-1 * F.relu(-1 * x)))
    )


def alpha_dropout(
        x,
        p=0.05,
        alpha=-1.7580993408473766,
        fixedPointMean=0,
        fixedPointVar=1,
        training=False,
):
    keep_prob = 1 - p
    if keep_prob == 1 or not training:
        return x
    a = np.sqrt(fixedPointVar / (keep_prob * (
        (1 - keep_prob) * pow(alpha - fixedPointMean, 2) + fixedPointVar)))
    b = fixedPointMean - a * (
        keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
    keep_prob = 1 - p

    random_tensor = keep_prob + torch.rand(x.size())
    binary_tensor = Variable(torch.floor(random_tensor))
    x = x.mul(binary_tensor)
    ret = x + alpha * (1 - binary_tensor)
    ret.mul_(a).add_(b)
    return ret


def alpha_selu(x, training=False):
    return alpha_dropout(selu(x), training=training)


def double_moments(x, y):
    """
    Returns the first two moments between x and y.

    Specifically, for each vector x_i and y_i in x and y, compute their
    outer-product. Flatten this resulting matrix and return it.

    The first moments (i.e. x_i and y_i) are included by appending a `1` to x_i
    and y_i before taking the outer product.
    :param x: Shape [batch_size, feature_x_dim]
    :param y: Shape [batch_size, feature_y_dim]
    :return: Shape [batch_size, (feature_x_dim + 1) * (feature_y_dim + 1)
    """
    batch_size, x_dim = x.size()
    _, y_dim = x.size()
    x = torch.cat((x, Variable(torch.ones(batch_size, 1))), dim=1)
    y = torch.cat((y, Variable(torch.ones(batch_size, 1))), dim=1)
    x_dim += 1
    y_dim += 1
    x = x.unsqueeze(2)
    y = y.unsqueeze(1)

    outer_prod = (
        x.expand(batch_size, x_dim, y_dim) * y.expand(batch_size, x_dim, y_dim)
    )
    return outer_prod.view(batch_size, -1)


def batch_diag(diag_values, diag_mask=None):
    batch_size, dim = diag_values.size()
    if diag_mask is None:
        diag_mask = torch.diag(torch.ones(dim))
    batch_diag_mask = diag_mask.unsqueeze(0).expand(batch_size, dim, dim)
    batch_diag_values = diag_values.unsqueeze(1).expand(batch_size, dim, dim)
    return batch_diag_values * batch_diag_mask


def batch_square_vector(vector, M):
    """
    Compute x^T M x
    """
    vector = vector.unsqueeze(2)
    return torch.bmm(torch.bmm(vector.transpose(2, 1), M), vector).squeeze(2)


def fill(tensor, value):
    with torch.no_grad():
        return tensor.fill_(value)



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
    # if isinstance(tensor, TorchVariable):
    #     return fanin_init(tensor.data)
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
    # if isinstance(tensor, TorchVariable):
    #     return fanin_init(tensor.data)
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


def almost_identity_weights_like(tensor):
    """
    Set W = I + lambda * Gaussian no
    :param tensor:
    :return:
    """
    shape = tensor.size()
    init_value = np.eye(*shape)
    init_value += 0.01 * np.random.rand(*shape)
    return FloatTensor(init_value)


def clip1(x):
    return torch.clamp(x, -1, 1)


def xavier_init(tensor, gain=1, uniform=True):
    if uniform:
        return xavier_uniform_init(tensor, gain)
    else:
        return xavier_norm_init(tensor, gain)


def xavier_norm_init(tensor, gain=1):
    return torch.nn.init.xavier_normal_(tensor, gain=gain)


def xavier_uniform_init(tensor, gain=1):
    return torch.nn.init.xavier_uniform_(tensor, gain=gain)


"""
GPU wrappers
"""
_use_gpu = False


def set_gpu_mode(mode):
    global _use_gpu
    _use_gpu = mode


def gpu_enabled():
    return _use_gpu


# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    if _use_gpu:
        return torch.cuda.FloatTensor(*args, **kwargs)
    else:
        # noinspection PyArgumentList
        return torch.FloatTensor(*args, **kwargs)


# noinspection PyPep8Naming
def BinaryTensor(*args, **kwargs):
    if _use_gpu:
        return torch.cuda.ByteTensor(*args, **kwargs)
    else:
        # noinspection PyArgumentList
        return torch.ByteTensor(*args, **kwargs)


# noinspection PyPep8Naming
def LongTensor(*args, **kwargs):
    if _use_gpu:
        return torch.cuda.LongTensor(*args, **kwargs)
    else:
        # noinspection PyArgumentList
        return torch.LongTensor(*args, **kwargs)


def Variable(tensor, **kwargs):
    if _use_gpu and not tensor.is_cuda:
        return TorchVariable(tensor.cuda(), **kwargs)
    else:
        return TorchVariable(tensor, **kwargs)


def from_numpy(*args, **kwargs):
    if _use_gpu:
        return torch.from_numpy(*args, **kwargs).float().cuda()
    else:
        return torch.from_numpy(*args, **kwargs).float()


def get_numpy(tensor):
    # if isinstance(tensor, TorchVariable):
    #     return get_numpy(tensor.data)
    if _use_gpu:
        return tensor.data.cpu().numpy()
    return tensor.data.numpy()


def np_to_var(np_array, **kwargs):
    if np_array.dtype == np.bool:
        np_array = np_array.astype(int)
    return Variable(from_numpy(np_array), **kwargs)


def zeros(*sizes, out=None):
    tensor = torch.zeros(*sizes, out=out)
    if _use_gpu:
        tensor = tensor.cuda()
    return tensor


def ones(*sizes, out=None):
    tensor = torch.ones(*sizes, out=out)
    if _use_gpu:
        tensor = tensor.cuda()
    return tensor


def zeros_like(*args, **kwargs):
    tensor = torch.zeros_like(*args, **kwargs)
    if _use_gpu:
        tensor = tensor.cuda()
    return tensor


def ones_like(*args, **kwargs):
    tensor = torch.ones_like(*args, **kwargs)
    if _use_gpu:
        tensor = tensor.cuda()
    return tensor

def rand(*args, **kwargs):
    tensor = torch.rand(*args, **kwargs)
    if _use_gpu:
        tensor = tensor.cuda()
    return tensor

def randn(*args, **kwargs):
    tensor = torch.randn(*args, **kwargs)
    if _use_gpu:
        tensor = tensor.cuda()
    return tensor


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


"""
Useful functions
"""


def activation(name):
    name = name.lower()
    if hasattr(torch, name):
        return getattr(torch, name)
    elif name == 'identity' or 'identity':
        return identity
    else:
        raise AttributeError("Pytorch does not have activation '%s'",
                             name)

