import numpy as np
import torch
from torch.autograd import Variable
from robolearn.torch.sql.kernel import adaptive_isotropic_gaussian_kernel

h_min = 1e-3
torch.manual_seed(7)

xs = Variable(torch.rand(3, 2, 5))
ys = Variable(torch.rand(3, 4, 5))


result = adaptive_isotropic_gaussian_kernel(xs, ys, h_min)

print(result)
