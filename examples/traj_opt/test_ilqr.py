import numpy as np
from robolearn.traj_opt.iLQR import iLQR
from robolearn.v010.policies.lin_gauss_policy import LinearGaussianPolicy


des_x = 0


def cost_fcn(x, des_x):
    return 1/2. * (np.)