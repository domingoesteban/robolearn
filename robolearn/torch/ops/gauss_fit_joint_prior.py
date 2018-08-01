"""
THIS FILE IS ADAPTED FROM FINN'S GPS
"""

import torch
import numpy as np


def gauss_fit_joint_prior(pts, mu0, Phi, m, n0, dwts, d1, d2, sig_reg):
    """
    Perform Gaussian fit to data with a prior.
    :param pts: (N x dA x dA)
    :param mu0:
    :param Phi:
    :param m:
    :param n0:
    :param dwts:
    :param d1: dimension of first set? E.g. dX
    :param d2: simension of second set? E.g. dU
    :param sig_reg:
    :return:
    """
    # Build weights matrix.
    D = torch.diag(dwts)
    # Compute empirical mean and covariance.
    mun = torch.sum((pts.t() * dwts).t(), dim=0)
    diff = pts - mun
    empsig = diff.t().matmul(D).matmul(diff)
    empsig = 0.5 * (empsig + empsig.t())
    # MAP estimate of joint distribution.
    N = dwts.shape[0]
    mu = mun
    sigma = (N * empsig + Phi + (N * m) / (N + m) * torch.ger(mun - mu0, mun - mu0)) / (N + n0)
    sigma = 0.5 * (sigma + sigma.T)
    # Add sigma regularization.
    sigma += sig_reg
    # Conditioning to get dynamics.
    fd = torch.gesv(sigma[:d1, :d1],
                    sigma[:d1, d1:d1 + d2]).t()
    fc = mu[d1:d1 + d2] - fd.matmul(mu[:d1])
    dynsig = sigma[d1:d1 + d2, d1:d1 + d2] - \
             fd.matmul(sigma[:d1, :d1]).matmul(fd.t())
    dynsig = 0.5 * (dynsig + dynsig.t())
    return fd, fc, dynsig
