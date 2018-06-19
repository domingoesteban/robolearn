"""
Based on Haarnoja's TF implementation

https://github.com/haarnoja/softqlearning
"""

import numpy as np
import torch


def adaptive_isotropic_gaussian_kernel(xs, ys, h_min=1e-3):
    """Gaussian kernel with dynamic bandwith.

    The bandwith is adjusted dynamically to match median_distance / log(Kx).
    See [2] for more information.

    Args:
        xs (`torch.Tensor`): A tensor of shape (N x Kx x D) containing N sets of
            Kx particles of dimension D. This is the first kernel argument.
        ys (`torch.Tensor`): A tensor of shape (N x Ky x D) containing N sets of
            Ky particles of dimension D. This is the second kernel argument.
        h_min (`float`): Minimum bandwith.

    Returns:
        `dict`: Returned dictionary has two fields:
            `output`: A `torch.Tensor` object of shape (N x Kx x Ky)
                representing the kernel matrix for inputs `xy` and `ys`.
            `gradient`: A `torch.Tensor`
    """
    Kx, D = xs.shape[-2:]
    Ky, D2 = ys.shape[-2:]
    assert D == D2

    leading_shape = list(xs.shape[:-2])[-1]

    # Compute the pairwise distances of left and right particles.
    diff = torch.unsqueeze(xs, -2) - torch.unsqueeze(ys, -3)
    # ... x Kx x Ky x D

    dist_sq = torch.sum(diff**2, dim=-1, keepdim=False)
    # ... x Kx x Ky

    # Get median.
    input_shape = (leading_shape, Kx * Ky)
    type(leading_shape)
    values, _ = torch.topk(
        dist_sq.view(*input_shape),
        k=(Kx * Ky // 2 + 1),  # This is exactly true only if Kx*Ky is odd.
        largest=True,
        sorted=True  # ... x floor(Ks*Kd/2)
    )

    medians_sq = values[..., -1]  # ... (shape) (last element is the median)

    h = medians_sq / np.log(Kx)  # ... (shape)
    h = torch.clamp(h, min=h_min)
    h.detach_()  # TODO: We can have a problem if inputs are not Variable
    h_expanded_twice = torch.unsqueeze(torch.unsqueeze(h, -1), -1)
    # ... x 1 x 1

    kappa = torch.exp(-dist_sq / h_expanded_twice)  # ... x Kx x Ky

    # Construct the gradient
    h_expanded_thrice = torch.unsqueeze(h_expanded_twice, -1)
    # ... x 1 x 1 x 1
    kappa_expanded = torch.unsqueeze(kappa, -1)
    kappa_grad = -2 * diff / h_expanded_thrice * kappa_expanded
    # ... x Kx x Ky x D

    return {'output': kappa, 'gradient': kappa_grad}