import torch.nn as nn


class HuberLoss(nn.Module):
    def __init__(self, delta=1):
        super(HuberLoss, self).__init__()
        self.huber_loss_delta1 = nn.SmoothL1Loss()
        self.delta = delta

    def forward(self, x, x_hat):
        loss = self.huber_loss_delta1(x / self.delta, x_hat / self.delta)
        return loss * self.delta * self.delta
