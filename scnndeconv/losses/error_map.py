import math
import numpy as np
import torch
import torch.nn as nn

from .utils import disk_patch, ring_patch


class SMSELoss(nn.Module):
    def __init__(self):
        super(SMSELoss, self).__init__()

    def forward(self, inputs, targets):
        tmp = (inputs - targets)**2
        return torch.mean(tmp)


class SAContrarioMap(nn.Module):
    """Define a loss with the a-contrario anomalies map

    loss = Area_anomalies + beta*MSE

    Parameters
    ----------
    radius: int
        Radius of the anomalies detection disk (in pixels)
    alpha: int
        Width of the anomalies detection ring (in pixel)
    beta: float
        Weighting parameter for the MSE term

    """
    def __init__(self, radius=1, alpha=7, beta=1):
        super(SAContrarioMap, self).__init__()

        self.beta = beta
        self.disk_mat = disk_patch(radius)
        self.ring_mat = ring_patch(radius, alpha)
        self.coefficient = math.sqrt(math.pi) * math.sqrt(1 - 1/(alpha*alpha)) * radius
        self.sqrt2 = math.sqrt(2)

    def forward(self, inputs, targets):

        inner_weights = torch.Tensor(self.disk_mat).unsqueeze(0).unsqueeze(0)
        inner_weights.require_grad = True
        outer_weights = torch.Tensor(self.ring_mat).unsqueeze(0).unsqueeze(0)
        outer_weights.require_grad = True

        inner_conv = nn.Conv2d(1, 1, kernel_size=self.disk_mat.shape[0], stride=1, padding=1, bias=False)
        outer_conv = nn.Conv2d(1, 1, kernel_size=self.ring_mat.shape[0], stride=1, padding=1, bias=False)
        with torch.no_grad():
            inner_conv.weight = nn.Parameter(inner_weights)
            outer_conv.weight = nn.Parameter(outer_weights)

        error = inputs - targets
        sigma = torch.sqrt(torch.var(error))
        stat = self.coefficient * (inner_conv(error) - outer_conv(error)) / sigma
        map = 0.5*torch.erfc(stat/sqrt2)

        threshold = nn.Threshold(0.9, 0)
        th_map = threshold(map)
        area = torch.count_nonzero(th_map)

        tmp = (inputs - targets) ** 2
        mse = torch.mean(tmp)

        return area + self.beta*mse
