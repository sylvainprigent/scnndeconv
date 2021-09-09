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


class SAContrarioLoss(nn.Module):
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
        super(SAContrarioLoss, self).__init__()

        self.beta = beta
        self.disk_mat = disk_patch(radius)
        self.ring_mat = ring_patch(radius, alpha)
        self.coefficient = math.sqrt(math.pi) * \
                           math.sqrt(1 - 1/(alpha*alpha)) * radius
        self.sqrt2 = math.sqrt(2)

    def forward(self, inputs, targets):

        inner_weights = torch.Tensor(self.disk_mat).unsqueeze(0).unsqueeze(0)
        inner_weights.require_grad = True
        outer_weights = torch.Tensor(self.ring_mat).unsqueeze(0).unsqueeze(0)
        outer_weights.require_grad = True

        inner_conv = nn.Conv2d(1, 1, kernel_size=self.disk_mat.shape[0],
                               stride=1,
                               padding=int((self.disk_mat.shape[0]-1)/2),
                               bias=False)
        outer_conv = nn.Conv2d(1, 1, kernel_size=self.ring_mat.shape[0],
                               stride=1,
                               padding=int((self.ring_mat.shape[0]-1)/2),
                               bias=False)
        with torch.no_grad():
            inner_conv.weight = nn.Parameter(inner_weights)
            outer_conv.weight = nn.Parameter(outer_weights)

        error = inputs - targets
        sigma = torch.sqrt(torch.var(error))
        print('sigma = ', sigma)

        inner_shape = inner_conv(error).shape
        print('inner shape = ', inner_shape)
        outer_shape = outer_conv(error).shape
        print('outer shape = ', outer_shape)

        stat = self.coefficient * \
               (inner_conv(error) - outer_conv(error)) / sigma
        map_ = 0.5*torch.erfc(stat/self.sqrt2)

        threshold = nn.Threshold(0.9, 0)
        th_map = threshold(map_)
        area = torch.count_nonzero(th_map) / torch.numel(th_map)

        tmp = (inputs - targets) ** 2
        mse = torch.mean(tmp)

        print('area = ', area)
        print('mse = ', mse)

        return area + self.beta*mse
