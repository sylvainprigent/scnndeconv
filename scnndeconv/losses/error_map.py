"""Define deep learning losses based on error maps

Classes
-------
SMSELoss
SAContrarioLoss

"""

import math
import torch
from torch import nn

from .utils import disk_patch, ring_patch
#from skimage import io


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

    def __init__(self, radius=1, alpha=7, beta=1, gamma=2):
        super().__init__()

        self.beta = beta
        self.gamma = gamma
        self.disk_mat = disk_patch(radius)
        self.ring_mat = ring_patch(radius, alpha)
        self.coefficient = math.sqrt(math.pi) * \
                           math.sqrt(1 - 1 / (alpha * alpha)) * radius
        self.sqrt2 = math.sqrt(2)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs, targets):
        """Calculate the loss

        Parameters
        ----------
        inputs: Tensor
            Network prediction
        targets: Tensor
            Expected result

        """
        # positive error map
        error_pos = inputs - targets
        inner_weights_pos = torch.Tensor(self.disk_mat).unsqueeze(0).unsqueeze(0).to(self.device)
        inner_weights_pos.require_grad = True
        outer_weights_pos = torch.Tensor(self.ring_mat).unsqueeze(0).unsqueeze(0).to(self.device)
        outer_weights_pos.require_grad = True

        inner_conv_pos = nn.Conv2d(1, 1, kernel_size=self.disk_mat.shape[0],
                                   stride=1,
                                   padding=int((self.disk_mat.shape[0] - 1) / 2),
                                   bias=False)
        outer_conv_pos = nn.Conv2d(1, 1, kernel_size=self.ring_mat.shape[0],
                                   stride=1,
                                   padding=int((self.ring_mat.shape[0] - 1) / 2),
                                   bias=False)
        with torch.no_grad():
            inner_conv_pos.weight = nn.Parameter(inner_weights_pos)
            outer_conv_pos.weight = nn.Parameter(outer_weights_pos)

        sigma_pos = torch.sqrt(torch.var(error_pos))
        stat_pos = self.coefficient * (inner_conv_pos(error_pos) - outer_conv_pos(error_pos)) / sigma_pos

        threshold_pos = nn.Threshold(0.9, -1)
        stat_pos_norm = 0.5 * torch.erfc(stat_pos / self.sqrt2)
        th_map_pos = threshold_pos(stat_pos_norm)

        threshold_pos2 = nn.Threshold(0, 0)
        th_map_pos2 = 1 - threshold_pos2(-th_map_pos)

        # negative error map
        error_neg = targets - inputs
        inner_weights_neg = torch.Tensor(self.disk_mat).unsqueeze(0).unsqueeze(0).to(self.device)
        inner_weights_neg.require_grad = True
        outer_weights_neg = torch.Tensor(self.ring_mat).unsqueeze(0).unsqueeze(0).to(self.device)
        outer_weights_neg.require_grad = True

        inner_conv_neg = nn.Conv2d(1, 1, kernel_size=self.disk_mat.shape[0],
                                   stride=1,
                                   padding=int((self.disk_mat.shape[0] - 1) / 2),
                                   bias=False)
        outer_conv_neg = nn.Conv2d(1, 1, kernel_size=self.ring_mat.shape[0],
                                   stride=1,
                                   padding=int((self.ring_mat.shape[0] - 1) / 2),
                                   bias=False)
        with torch.no_grad():
            inner_conv_neg.weight = nn.Parameter(inner_weights_neg)
            outer_conv_neg.weight = nn.Parameter(outer_weights_neg)

        sigma_neg = torch.sqrt(torch.var(error_neg))
        stat_neg = self.coefficient * (inner_conv_neg(error_neg) - outer_conv_neg(error_neg)) / sigma_neg

        threshold_neg = nn.Threshold(0.9, -1)
        stat_neg_norm = 0.5 * torch.erfc(stat_neg / self.sqrt2)
        th_map_neg = threshold_neg(stat_neg_norm)

        threshold_neg2 = nn.Threshold(0, 0)
        th_map_neg2 = 1 - threshold_neg2(-th_map_neg)

        # map combination
        th_map = th_map_pos2+th_map_neg2

        #io.imsave('error_map.tif', th_map[0, 0, :, :].cpu().detach().numpy())

        # Map area to criterion
        area = torch.sum(th_map)/torch.numel(th_map)

        # Map density criterion
        dense_weights = torch.Tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).unsqueeze(0).unsqueeze(0).to(self.device)
        dense_weights.require_grad = True
        dense_conv = nn.Conv2d(1, 1, kernel_size=self.disk_mat.shape[0],
                               stride=1,
                               padding=1,
                               bias=False)
        with torch.no_grad():
            dense_conv.weight = nn.Parameter(dense_weights)
        dense_map = dense_conv(th_map)*th_map
        threshold_dense = nn.Threshold(4, -1)

        threshold_dense2 = nn.Threshold(0, 0)
        th_dense = 1 - threshold_dense2(-threshold_dense(dense_map))

        #io.imsave('error_map_density.tif',
        #          th_dense[0, 0, :, :].cpu().detach().numpy())

        density = torch.sum(th_dense) / torch.numel(th_map)

        # MSE
        mse = torch.mean((inputs - targets) ** 2)

        #print('MSE=', mse)
        #print('area=', area)
        #print('density=', density)
        output = area + self.gamma*density + self.beta*mse
        return output
