import os
import numpy as np
from scnndeconv.losses import SAContrarioLoss
import torch
from torch import nn
from skimage import io, metrics
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gt_image_file = '/home/sprigent/Documents/codes/scnndeconv/data/data/Lyon_1842_5175_50cm_F06opt_32bits.tif'
test_img_file = '/home/sprigent/Documents/codes/scnndeconv/data/data/Lyon_1842_5175_50cm_F06opt_w1.0_reg12_HSV.tif'
gt_img = np.float32(io.imread(gt_image_file))
test_img = np.float32(io.imread(test_img_file))

gt_img = gt_img[1157:1492, 731:1107]
test_img = test_img[1157:1492, 731:1107]

gt_img_torch = torch.from_numpy(gt_img).view(1, 1, *gt_img.shape).float().to(device)
test_img_torch = torch.from_numpy(test_img).view(1, 1, *test_img.shape).float().to(device)

loss_function = nn.MSELoss()
mse_loss = loss_function(gt_img_torch, test_img_torch)

loss_acontrario = SAContrarioLoss(radius=1, alpha=7, beta=0.01)
a_loss = loss_acontrario(gt_img_torch, test_img_torch)

print('loss = ', mse_loss.item())
print('a loss = ', a_loss.item())
