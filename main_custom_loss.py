import os
import numpy as np
from scnndeconv.losses import SMSELoss, SAContrarioLoss
import torch
from skimage import io, metrics
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gt_image_file = '/Users/sprigent/Documents/data/2021-03-24_airbus_denoising/test/Brest0146_6836_50cm_F06opt.png'
test_img_file = '/Users/sprigent/Documents/data/2021-03-24_airbus_denoising/test/Brest0146_6836_50cm_F06opt_B120.png'
gt_img = np.float32(io.imread(gt_image_file))
test_img = np.float32(io.imread(test_img_file))

gt_img_torch = torch.from_numpy(gt_img).view(1, 1, *gt_img.shape).float()
test_img_torch = torch.from_numpy(test_img).view(1, 1, *test_img.shape).float()

loss_function = SMSELoss()
mse_loss = loss_function(gt_img_torch, test_img_torch)

loss_acontrario = SAContrarioLoss(radius=1, alpha=7, beta=0.001)
a_loss = loss_acontrario(gt_img_torch, test_img_torch)

print('loss = ', mse_loss.item())
print('a loss = ', a_loss.item())
