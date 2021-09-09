import os
import numpy as np
from scnndeconv.models import DnCNN, train_dncnn
from scnndeconv.losses import ErrorMapLoss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from skimage import io, metrics
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gt_image_file = '/home/sprigent/Documents/datasets/airbus/original/test/GT/Brest0146_6836_50cm_F06opt.tif'
test_img_file = '/home/sprigent/Documents/datasets/airbus/original/test/noise120/Brest0146_6836_50cm_F06opt.tif'
gt_img = np.float32(io.imread(gt_image_file))
test_img = np.float32(io.imread(test_img_file))

gt_img_torch = torch.from_numpy(gt_img).view(1, 1, *gt_img.shape).float().to(device)
test_img_torch = torch.from_numpy(test_img).view(1, 1, *test_img.shape).float().to(device)

loss_function = ErrorMapLoss()
lass = loss_function(gt_img_torch, test_img_torch)

print('loss = ', lass.item())
