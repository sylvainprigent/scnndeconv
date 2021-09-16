import os
import numpy as np
from scnndeconv.models import DnCNN, DnCNNMapLoss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from skimage import io, metrics
import matplotlib.pyplot as plt

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device: {torch.cuda.current_device()}")
print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

#checkpoints_file = '/home/sprigent/Documents/codes/scnndeconv/lightning_logs/version_1/checkpoints/epoch=49-step=56299.ckpt'

#checkpoints_file = '/home/sprigent/Documents/codes/scnndeconv/lightning_logs/version_2/checkpoints/epoch=49-step=56299.ckpt'
#checkpoints_file = '/home/sprigent/Documents/codes/scnndeconv/lightning_logs/version_23/checkpoints/epoch=49-step=198449.ckpt'
#checkpoints_file = '/home/sprigent/Documents/codes/scnndeconv/lightning_logs/version_27/checkpoints/epoch=0-step=3968.ckpt'
#checkpoints_file = '/home/sprigent/Documents/codes/scnndeconv/lightning_logs/version_28/checkpoints/epoch=0-step=3968.ckpt'
#checkpoints_file = '/home/sprigent/Documents/codes/scnndeconv/lightning_logs/version_29/checkpoints/epoch=49-step=198449.ckpt'
#checkpoints_file = '/home/sprigent/Documents/codes/scnndeconv/lightning_logs/version_23_dncnn_50/checkpoints/epoch=49-step=198449.ckpt'
checkpoints_file = '/home/sprigent/Documents/codes/scnndeconv/lightning_logs/version_32/checkpoints/epoch=0-step=3968.ckpt' # A+0.01MSE loss

pretrained_model = DnCNNMapLoss.load_from_checkpoint(checkpoints_file)


gt_image_file = '/home/sprigent/Documents/datasets/airbus/original/test/GT/Brest0146_6836_50cm_F06opt.tif'
test_img_file = '/home/sprigent/Documents/datasets/airbus/original/test/noise120/Brest0146_6836_50cm_F06opt.tif'
gt_img = np.float32(io.imread(gt_image_file))
test_img = np.float32(io.imread(test_img_file))

maxi = test_img.max()

with torch.no_grad():
  out = pretrained_model(torch.from_numpy(test_img).unsqueeze(0).unsqueeze(0))


mse_noisy = metrics.mean_squared_error(gt_img, test_img)
mse_dncnn = metrics.mean_squared_error(gt_img, out[0, 0, :, :].numpy())

print('MSE noisy:', mse_noisy)
print('MSE dncnn:', mse_dncnn)

plt.figure()
plt.imshow(test_img, cmap='gray')
plt.figure()
plt.imshow(out[0, 0, :, :], cmap='gray')
plt.figure()
plt.imshow(np.float32(test_img) - out[0, 0, :, :].numpy(), cmap='gray')
plt.show()
