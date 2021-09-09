import os
import numpy as np
from skimage import io, metrics
import matplotlib.pyplot as plt


gt_image_file = '/home/sprigent/Documents/datasets/airbus/original/test/GT/Brest0146_6836_50cm_F06opt.tif'
test_img_file = '/home/sprigent/Documents/datasets/airbus/original/test/noise120/Brest0146_6836_50cm_F06opt.tif'
dncnn_image_file = 'dncnn8_Brest0146_6836_50cm_F06opt_B120.tif'
gt_img = np.float32(io.imread(gt_image_file))
test_img = np.float32(io.imread(test_img_file))
dncnn_image = np.float32(io.imread(dncnn_image_file))

mse_noisy = metrics.mean_squared_error(gt_img, test_img)
mse_dncnn = metrics.mean_squared_error(gt_img, dncnn_image)

print('MSE noisy:', mse_noisy)
print('MSE dncnn:', mse_dncnn)

