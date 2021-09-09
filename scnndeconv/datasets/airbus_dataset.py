"""Datasets to train on the Airbus dataset

Classes
-------
AirbusDataset

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io


class AirbusDataset(Dataset):
    """Dataset to train on the airbus dataset

    The images in the training folders must be names with the pattern:
    'img_xxxx.tif' where xxxx is the id number of the image

    Parameters
    ----------
    folder_gt_train_250x250: str
        Path to the training ground truth data: 250x250 gray scaled images
    folder_noisy_train_250x250: str
        Path to the training noisy data: 250x250 gray scaled images
    patch_size: int
        Size of the training patches
    stride: int
        Tiling shift to extract patches in raw images

    """
    def __init__(self, folder_gt_train_250x250, folder_noisy_train_250x250,
                 patch_size=40, stride=10):
        self.folder_gt = folder_gt_train_250x250
        self.folder_noisy = folder_noisy_train_250x250
        self.nb_images = len(os.listdir(folder_gt_train_250x250))
        self.stride = stride
        self.patch_size = patch_size
        self.n_patches = self.nb_images * ((250 - patch_size) // stride) * \
                         ((250 - patch_size) // stride)
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        nb_patch_per_img = self.n_patches // self.nb_images

        img_number = idx // nb_patch_per_img
        elt = 'img_' + str(img_number).zfill(4) + '.tif'

        img_np = np.float32(io.imread(os.path.join(self.folder_gt, elt)))
        img_noisy_np = np.float32(io.imread(os.path.join(
            self.folder_noisy, elt)))

        nb_patch_w = (img_np.shape[1] - self.patch_size) // self.stride
        idx = idx % nb_patch_per_img
        i, j = idx // nb_patch_w, idx % nb_patch_w
        patch = img_np[i*self.stride:i*self.stride+self.patch_size,
                       j*self.stride:j*self.stride+self.patch_size]
        patch_noisy = img_noisy_np[i*self.stride:i*self.stride+self.patch_size,
                                   j*self.stride:j*self.stride+self.patch_size]

        k_val = np.random.randint(4)
        patch = np.rot90(patch, k_val)
        patch_noisy = np.rot90(patch_noisy, k_val)
        patch = np.ascontiguousarray(patch)
        patch_noisy = np.ascontiguousarray(patch_noisy)

        img_torch = torch.from_numpy(patch).view(1, *patch.shape).\
            float().to(self.device)
        img_noisy_torch = torch.from_numpy(patch_noisy).view(1, *patch.shape).\
            float().to(self.device)

        return img_torch, img_noisy_torch
