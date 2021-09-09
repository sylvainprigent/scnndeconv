import os
import numpy as np
import torch
from torch.utils.data import Dataset
from natsort import natsorted
from torchvision import datasets, transforms
from skimage import io


class AirbusDataset(Dataset):

    def __init__(self, folder_gt_train_250x250, folder_noisy_train_250x250, patch_size=40, stride=10):
        self.folder_gt = folder_gt_train_250x250
        self.folder_noisy = folder_noisy_train_250x250
        self.nb_images = len(os.listdir(folder_gt_train_250x250))
        self.stride = stride
        self.patch_size = patch_size
        self.n_patches = self.nb_images * ((250 - patch_size) // stride) * ((250 - patch_size) // stride)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        nb_patch_per_img = self.n_patches // self.nb_images

        img_number = idx // nb_patch_per_img
        elt = 'img_' + str(img_number).zfill(4) + '.tif'

        img_path = os.path.join(self.folder_gt, elt)
        #print('img_path = ', img_path)
        img_np = np.float32(io.imread(img_path))

        img_noisy_path = os.path.join(self.folder_noisy, elt)
        #print('img_noisy_path = ', img_noisy_path)
        img_noisy_np = np.float32(io.imread(img_noisy_path))

        p = self.patch_size
        s = self.stride

        h, w = img_np.shape
        nb_patch_w = (w - p) // s
        idx = idx % nb_patch_per_img
        i, j = idx // nb_patch_w, idx % nb_patch_w
        patch = img_np[i*s:i*s+p, j*s:j*s+p]
        patch_noisy = img_noisy_np[i*s:i*s+p, j*s:j*s+p]

        k = np.random.randint(4)
        patch = np.rot90(patch, k)
        patch_noisy = np.rot90(patch_noisy, k)
        patch = np.ascontiguousarray(patch)
        patch_noisy = np.ascontiguousarray(patch_noisy)

        img_torch = torch.from_numpy(patch).view(1, *patch.shape).float().to(self.device)
        img_noisy_torch = torch.from_numpy(patch_noisy).view(1, *patch.shape).float().to(self.device)

        return img_torch, img_noisy_torch
