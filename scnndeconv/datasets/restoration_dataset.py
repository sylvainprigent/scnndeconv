"""Default dataset for training image restoration network

Classes
-------
RestorationDataset


"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from natsort import natsorted
from skimage import io


class RestorationDataset(Dataset):
    """Dataset to train from patches

    All the training images must be saved as individual images in source and
    target folders.

    Parameters
    ----------
    source_dir: str
        Path of the noisy training images (or patches)
    target_dir: str
        Path of the ground truth images (or patches)

    """
    def __init__(self, source_dir, target_dir):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")

        self.source_images = natsorted(os.listdir(source_dir))
        self.target_images = natsorted(os.listdir(target_dir))
        if len(self.source_images) != len(self.target_images):
            raise Exception("Source and target dirs are not the same length")

    def __len__(self):
        return len(self.source_images)

    def __getitem__(self, idx):
        source_path = os.path.join(self.source_dir, self.source_images[idx])
        target_path = os.path.join(self.target_dir, self.target_images[idx])
        source_image = np.float32(io.imread(source_path))
        target_image = np.float32(io.imread(target_path))

        # data augmentation
        k_1, k_2 = np.random.randint(4), np.random.randint(3)
        target_image = np.rot90(target_image, k_1)
        source_image = np.rot90(source_image, k_1)
        if k_2 < 2:
            target_image = np.flip(target_image, k_2)
            source_image = np.flip(source_image, k_2)

        s_image = torch.from_numpy(source_image.copy()).\
            view(1, *source_image.shape).to(self.device)
        t_image = torch.from_numpy(target_image.copy()).\
            view(1, *target_image.shape).to(self.device)

        return s_image, t_image
