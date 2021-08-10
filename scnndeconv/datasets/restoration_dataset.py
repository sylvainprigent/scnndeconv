import os
import numpy as np
from torch.utils.data import Dataset
from natsort import natsorted
from torchvision import datasets, transforms
from skimage import io


class RestorationDataset(Dataset):
    def __init__(self, source_dir, target_dir, transform=None, target_transform=None):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.transform = transform
        self.target_transform = target_transform

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
        if self.transform:
            source_image = self.transform(source_image)
        if self.target_transform:
            target_image = self.target_transform(target_image)

        #source_image = source_image.view(1, 1, source_image.shape[0], source_image.shape[1])
        #target_image = target_image.view(1, 1, target_image.shape[0], target_image.shape[1])
        #return source_image/source_image.max(), target_image/source_image.max()

        return source_image, target_image
