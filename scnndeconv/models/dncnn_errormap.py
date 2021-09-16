"""CnCNN pytorch lightning module

Implementation of the DnCNN network in pytorch lightning

Classes
-------
DnCNN

Methods
-------
train_dncnn

"""

import os
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from scnndeconv.datasets import RestorationDataset

from scnndeconv.losses import SAContrarioLoss


class DnCNNMapLoss(pl.LightningModule):
    """Implementation of the DnCNN network

    Parameters
    ----------
    num_of_layers: int
        Number of layers in the model
    channels: int
        Number of channels in the images
    features: int
        Number of features in hidden layers
    data_loader: DataLoader
        Data loader that manage the training

    """

    def __init__(self, num_of_layers=8, channels=1, features=64,
                 data_loader=None):
        super().__init__()
        datasets_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_source_dir = os.path.join(datasets_dir, 'train/noisy/')
        self.dataset_target_dir = os.path.join(datasets_dir, 'train/gt/')
        self.data_loader = data_loader

        kernel_size = 3
        padding = 1
        layers = [nn.Conv2d(in_channels=channels, out_channels=features,
                            kernel_size=kernel_size, padding=padding,
                            bias=True),
                  nn.ReLU(inplace=True)]
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features,
                          kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels,
                                kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)
        print(
            sum(p.numel() for p in self.dncnn.parameters() if p.requires_grad))

    def forward(self, x):
        """Network forward method

        Parameters
        ----------
        x: Tensor
            Network input batch

        Returns
        -------
        Tensor containing the network output

        """
        y = x
        residue = self.dncnn(x)
        return y - residue

    def train_dataloader(self):
        """Load the dataset

        Returns
        -------
        A data loader for the dataset

        """
        if self.data_loader is None:
            return DataLoader(RestorationDataset(self.dataset_source_dir,
                                                 self.dataset_target_dir),
                              batch_size=128,
                              shuffle=True,
                              drop_last=True)
        return self.data_loader

    def training_step(self, batch, batch_idx):
        """One training step

        Parameters
        ----------
        batch: tuple
            Batch data
        batch_idx: int
            index of the batch

        """
        x, y = batch
        out = self.forward(x)
        mse = SAContrarioLoss(radius=1, alpha=1, beta=0.01, gamma=2)
        loss = mse(y, out)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        """Configure the optimizer"""
        return torch.optim.Adam(self.parameters(), lr=0.001)
