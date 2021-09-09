import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scnndeconv.datasets import RestorationDataset
from torchvision.transforms import ToTensor


class DnCNN(pl.LightningModule):
    def __init__(self, num_of_layers=8, channels=1, features=64, data_loader=None):
        super(DnCNN, self).__init__()
        datasets_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_source_dir = os.path.join(datasets_dir, '../datasets/airbus/train/noise120/')
        self.dataset_target_dir = os.path.join(datasets_dir, '../datasets/airbus/train/GT/')
        self.data_loader = data_loader

        kernel_size = 3
        padding = 1
        layers = [nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                            bias=True),
                  nn.ReLU(inplace=True)]
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)
        print(sum(p.numel() for p in self.dncnn.parameters() if p.requires_grad))

    def forward(self, x):
        y = x
        residue = self.dncnn(x)
        return y-residue

    def train_dataloader(self):
        if self.data_loader is None:
            return DataLoader(RestorationDataset(self.dataset_source_dir, self.dataset_target_dir), batch_size=128,
                              shuffle=True, drop_last=True)
        else:
            return self.data_loader

    def training_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)
        mse = nn.MSELoss()
        loss = mse(y, out)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def train_dncnn(model, max_epochs=50, checkpoints_dir='./checkpoints/'):

    #checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_dir)
    #checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_dir,
    #                                      save_top_k=1,
    #                                      monitor='val_loss',
    #                                      verbose=True)
    trainer = pl.Trainer(gpus=1, max_epochs=max_epochs)
    trainer.fit(model)
