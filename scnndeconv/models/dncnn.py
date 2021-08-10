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
    def __init__(self, channels=1, features=64, data_loader=None):
        super(DnCNN, self).__init__()
        datasets_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_source_dir = os.path.join(datasets_dir, '../datasets/airbus/train/noise120/')
        self.dataset_target_dir = os.path.join(datasets_dir, '../datasets/airbus/train/GT/')
        self.data_loader = data_loader

        # convolution layers
        self._conv1 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1, bias=False)

        self._conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)
        self._conv3 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)
        self._conv4 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)
        self._conv5 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)
        self._conv6 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)
        self._conv7 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)

        self._convN = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=3, padding=1, bias=False)

        # batch norms layers
        self._bn2 = nn.BatchNorm2d(features, features)
        self._bn3 = nn.BatchNorm2d(features, features)
        self._bn4 = nn.BatchNorm2d(features, features)
        self._bn5 = nn.BatchNorm2d(features, features)
        self._bn6 = nn.BatchNorm2d(features, features)
        self._bn7 = nn.BatchNorm2d(features, features)

    def forward(self, x):
        in_data = F.relu(self._conv1(x))
        in_data = F.relu(self._bn2(self._conv2(in_data)))
        in_data = F.relu(self._bn3(self._conv3(in_data)))
        in_data = F.relu(self._bn4(self._conv4(in_data)))
        in_data = F.relu(self._bn5(self._conv5(in_data)))
        in_data = F.relu(self._bn6(self._conv6(in_data)))
        in_data = F.relu(self._bn7(self._conv7(in_data)))
        residue = self._convN(in_data)
        return residue
        #y = x+residue
        #return y

    def train_dataloader(self):
        if self.data_loader is None:
            return DataLoader(RestorationDataset(self.dataset_source_dir, self.dataset_target_dir, transform=ToTensor(),
                                                 target_transform=ToTensor()), batch_size=128)
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
