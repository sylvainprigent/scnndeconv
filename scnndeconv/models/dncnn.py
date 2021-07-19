import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class DnCNN(pl.LightningModule):
    def __init__(self):
        super(DnCNN, self).__init__(data_channels = 1, layers = 7, layers_channels = 64)
        self.dataset = None
        
        if layers < 3:
            raise Exception('DnCNN layers must be greater or equal than 3')

        self.data_channels = data_channels
        self.layers = layers
        self.layers_channels = layers_channels

        # convolution layers
        self._conv1 = nn.Conv2d(in_channels = self.data_channels, out_channels = self.layers_channels, kernel_size = 3, padding=1)
        self._hidden_conv_layers = []
        for i in range(self.layers - 2):
            self._hidden_conv_layers.append(nn.Conv2d(in_channels = self.layers_channels, out_channels = self.layers_channels, kernel_size = 3, padding=1))

        self._convN = nn.Conv2d(self.layers_channels, out_channels = self.data_channels, kernel_size = 3, padding=1)
        
        # batch norms layers
        self._hidden_bn_layers = []
        for i in range(self.layers - 2):
            self._hidden_bn_layers.append(nn.BatchNorm2d(self.layers_channels, self.layers_channels))

    def forward(self, x):
        in_data = F.relu(self._conv1(x))
        for i in range(self.layers - 2): 
            in_data = F.relu(self._hidden_bn_layers[i](self._hidden_conv_layers[i](in_data)))
        residual = self._convN(in_data)
        y = residual + x
      
        return y 

    def train_dataloader(self):
        if self.dataset is None:
            return DataLoader(DefaultDataset(self.dataset_dir), batch_size=20)
        else:
            return self.dataset    

  
    def training_step(self, batch, batch_nb):
        x, y = batch
        out = self(x)
        mse = nn.MSELoss()
        loss = mse(y, out)

        tensorboard_logs = {'train_loss': loss}
        return {'loss' : loss, 'log' : tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train(dataset, max_epochs=40, checkpoints_dir='./checkpoints/'):
        self.dataset = dataset
        checkpoint_callback = ModelCheckpoint(filepath=checkpoints_dir,  
                                              save_top_k=1,  
                                              monitor='loss',
                                              verbose=True)
        trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, checkpoint_callback=checkpoint_callback)
        trainer.fit(self)  
