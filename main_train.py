import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader

from scnndeconv.models import DnCNN, DnCNNMapLoss
from scnndeconv.datasets import AirbusDataset

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device: {torch.cuda.current_device()}")
print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")


dataset_source_dir = '/home/sprigent/Documents/datasets/airbus/original_250/train/noise120/'
dataset_target_dir = '/home/sprigent/Documents/datasets/airbus/original_250/train/GT'
data_loader = DataLoader(AirbusDataset(dataset_source_dir, dataset_target_dir, patch_size=40, stride=10),
                         batch_size=128,
                         shuffle=True,
                         drop_last=True,
                         num_workers=0)
model = DnCNNMapLoss(num_of_layers=8, channels=1, features=64, data_loader=data_loader)
#model = DnCNN(num_of_layers=8, channels=1, features=64, data_loader=data_loader)
trainer = pl.Trainer(gpus=1, max_epochs=50)
trainer.fit(model)
