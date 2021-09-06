from scnndeconv.models import DnCNN, train_dncnn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device: {torch.cuda.current_device()}")
print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

model = DnCNN()
train_dncnn(model, max_epochs=50, checkpoints_dir='/home/sprigent/Documents/codes/scnndeconv/checkpoints/')
