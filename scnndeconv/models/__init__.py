"""Deep learning models

Implementation of deep learning models for image denoising and deconvolution

"""
from .dncnn import DnCNN, train_dncnn
from .dncnn_errormap import DnCNNMapLoss

__all__ = ['DnCNN', 'train_dncnn', 'DnCNNMapLoss']
