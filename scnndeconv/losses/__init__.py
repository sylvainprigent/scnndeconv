"""Losses module

Module that implement losses dedicated to image denoising and deconvolution

"""

from .error_map import SAContrarioLoss

__all__ = ['SAContrarioLoss']
