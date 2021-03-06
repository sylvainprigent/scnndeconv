"""Datasets module

Module that contains all the available datasets

"""

from .restoration_dataset import RestorationDataset
from .airbus_dataset import AirbusDataset

__all__ = ['RestorationDataset',
           'AirbusDataset']
