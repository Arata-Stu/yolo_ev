import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig

from yolo_ev.module.data.dataset.coco.build_coco_dataset import build_coco_dataset

class DataModule(pl.LightningDataModule):

    def __init__(self, full_config: DictConfig):
        super().__init__()
        self.full_config = full_config

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = build_coco_dataset(self.full_config, mode="train")
            self.valid_dataset = build_coco_dataset(self.full_config, mode="val")
        if stage == 'test':
            self.test_dataset = build_coco_dataset(self.full_config, mode="test")

    def train_dataloader(self):
        if len(self.train_dataset) != 0:
            return DataLoader(
                self.train_dataset,
                batch_size=self.full_config.dataset.train.batch_size,
                shuffle=True,
                num_workers=self.full_config.dataset.train.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        else:
            raise ValueError('Train dataset is empty. Please check the dataset path or configuration.')

    def val_dataloader(self):
        if len(self.valid_dataset) != 0:
            return DataLoader(
                self.valid_dataset,
                batch_size=self.full_config.dataset.val.batch_size,
                shuffle=False,
                num_workers=self.full_config.dataset.val.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        else:
            raise ValueError('Validation dataset is empty. Please check the dataset path or configuration.')

    def test_dataloader(self):
        if len(self.test_dataset) != 0:
            return DataLoader(
                self.test_dataset,
                batch_size=self.full_config.dataset.test.batch_size,
                shuffle=False,
                num_workers=self.full_config.dataset.test.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        else:
            raise ValueError('Test dataset is empty. Please check the dataset path or configuration.')
