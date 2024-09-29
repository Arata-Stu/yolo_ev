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

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = build_coco_dataset(self.full_config, mode="train")
            print(f"Train dataset size: {len(self.train_dataset)}")
            self.valid_dataset = build_coco_dataset(self.full_config, mode="val")
            print(f"Validation dataset size: {len(self.valid_dataset)}")

    def train_dataloader(self):
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.full_config.dataset.train.batch_size,
            shuffle=True,
            num_workers=self.full_config.dataset.train.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
    def val_dataloader(self):
        
        return DataLoader(
            self.valid_dataset,
            batch_size=self.full_config.dataset.val.batch_size,
            shuffle=False,
            num_workers=self.full_config.dataset.val.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        
    def test_dataloader(self):
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.full_config.dataset.test.batch_size,
            shuffle=False,
            num_workers=self.full_config.dataset.test.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        