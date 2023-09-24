import glob
import numpy as np
import torch
import loguru
from loguru import logger

class VoidDataset(torch.utils.data.Dataset):

    def __init__(self, cfg):
        self.cfg = cfg
        pass

    def __len__(self):
        return 256

    def __getitem__(self, idx):
        return torch.tensor(0)
    
def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.solver.batch_size
        shuffle = cfg.input.shuffle
        if not shuffle:
            logger.warning("SHUFFLE is FALSE!")
    else:
        batch_size = cfg.test.batch_size
        shuffle = False

    dataset = VoidDataset(cfg)

    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
        )
    return data_loader

