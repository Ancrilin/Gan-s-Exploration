# coding: utf-8
# @author: Ross
# @file: loader.py
# @time: 2020/01/13
# @contact: devross@gmail.com

from torch.utils.data import Dataset
import torch
import numpy as np


class PosOOSDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = np.array(dataset)

    def __getitem__(self, index: int):
        token_ids, mask_ids, type_ids, label_ids, pos1, pos2, pos_mask= self.dataset[index]
        return (torch.tensor(token_ids, dtype=torch.long),
                torch.tensor(mask_ids, dtype=torch.long),
                torch.tensor(type_ids, dtype=torch.long),
                torch.tensor(pos1, dtype=torch.float),
                torch.tensor(pos2),
                torch.tensor(pos_mask),
                torch.tensor(label_ids, dtype=torch.float32),
                )

    def __len__(self) -> int:
        return len(self.dataset)
