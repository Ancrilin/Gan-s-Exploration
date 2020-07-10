from .dataset import OOSDataset
from .dataset import SMPDataset
from .pos_tagging_dataset import PosOOSDataset
from torch.utils.data import DataLoader, Dataset


def get_dataloader(torch_dataset: Dataset, batch_size, shuffle, num_workers=2) -> DataLoader:
    return DataLoader(torch_dataset, batch_size, shuffle, num_workers=num_workers)
