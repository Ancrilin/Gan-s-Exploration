from .dataset import OOSDataset
from torch.utils.data import DataLoader, Dataset


def get_dataloader(torch_dataset: Dataset, batch_size, shuffle, num_workers=2) -> DataLoader:
    return DataLoader(torch_dataset, batch_size, shuffle, num_workers=num_workers)
