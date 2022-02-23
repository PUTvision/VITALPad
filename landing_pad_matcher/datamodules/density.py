from pathlib import Path
from typing import Optional

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from landing_pad_matcher.datasets.density import DensityDataset


class DensityDataModule(LightningDataModule):
    def __init__(self, data_path: Path, batch_size: int, validation_batch_size: int, number_of_workers: int):
        super().__init__()

        self._data_path = data_path
        self._batch_size = batch_size
        self._validation_batch_size = validation_batch_size
        self._number_of_workers = number_of_workers

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        file_ids = sorted([path.stem.split('_')[1] for path in (self._data_path / 'rgb').iterdir()])
        train_file_ids, val_file_ids = train_test_split(file_ids, test_size=0.15, random_state=42)

        self.train_dataset = DensityDataset(self._data_path, train_file_ids, augment=True)
        self.val_dataset = DensityDataset(self._data_path, val_file_ids, augment=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self._batch_size, num_workers=self._number_of_workers,
            pin_memory=True, drop_last=True, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self._validation_batch_size, num_workers=self._number_of_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self._validation_batch_size, num_workers=self._number_of_workers,
            pin_memory=True
        )
