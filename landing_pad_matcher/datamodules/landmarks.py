import multiprocessing as mp
from pathlib import Path
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from landing_pad_matcher.datasets.landmarks import LandmarksDataset


class LandmarksDataModule(LightningDataModule):
    def __init__(self, data_path: Path, textures_path: Path, batch_size: int, validation_batch_size: int,
                 number_of_workers: Optional[int] = mp.cpu_count()):
        super().__init__()

        self._data_path = data_path
        self._textures_path = textures_path
        self._batch_size = batch_size
        self._validation_batch_size = validation_batch_size
        self._number_of_workers = number_of_workers

        self.train_dataset = None
        self.valid_dataset = None

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = LandmarksDataset(self._data_path, self._textures_path, num_samples=500000)
        self.valid_dataset = LandmarksDataset(self._data_path, self._textures_path, num_samples=50000)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self._batch_size, num_workers=self._number_of_workers,
            pin_memory=True, drop_last=True, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=self._validation_batch_size, num_workers=self._number_of_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=self._validation_batch_size, num_workers=self._number_of_workers,
            pin_memory=True
        )
