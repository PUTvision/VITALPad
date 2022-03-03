from pathlib import Path
from typing import Optional, Tuple, List

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

    @staticmethod
    def get_paths(data_path: Path) -> List[Tuple[Path, Path]]:
        paths = []
        for data_dir in data_path.iterdir():
            rgb_paths = sorted((data_dir / 'rgb').iterdir())
            seg_paths = sorted((data_dir / 'seg').iterdir())
            assert len(rgb_paths) == len(seg_paths)

            paths.extend(zip(rgb_paths, seg_paths))

        return paths

    def setup(self, stage: Optional[str] = None):
        paths = self.get_paths(self._data_path)
        train_paths, val_paths = train_test_split(paths, test_size=0.15, random_state=42)

        self.train_dataset = DensityDataset(train_paths, augment=True)
        self.val_dataset = DensityDataset(val_paths, augment=False)

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
