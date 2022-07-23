from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import torch.utils
import torch.utils.data
from albumentations import Compose, Affine, SmallestMaxSize, RandomCrop, CenterCrop
from albumentations.augmentations.transforms import (
    ToFloat, Flip, RandomShadow, MotionBlur, ColorJitter, RandomGamma, RandomSunFlare, ISONoise, PadIfNeeded
)
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import gaussian_filter


class DensityDataset(torch.utils.data.Dataset):
    def __init__(self, paths: List[Tuple[Path, Path]], augment: bool = False):
        self._paths = paths
        self._augment = augment

        self._augmentations = Compose([
            SmallestMaxSize(max_size=480, interpolation=cv2.INTER_AREA),
            RandomCrop(height=480, width=640),
            RandomGamma(gamma_limit=(80, 120)),
            ColorJitter(brightness=0.2, contrast=0.2, hue=0.1, saturation=0.5),
            RandomSunFlare(),
            RandomShadow(num_shadows_upper=5),
            ISONoise(),
            MotionBlur(),
            Flip(),
            Affine(rotate=(-10, 10), scale=(0.9, 1.1)),
            ToFloat(max_value=255.0),
            ToTensorV2()
        ])
        self._transforms = Compose([
            SmallestMaxSize(max_size=480, interpolation=cv2.INTER_AREA),
            CenterCrop(height=480, width=640),
            ToFloat(max_value=255.0),
            ToTensorV2()
        ])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, gt = self._load_data(index)
        image, gt = self._perform_augmentations(image, gt)
        gt = self._transform_gt(gt)

        return image, gt

    def __len__(self) -> int:
        return len(self._paths)

    def _perform_augmentations(self, rgb: np.ndarray, gt: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._augment:
            transformed = self._augmentations(image=rgb, mask=gt)
        else:
            transformed = self._transforms(image=rgb, mask=gt)

        return transformed['image'], transformed['mask']

    @staticmethod
    @torch.no_grad()
    def _transform_gt(original_gt: torch.Tensor) -> torch.Tensor:
        # The aim of the code below is to produce softmax-like output
        gt = torch.ones(2, *original_gt.shape[:2], dtype=torch.float32)
        for i, color in enumerate(((255, 0, 0),)):  # enumerate(((255, 0, 0), (0, 255, 0)))
            mask = torch.eq(original_gt, torch.tensor(color)).all(dim=-1).type(torch.float32)
            mask = torch.from_numpy(gaussian_filter(mask, sigma=(1, 1)))
            gt[i + 1] = mask

        # If people and landing pad masks overlap, divide them so that their sum is equal to 1
        divisors = torch.sum(gt[1:], dim=0)
        divisors[divisors < 1] = 1
        gt[1:] /= divisors

        # Remove background from where people or landing pad appear
        gt[0] -= torch.sum(gt[1:], dim=0)

        return gt

    def _load_data(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        rgb_path, gt_path = self._paths[index]
        rgb_image = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        gt_image = cv2.cvtColor(cv2.imread(str(gt_path)), cv2.COLOR_BGR2RGB)

        return rgb_image, gt_image

