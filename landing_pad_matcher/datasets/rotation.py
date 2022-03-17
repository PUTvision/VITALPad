import math
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch.utils
import torch.utils.data
from albumentations import Compose, Affine, Perspective, Resize, RandomCrop
from albumentations.augmentations.transforms import (
    RandomGamma, ColorJitter, ToFloat, MotionBlur, ISONoise, RandomShadow
)
from albumentations.pytorch import ToTensorV2


class RotationDataset(torch.utils.data.Dataset):
    def __init__(self, image_path: Path, num_samples: int):
        self._image = self._load_data(image_path)
        self._num_samples = num_samples

        self._pre_rotation_augmentations = Compose([
            RandomShadow(),
            MotionBlur(),
            Resize(height=141, width=141, interpolation=cv2.INTER_AREA),
            RandomGamma(gamma_limit=(80, 120)),
            ColorJitter(brightness=0.2, contrast=0.2, hue=0.1, saturation=0.5),
        ])
        self._post_rotation_augmentations = Compose([
            Affine(translate_px=(-20, 20), scale=(0.7, 1.1), shear=(0.7, 1.3), cval=(120, 120, 120)),
            Perspective(scale=(0.01, 0.05), pad_val=(120, 120, 120)),
            ISONoise(),
            RandomCrop(height=128, width=128),
            ToFloat(max_value=255.0),
            ToTensorV2()
        ])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, rotation = self._perform_augmentations(self._image)
        return image, rotation

    def __len__(self) -> int:
        return self._num_samples

    def _perform_augmentations(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self._pre_rotation_augmentations(image=image)['image']

        random_rotation = random.uniform(0, 359.999)
        image = rotate(image, random_rotation)

        image = self._post_rotation_augmentations(image=image)['image']

        return image, torch.tensor([math.radians(random_rotation)], dtype=torch.float32)

    @staticmethod
    def _load_data(image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image


def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(120, 120, 120))

    # return the rotated image
    return rotated
