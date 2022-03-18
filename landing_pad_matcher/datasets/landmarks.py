import math
from pathlib import Path
from typing import Tuple

import albumentations
import cv2
import random
import numpy as np
import torch.utils
import torch.utils.data
from albumentations import Compose, Affine, KeypointParams, Perspective, Resize, RandomCrop
from albumentations.augmentations.transforms import (
    RandomGamma, ColorJitter, ToFloat, MotionBlur, ISONoise, RandomShadow
)
from albumentations.pytorch import ToTensorV2

from landing_pad_matcher.datasets.rotation import rotate


class LandmarksDataset(torch.utils.data.Dataset):
    def __init__(self, image_path: Path, num_samples: int):
        self._image, self._keypoints = self._load_data(image_path)
        self._num_samples = num_samples

        self._pre_rotation_augmentations = Compose([
            RandomShadow(),
            MotionBlur(),
            Resize(height=141, width=141, interpolation=cv2.INTER_AREA),
            RandomGamma(gamma_limit=(80, 120)),
            ColorJitter(brightness=0.2, contrast=0.2, hue=0.1, saturation=0.5),
        ], keypoint_params=KeypointParams(format='xy', remove_invisible=True))
        self._post_rotation_augmentations = Compose([
            Affine(translate_px=(-20, 20), scale=(0.7, 1.1), shear=(0.7, 1.3), cval=(120, 120, 120)),
            Perspective(scale=(0.01, 0.05), pad_val=(120, 120, 120)),
            ISONoise(),
            RandomCrop(height=128, width=128),
            ToFloat(max_value=255.0),
            ToTensorV2()
        ], keypoint_params=KeypointParams(format='xy', remove_invisible=True))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, keypoints = self._perform_augmentations(self._image, self._keypoints)
        return image, keypoints

    def __len__(self) -> int:
        return self._num_samples

    def _perform_augmentations(self, image: np.ndarray, keypoints: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        transformed = self._pre_rotation_augmentations(image=image, keypoints=keypoints)
        image = transformed['image']
        keypoints = transformed['keypoints']

        random_rotation = random.uniform(0, 359.999)
        image = albumentations.rotate(image, random_rotation, border_mode=cv2.BORDER_CONSTANT, value=(120, 120, 120))
        keypoints = [albumentations.keypoint_rotate([*keypoint, 0, 1], angle=random_rotation, rows=141, cols=141)[:2]
                     for keypoint in keypoints]

        transformed = self._post_rotation_augmentations(image=image, keypoints=keypoints)
        image = transformed['image']
        keypoints = transformed['keypoints']

        targets = torch.zeros(10)
        targets[0] = math.radians(random_rotation)
        if len(keypoints) == 4:
            targets[1] = 1
            targets[2:] = torch.tensor(keypoints).flatten() / 128

        return image, targets

    @staticmethod
    def _load_data(image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = np.array([
            [274.5, 274.5],
            [274.5, 82.5],
            [443.5, 367.5],
            [105.5, 367.5]
        ])

        return image, keypoints
