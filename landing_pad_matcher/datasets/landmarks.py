from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch.utils
import torch.utils.data
from albumentations import Compose, Affine, KeypointParams, Perspective
from albumentations.augmentations.transforms import (
    RandomGamma, ColorJitter, ToFloat, MotionBlur, ISONoise, RandomShadow
)
from albumentations.pytorch import ToTensorV2


class LandmarksDataset(torch.utils.data.Dataset):
    def __init__(self, image_path: Path, num_samples: int):
        self._image_path = image_path
        self._num_samples = num_samples

        self._augmentations = Compose([
            RandomGamma(gamma_limit=(80, 120)),
            ColorJitter(brightness=0.2, contrast=0.2, hue=0.1, saturation=0.5),
            Affine(rotate=(-180, 180), translate_px=(-20, 20), scale=(0.7, 1.1), shear=(0.7, 1.3),
                   cval=(112, 112, 112), p=1.0),
            Perspective(pad_val=(112, 112, 112)),
            RandomShadow(),
            MotionBlur(),
            ISONoise(),
            ToFloat(max_value=255.0),
            ToTensorV2()
        ], keypoint_params=KeypointParams(format='xy', remove_invisible=True))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, keypoints = self._load_data()
        image, keypoints = self._perform_augmentations(image, keypoints)

        return image, keypoints

    def __len__(self) -> int:
        return self._num_samples

    def _perform_augmentations(self, image: np.ndarray, keypoints: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        transformed = self._augmentations(image=image, keypoints=keypoints)
        image = transformed['image']
        keypoints = transformed['keypoints']

        targets = torch.zeros(7)
        if len(keypoints) == 3:
            targets[0] = 1
            targets[1:] = torch.tensor(keypoints).flatten() / 128

        return image, targets

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(str(self._image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = np.array([
            [63.5, 15],
            [106, 87],
            [21, 87]
        ])

        return image, keypoints
