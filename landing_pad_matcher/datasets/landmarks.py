import random
from pathlib import Path
from typing import Tuple, Dict, Any

import cv2
import numpy as np
import torch.utils
import torch.utils.data
from albumentations import Compose, Affine, KeypointParams, Perspective, Resize, RandomCrop, ImageOnlyTransform, \
    RandomScale
from albumentations.augmentations.transforms import (
    RandomGamma, ColorJitter, ToFloat, MotionBlur, ISONoise, RandomShadow, RandomSunFlare, Flip
)
from albumentations.pytorch import ToTensorV2


class LandmarksDataset(torch.utils.data.Dataset):
    def __init__(self, image_path: Path, textures_path: Path, num_samples: int):
        self._image, self._keypoints = self._load_data(image_path)
        self._num_samples = num_samples

        self._augmentations = Compose([
            Resize(height=141, width=141, interpolation=cv2.INTER_AREA),
            RandomGamma(gamma_limit=(80, 120)),
            ColorJitter(brightness=0.2, contrast=0.2, hue=0.1, saturation=0.5),
            Affine(rotate=(-180, 180), translate_px=(-20, 20), scale=(0.4, 1.1), shear=(0.7, 1.3),
                   cval=(0, 0, 0), p=1.0, interpolation=cv2.INTER_AREA),
            RandomBackground(textures_path=textures_path),
            RandomShadow(),
            MotionBlur(),
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
        transformed = self._augmentations(image=image, keypoints=keypoints)
        image = transformed['image']
        keypoints = transformed['keypoints']

        targets = torch.zeros(9)
        if len(keypoints) == 4:
            targets[0] = 1
            targets[1:] = torch.tensor(keypoints).flatten() / 128

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


class RandomBackground(ImageOnlyTransform):
    def __init__(self, textures_path: Path, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)

        self._textures = [
            cv2.cvtColor(cv2.imread(str(texture_path)), cv2.COLOR_BGR2RGB)
            for texture_path in textures_path.iterdir()
         ]
        self._augmentations = Compose([
            RandomScale((0.15, 1.0), interpolation=cv2.INTER_AREA, p=1.0),
            ColorJitter(),
            Flip(),
            RandomCrop(height=141, width=141)
        ])

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        pass

    def apply(self, img, **params) -> np.ndarray:
        if random.random() >= 0.5:
            texture = random.choice(self._textures)
            texture = self._augmentations(image=texture)['image']
            return np.where(img == (0, 0, 0), texture, img)

        return np.where(img == (0, 0, 0), np.random.randint(0, 255, size=(3,), dtype=np.uint8), img)
