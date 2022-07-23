import json
import random
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import cv2
import numpy as np
import torch.utils
import torch.utils.data
from albumentations import Compose, Affine, KeypointParams, Perspective, Resize, RandomCrop, ImageOnlyTransform, \
    RandomScale
from albumentations.augmentations.transforms import (
    RandomGamma, ColorJitter, ToFloat, MotionBlur, ISONoise, RandomShadow, Flip, ImageCompression
)
from albumentations.pytorch import ToTensorV2


class LandmarksDataset(torch.utils.data.Dataset):
    def __init__(self, image_path: Path, textures_path: Path, photos_path: Optional[Path], num_samples: int):
        self._image, self._keypoints = self._load_data(image_path)
        self._num_samples = num_samples
        if photos_path is not None:
            self._photos_paths, self._photos_labels = self._load_photos(photos_path)
        else:
            self._photos_paths, self._photos_labels = None, None

        self._augmentations = Compose([
            ImageCompression(quality_lower=80),
            RandomScale(scale_limit=(-0.98, -0.75), interpolation=cv2.INTER_AREA, p=0.75),
            Resize(height=141, width=141, interpolation=cv2.INTER_AREA),
            RandomGamma(gamma_limit=(70, 130)),
            ColorJitter(brightness=0.8, contrast=0.5, hue=0.3, saturation=0.7),
            Affine(rotate=(-180, 180), translate_px=(-20, 20), scale=(0.4, 1.1), shear=(0.7, 1.3),
                   cval=(0, 0, 0), p=1.0, interpolation=cv2.INTER_AREA),
            RandomBackground(textures_path=textures_path),
            RandomShadow(),
            MotionBlur(),
            Perspective(scale=(0.01, 0.05), pad_val=(0, 0, 0)),
            ISONoise(),
            RandomCrop(height=128, width=128),
            ToFloat(max_value=255.0),
            ToTensorV2()
        ], keypoint_params=KeypointParams(format='xy', remove_invisible=False))
        self._photos_augmentations = Compose([
            RandomGamma(gamma_limit=(70, 130)),
            ColorJitter(brightness=0.2, contrast=0.2, hue=0.2, saturation=0.2),
            Affine(rotate=(-180, 180), mode=cv2.BORDER_REPLICATE),
            RandomShadow(),
            MotionBlur(),
            ISONoise(),
            ToFloat(max_value=255.0),
            ToTensorV2()
        ], keypoint_params=KeypointParams(format='xy', remove_invisible=False))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._photos_paths is not None and random.random() <= 0.1:
            index = random.randrange(0, len(self._photos_paths))
            image = cv2.cvtColor(cv2.imread(str(self._photos_paths[index])), cv2.COLOR_BGR2RGB)
            keypoints = self._photos_labels[index]
            image, is_object, keypoints = self._perform_augmentations(image, keypoints, self._photos_augmentations)
        else:
            image, is_object, keypoints = self._perform_augmentations(self._image, self._keypoints, self._augmentations)

        return image, is_object, keypoints

    def __len__(self) -> int:
        return self._num_samples

    @staticmethod
    def _perform_augmentations(image: np.ndarray, keypoints: np.ndarray,
                               augmentations: Compose) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transformed = augmentations(image=image, keypoints=keypoints)
        image = transformed['image']
        keypoints = transformed['keypoints']

        is_object = torch.zeros(8)
        for i, (x, y) in enumerate(keypoints):
            is_object[i] = 1 if 0 <= x <= 128 and 0 <= y <= 128 else 0

        targets = torch.tensor(keypoints) / 128

        return image, is_object, targets

    @staticmethod
    def _load_data(image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = np.array([
            [274.5, 274.5],
            [248, 263],
            [301, 263],
            [301, 283],
            [248, 283],
            [274.5, 82.5],
            [443.5, 367.5],
            [105.5, 367.5]
        ], dtype=np.float64)

        return image, keypoints

    @staticmethod
    def _load_photos(photos_path: Path) -> Tuple[List[Path], List[np.ndarray]]:
        with (photos_path / 'keypoints.json').open() as file:
            labels = json.load(file)

        photos_paths = []
        photos_labels = []
        for photo_name, keypoints in labels.items():
            photos_paths.append(photos_path / 'images' / photo_name)
            photos_labels.append(np.array(keypoints, dtype=np.float64))

        return photos_paths, photos_labels


class RandomBackground(ImageOnlyTransform):
    def __init__(self, textures_path: Path, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)

        self._textures = [
            cv2.cvtColor(cv2.imread(str(texture_path)), cv2.COLOR_BGR2RGB)
            for texture_path in textures_path.iterdir()
         ]
        self._augmentations = Compose([
            RandomScale((0.05, 1.0), interpolation=cv2.INTER_AREA, p=1.0),
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
