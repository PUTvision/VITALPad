from pathlib import Path

import cv2
import numpy as np
import torch

from landing_pad_matcher.datasets.rotation import RotationDataset

torch.set_grad_enabled(False)

dataset = RotationDataset(Path('../data/pad.png'), 5000)
for image, rotation in dataset:
    image *= 255
    image = image.permute(1, 2, 0).numpy().astype(np.uint8)[..., ::-1]
    print(rotation * 359)
    cv2.imshow('image', image)
    cv2.waitKey()
