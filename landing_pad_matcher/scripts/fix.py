from pathlib import Path

import cv2
import numpy as np

from landing_pad_matcher.datamodules.density import DensityDataModule

paths = DensityDataModule.get_paths(Path('/home/rivi/Datasets/Landing'))
for _, gt_path in paths:
    gt = cv2.imread(str(gt_path))
    gt[gt[..., 0] == 4] = np.array([0, 255, 0])

    cv2.imwrite(str(gt_path), gt)
