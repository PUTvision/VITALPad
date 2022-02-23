import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.measure import label, regionprops

from landing_pad_matcher.datasets.density import DensityDataset
from landing_pad_matcher.models.density_estimator import DensityEstimator

torch.set_grad_enabled(False)

estimator = DensityEstimator.load_from_checkpoint('/home/rivi/Downloads/estimator.ckpt', strict=False)
estimator.eval()

data_path = Path('/home/rivi/Datasets/Landing')
file_ids = sorted([path.stem.split('_')[1] for path in (data_path / 'rgb').iterdir()])
dataset = DensityDataset(data_path=data_path, allowed_file_ids=file_ids)

for i in range(len(dataset)):
    image, gt = dataset[i]
    image = image[None, ...]
    result = torch.softmax(estimator(image).squeeze(), dim=0).numpy()

    # people = np.where(result < 0, np.abs(result), 0)
    # landing_pad = np.where(result > 0, result, 0)
    people = result.squeeze()[1]
    landing_pad = result.squeeze()[2]

    image = image.squeeze().permute(1, 2, 0).numpy()
    thresholded_landing_pad = (landing_pad > 0.5).astype(np.uint8)

    blobs = label(thresholded_landing_pad)
    blobs = regionprops(blobs, intensity_image=landing_pad)
    blobs = sorted(blobs, key=lambda blob: blob.area)
    if blobs:
        biggest_blob = blobs[0]
        if biggest_blob.intensity_mean < 0.8:
            continue

        pad_image = image[biggest_blob.slice]
        cv2.imshow('pad image', pad_image)

    end = time.perf_counter()

    cv2.imshow('image', image)
    cv2.imshow('people', people)
    cv2.imshow('landing pad', landing_pad)
    cv2.waitKey()
