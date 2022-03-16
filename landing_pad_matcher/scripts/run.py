from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch.nn.functional
from torch import nn

from landing_pad_matcher.datasets.landmarks import LandmarksDataset
from landing_pad_matcher.models.landmarks_regressor import LandmarksRegressor

torch.set_grad_enabled(False)

model = LandmarksRegressor.load_from_checkpoint('../keypoints_detector.ckpt')
model.eval()
dataset = LandmarksDataset(image_path=Path('../data/pad_128.png'), num_samples=100)

stds = []
maes = []
for i in range(len(dataset)):
    image, gt_ = dataset[i]

    gt = gt_.tolist()
    preds_ = model(image[None, ...]).squeeze()
    preds = preds_.tolist()

    gt_image = image.permute(1, 2, 0).numpy()[..., ::-1].copy()
    cv2.circle(gt_image, (round(gt[1] * 128), round(gt[2] * 128)), radius=5, color=(0, 0, 255))
    cv2.circle(gt_image, (round(gt[3] * 128), round(gt[4] * 128)), radius=5, color=(0, 255, 0))
    cv2.circle(gt_image, (round(gt[5] * 128), round(gt[6] * 128)), radius=5, color=(255, 0, 0))
    cv2.circle(gt_image, (round(gt[7] * 128), round(gt[8] * 128)), radius=5, color=(255, 255, 0))

    preds_image = image.permute(1, 2, 0).numpy()[..., ::-1].copy()
    cv2.circle(preds_image, (round(preds[2] * 128), round(preds[3] * 128)), radius=5, color=(0, 0, 255))
    cv2.circle(preds_image, (round(preds[4] * 128), round(preds[5] * 128)), radius=5, color=(0, 255, 0))
    cv2.circle(preds_image, (round(preds[6] * 128), round(preds[7] * 128)), radius=5, color=(255, 0, 0))
    cv2.circle(preds_image, (round(preds[8] * 128), round(preds[9] * 128)), radius=5, color=(255, 255, 0))

    std = nn.functional.softplus(preds_[1]).item()
    mae = torch.mean(torch.absolute(preds_[2:] - gt_[1:])).item()

    if gt[0]:
        stds.append(std)
        maes.append(mae / 100)

    print(torch.sigmoid(preds_[0]), std)

    cv2.imshow('gt', gt_image)
    cv2.imshow('pred', preds_image)
    cv2.waitKey()

plt.plot(stds)
plt.plot(maes)
plt.show()
