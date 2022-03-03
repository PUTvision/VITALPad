from pathlib import Path
from typing import Tuple

import click
import cv2
import torch
from albumentations import Compose, ToFloat, CenterCrop, LongestMaxSize, PadIfNeeded, SmallestMaxSize
from albumentations.pytorch import ToTensorV2

from landing_pad_matcher.models.density_estimator import DensityEstimator


@click.command()
@click.argument('video-path', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument('model-path', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--target-resolution', type=click.Tuple((int, int)))
def run_density_estimation(video_path: Path, model_path: Path, target_resolution: Tuple[int, int]):
    torch.set_grad_enabled(False)

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError('Error reading video')

    transforms = Compose([
        SmallestMaxSize(max_size=target_resolution[1]),
        CenterCrop(height=target_resolution[1], width=target_resolution[0]),
        ToFloat(max_value=255.0),
        ToTensorV2()
    ])

    density_estimator = DensityEstimator.load_from_checkpoint(str(model_path)).eval()

    while True:
        status, frame = video.read()
        if not status:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transforms(image=frame)['image'][None, ...]
        bgr_frame = cv2.cvtColor(frame.squeeze().permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)

        result = torch.softmax(density_estimator(frame).squeeze(), dim=0).numpy()
        cv2.imshow('frame', bgr_frame)
        cv2.imshow('people', result[1])
        cv2.imshow('landing pad', result[2])
        cv2.waitKey(1)


if __name__ == '__main__':
    run_density_estimation()
