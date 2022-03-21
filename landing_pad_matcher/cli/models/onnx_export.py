from pathlib import Path
from typing import Tuple

import click
import torch
from torch import nn

from landing_pad_matcher.models.landmarks_regressor import LandmarksRegressor
from landing_pad_matcher.models.density_estimator import DensityEstimator


class LandmarksRegressorWrapper(nn.Module):
    def __init__(self, checkpoint_path: Path):
        super().__init__()

        regressor = LandmarksRegressor.load_from_checkpoint(checkpoint_path, strict=False)

        self.network = regressor.network
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.network(x)
        return self.sigmoid(x[:, 0]), self.softplus(x[:, 1]), x[:, 2:]


class DensityEstimatorWrapper(nn.Module):
    def __init__(self, checkpoint_path: Path):
        super().__init__()

        estimator = DensityEstimator.load_from_checkpoint(checkpoint_path, strict=False)

        self.network = estimator.network
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.softmax(self.network(x))


@click.command()
@click.argument('keypoints-detector-path', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument('density-estimator-path', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument('output-dir', type=click.Path(file_okay=False, path_type=Path))
def export(keypoints_detector_path: Path, density_estimator_path: Path, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)

    model = LandmarksRegressorWrapper(keypoints_detector_path).eval()
    inputs = torch.rand(1, 3, 128, 128)
    torch.onnx.export(model, inputs, output_dir / 'keypoints_detector.onnx', opset_version=15, input_names=['input'],
                      output_names=['confidence', 'std', 'coords'])

    model = DensityEstimatorWrapper(density_estimator_path).eval()
    inputs = torch.rand(1, 3, 480, 640)
    torch.onnx.export(model, inputs, output_dir / 'density_estimator.onnx', opset_version=15, input_names=['input'],
                      output_names=['output'])


if __name__ == '__main__':
    export()
