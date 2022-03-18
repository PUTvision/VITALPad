from typing import Tuple

import torch
from torch import nn

from landing_pad_matcher.models.landmarks_regressor import LandmarksRegressor
from landing_pad_matcher.models.density_estimator import DensityEstimator


class LandmarksRegressorWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        regressor = LandmarksRegressor.load_from_checkpoint('keypoints_detector.ckpt', strict=False)

        self.network = regressor.network
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.network(x)
        return self.sigmoid(x[:, 2]), x[:, 0], self.softplus(x[:, 1]), x[:, 2:], self.softplus(x[:, 1])


class DensityEstimatorWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        estimator = DensityEstimator.load_from_checkpoint('density_estimator.ckpt')

        self.network = estimator.network
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.softmax(self.network(x))


model = LandmarksRegressorWrapper().eval()
inputs = torch.rand(1, 3, 128, 128)
torch.onnx.export(model, inputs, 'keypoints_detector.onnx', opset_version=15, input_names=['input'],
                  output_names=['confidence', 'rotation', 'rotation_std', 'coords', 'coords_std'])

model = DensityEstimatorWrapper().eval()
inputs = torch.rand(1, 3, 480, 640)
torch.onnx.export(model, inputs, 'density_estimator.onnx', opset_version=15, input_names=['input'],
                  output_names=['output'])
