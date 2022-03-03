import torch

from landing_pad_matcher.models.landmarks_regressor import LandmarksRegressor
from landing_pad_matcher.models.density_estimator import DensityEstimator

model = LandmarksRegressor.load_from_checkpoint('keypoints_detector.ckpt')
inputs = torch.rand(1, 3, 128, 128)
torch.onnx.export(model, inputs, 'keypoints_detector.onnx', opset_version=14, input_names=['input'],
                  output_names=['output'])

model = DensityEstimator.load_from_checkpoint('density_estimator.ckpt')
inputs = torch.rand(1, 3, 640, 480)
torch.onnx.export(model, inputs, 'density_estimator.onnx', opset_version=14, input_names=['input'],
                  output_names=['output'])
