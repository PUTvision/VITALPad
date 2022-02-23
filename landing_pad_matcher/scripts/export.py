import torch

from landing_pad_matcher.models.landmarks_regressor import LandmarksRegressor

# model = LandmarksRegressor.load_from_checkpoint('model.ckpt')
model = LandmarksRegressor(lr=1e-3)
model.eval()

inputs = torch.rand(1, 3, 128, 128)
torch.onnx.export(model, inputs, 'model.onnx', opset_version=14, input_names=['input'], output_names=['output'])
