import timm
import torch.onnx
from torch import nn

model = timm.create_model('lcnet_075', pretrained=True, num_classes=10, act_layer=nn.ReLU6)

inputs = torch.rand(1, 3, 128, 128)

torch.onnx.export(model.eval(), inputs, 'keypoints_detector.onnx', opset_version=15, input_names=['input'],
                  output_names=['output'])
