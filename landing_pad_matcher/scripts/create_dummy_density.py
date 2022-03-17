import torch.onnx
from segmentation_models_pytorch import DeepLabV3Plus, Unet, MAnet, Linknet, EfficientUnetPlusPlus
from segmentation_models_pytorch.decoders.pyunet.model import PyUnet
from torch import nn

model = PyUnet('tu-lcnet_050', classes=3, encoder_kwargs={'act_layer': nn.ReLU})
inputs = torch.rand(1, 3, 480, 640)

torch.onnx.export(model.eval(), inputs, 'density_estimator.onnx', opset_version=15, input_names=['input'],
                  output_names=['output'])
