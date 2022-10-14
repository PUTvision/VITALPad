from typing import Dict, Any

import torch
from timm import create_model
from torch import nn


class FOMO(nn.Module):
    def __init__(self, encoder_name: str, classes: int, encoder_kwargs: Dict[str, Any] = {}):
        super().__init__()

        self.network = create_model(encoder_name, pretrained=True, num_classes=0,
                                    global_pool='', **encoder_kwargs)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=classes, kernel_size=(1, 1)),
            nn.UpsamplingNearest2d(scale_factor=32),
        )

    def forward(self, inputs: torch.Tensor):
        inputs = self.network(inputs)
        return self.head(inputs)
