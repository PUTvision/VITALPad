import torch
from torch import nn


class BetaNLLLoss(nn.Module):
    def __init__(self, beta: float):
        super().__init__()

        self._beta = beta

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        var = var.unsqueeze(-1)
        return (0.5 * ((targets - inputs) ** 2 / var + var.log()) * var.detach() ** self._beta).sum(dim=-1)
