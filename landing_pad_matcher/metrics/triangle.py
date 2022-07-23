import torch
import torch.nn.functional

from torchmetrics import Metric


class TriangleError(Metric):
    def __init__(self):
        super().__init__()

        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        a = torch.linalg.norm(preds[..., 2:4] - preds[..., 4:6])
        b = torch.linalg.norm(preds[..., 4:6] - preds[..., 6:8])
        c = torch.linalg.norm(preds[..., 6:8] - preds[..., 2:4])

        errors = torch.abs(a - c) + torch.abs(1.0201011079204643 - b / a) + torch.abs(1.0201011079204643 - c / a)

        self.sum += torch.mean(errors)
        self.count += 1

    def compute(self):
        return self.sum / self.count
