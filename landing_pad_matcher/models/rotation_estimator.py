import math

import pytorch_lightning as pl
import timm
import torch.nn.functional
import torchmetrics
from torch import nn
from torchmetrics import MetricCollection


class RotationEstimator(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.network = timm.create_model('lcnet_050', pretrained=True, num_classes=1, act_layer=nn.ReLU6)

        self.loss = nn.MSELoss()

        metrics = MetricCollection([
            torchmetrics.MeanAbsoluteError(),
            torchmetrics.MeanSquaredError()
        ])

        self.train_metrics = metrics.clone('train_')
        self.valid_metrics = metrics.clone('val_')
        self.test_metrics = metrics.clone('test_')

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp_(self.network(x), 0, 2 * math.pi)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_pred = self.forward(x)

        loss = self.loss(y_pred, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(self.train_metrics(y_pred, y))

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y = batch
        y_pred = self.forward(x)

        loss = self.loss(y_pred, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(self.valid_metrics(y_pred, y))

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y = batch
        y_pred = self.forward(x)

        loss = self.loss(y_pred, y)

        self.log('test_loss', loss, sync_dist=True)
        self.log_dict(self.test_metrics(y_pred, y))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                                          patience=3, min_lr=1e-6, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': reduce_lr_on_plateau,
            'monitor': 'train_loss_epoch'
        }
