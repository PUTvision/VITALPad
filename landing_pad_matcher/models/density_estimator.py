import pytorch_lightning as pl
import torch.nn.functional
import torchmetrics
from segmentation_models_pytorch import DeepLabV3Plus
from torch import nn
from torchmetrics import MetricCollection


class DensityEstimator(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.network = DeepLabV3Plus('tu-lcnet_075', classes=3)

        self.loss = nn.CrossEntropyLoss()

        metrics = MetricCollection([
            torchmetrics.MeanSquaredError(),
            torchmetrics.MeanAbsoluteError(),
            torchmetrics.MeanAbsolutePercentageError()
        ])
        self.train_metrics = metrics.clone('train_')
        self.valid_metrics = metrics.clone('val_')
        self.test_metrics = metrics.clone('test_')

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(self.train_metrics(torch.softmax(y_pred, dim=1), y))

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(self.valid_metrics(torch.softmax(y_pred, dim=1), y))

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        self.log('test_loss', loss, sync_dist=True)
        self.log_dict(self.test_metrics(torch.softmax(y_pred, dim=1), y), on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                                          patience=3, min_lr=1e-6, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': reduce_lr_on_plateau,
            'monitor': 'train_loss_epoch'
        }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)
