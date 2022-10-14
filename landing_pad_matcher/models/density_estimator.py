import pytorch_lightning as pl
import torch.nn.functional
import torchmetrics
from segmentation_models_pytorch import DeepLabV3Plus, Unet
from torch import nn
from torchmetrics import MetricCollection

from landing_pad_matcher.models.architectures.fomo import FOMO


class DensityEstimator(pl.LightningModule):
    def __init__(self, model_name: str, encoder_name: str, lr: float, classes_weights: torch.Tensor, **kwargs):
        super().__init__()

        encoder_kwargs = {'act_layer': nn.ReLU6} if 'lcnet' in encoder_name else {}

        match model_name:
            case 'UNet':
                self.network = Unet(encoder_name, classes=2, encoder_kwargs=encoder_kwargs)
            case 'DeepLabV3Plus':
                self.network = DeepLabV3Plus(encoder_name, classes=2, encoder_kwargs=encoder_kwargs)
            case 'FOMO':
                self.network = FOMO(encoder_name, classes=3, encoder_kwargs=encoder_kwargs)
            case _:
                raise RuntimeError(f'Unknown model: {model_name}')

        self.loss = nn.CrossEntropyLoss(weight=classes_weights)

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
        self.log_dict(self.train_metrics(torch.softmax(y_pred, dim=1), y), sync_dist=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(self.valid_metrics(torch.softmax(y_pred, dim=1), y), sync_dist=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        self.log('test_loss', loss, sync_dist=True)
        self.log_dict(self.test_metrics(torch.softmax(y_pred, dim=1), y), sync_dist=True)

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
