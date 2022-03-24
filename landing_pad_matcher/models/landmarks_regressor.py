import pytorch_lightning as pl
import timm
import torch.nn.functional
import torchmetrics
from torch import nn
from torchmetrics import MetricCollection


class LandmarksRegressor(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.network = timm.create_model('lcnet_050', pretrained=True, num_classes=10, act_layer=nn.ReLU)

        self.loss = nn.GaussianNLLLoss(reduction='none')

        classification_metrics = MetricCollection([
            torchmetrics.Accuracy(),
        ])
        regression_metrics = MetricCollection([
            torchmetrics.MeanAbsoluteError(),
            torchmetrics.MeanAbsolutePercentageError(),
            torchmetrics.MeanSquaredError(),
        ])

        self.train_classification_metrics = classification_metrics.clone('train_')
        self.valid_classification_metrics = classification_metrics.clone('val_')
        self.test_classification_metrics = classification_metrics.clone('test_')
        self.train_regression_metrics = regression_metrics.clone('train_')
        self.valid_regression_metrics = regression_metrics.clone('val_')
        self.test_regression_metrics = regression_metrics.clone('test_')

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_pred = self.forward(x)

        is_object = y[:, 0]

        loss = torch.mean(torch.binary_cross_entropy_with_logits(y_pred[:, 0], is_object))
        loss += 5 * torch.mean(is_object * self.loss(y_pred[:, 2:], y[:, 1:],
                                                     nn.functional.softplus(y_pred[:, 1])).mean(dim=1))

        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(self.train_classification_metrics(torch.sigmoid(y_pred[:, 0]), y[:, 0].long()))
        self.log_dict(self.train_regression_metrics(y_pred[:, 2:], y[:, 1:]))

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y = batch
        y_pred = self.forward(x)

        is_object = y[:, 0]

        loss = torch.mean(torch.binary_cross_entropy_with_logits(y_pred[:, 0], is_object))
        loss += 5 * torch.mean(is_object * self.loss(y_pred[:, 2:], y[:, 1:],
                                                     nn.functional.softplus(y_pred[:, 1])).mean(dim=1))

        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(self.valid_classification_metrics(torch.sigmoid(y_pred[:, 0]), y[:, 0].long()))
        self.log_dict(self.valid_regression_metrics(y_pred[:, 2:], y[:, 1:]))

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y = batch
        y_pred = self.forward(x)

        is_object = y[:, 0]

        loss = torch.mean(torch.binary_cross_entropy_with_logits(y_pred[:, 0], is_object))
        loss += 5 * torch.mean(is_object * self.loss(y_pred[:, 2:], y[:, 1:],
                                                     nn.functional.softplus(y_pred[:, 1])).mean(dim=1))

        self.log('test_loss', loss, sync_dist=True)
        self.log_dict(self.test_classification_metrics(torch.sigmoid(y_pred[:, 0]), y[:, 0].long()))
        self.log_dict(self.test_regression_metrics(y_pred[:, 2:], y[:, 1:]))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                                          patience=3, min_lr=1e-6, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': reduce_lr_on_plateau,
            'monitor': 'train_loss_epoch'
        }
