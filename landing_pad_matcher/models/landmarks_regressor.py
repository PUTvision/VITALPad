import pytorch_lightning as pl
import timm
import torch.nn.functional
import torchmetrics
from torch import nn
from torchmetrics import MetricCollection

from landing_pad_matcher.losses.betanll import BetaNLLLoss
from landing_pad_matcher.metrics.triangle import TriangleError


class LandmarksRegressor(pl.LightningModule):
    def __init__(self, model_name: str, **kwargs):
        super().__init__()

        self.network = timm.create_model(model_name, pretrained=True, num_classes=32)

        self.loss = nn.GaussianNLLLoss(reduction='none')
        # self.loss = BetaNLLLoss(beta=0.5)

        classification_metrics = MetricCollection([
            torchmetrics.Accuracy(multiclass=False),
        ])
        regression_metrics = MetricCollection([
            torchmetrics.MeanAbsoluteError(),
            torchmetrics.MeanSquaredError(),
        ])

        self.train_classification_metrics = classification_metrics.clone('train_')
        self.valid_classification_metrics = classification_metrics.clone('val_')
        self.test_classification_metrics = classification_metrics.clone('test_')
        self.train_regression_metrics = regression_metrics.clone('train_')
        self.valid_regression_metrics = regression_metrics.clone('val_')
        self.test_regression_metrics = regression_metrics.clone('test_')
        self.train_var_metrics = regression_metrics.clone('train_Var')
        self.valid_var_metrics = regression_metrics.clone('val_Var')
        self.test_var_metrics = regression_metrics.clone('test_Var')

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, is_object, y = batch
        y_pred = self.forward(x)
        y_pred_var = nn.functional.softplus(y_pred[:, 8:16])

        loss = torch.mean(torch.binary_cross_entropy_with_logits(y_pred[:, :8], is_object))
        loss += 5 * torch.mean(is_object * self.loss(y_pred[:, 16:].view(-1, 8, 2), y, y_pred_var).mean(dim=-1))
        # loss += 5 * torch.mean(is_object * self.loss(y_pred[:, 16:].view(-1, 8, 2), y, y_pred_var))

        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(self.train_classification_metrics(torch.sigmoid(y_pred[:, :8]), is_object.long()))
        self.log_dict(self.train_regression_metrics(y_pred[:, 16:].view(-1, 8, 2), y))
        self.log_dict(self.train_var_metrics(y_pred_var.sqrt(),
                                             torch.abs(y_pred[:, 16:].view(-1, 8, 2) - y).mean(dim=-1)))

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, is_object, y = batch
        y_pred = self.forward(x)
        y_pred_var = nn.functional.softplus(y_pred[:, 8:16])

        loss = torch.mean(torch.binary_cross_entropy_with_logits(y_pred[:, :8], is_object))
        loss += 5 * torch.mean(is_object * self.loss(y_pred[:, 16:].view(-1, 8, 2), y, y_pred_var).mean(dim=-1))
        # loss += 5 * torch.mean(is_object * self.loss(y_pred[:, 16:].view(-1, 8, 2), y, y_pred_var))

        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(self.valid_classification_metrics(torch.sigmoid(y_pred[:, :8]), is_object.long()))
        self.log_dict(self.valid_regression_metrics(y_pred[:, 16:].view(-1, 8, 2), y))
        self.log_dict(self.train_var_metrics(y_pred_var.sqrt(),
                                             torch.abs(y_pred[:, 16:].view(-1, 8, 2) - y).mean(dim=-1)))

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, is_object, y = batch
        y_pred = self.forward(x)
        y_pred_var = nn.functional.softplus(y_pred[:, 8:16])

        loss = torch.mean(torch.binary_cross_entropy_with_logits(y_pred[:, :8], is_object))
        loss += 5 * torch.mean(is_object * self.loss(y_pred[:, 16:].view(-1, 8, 2), y, y_pred_var).mean(dim=-1))
        # loss += 5 * torch.mean(is_object * self.loss(y_pred[:, 16:].view(-1, 8, 2), y, y_pred_var))

        self.log('test_loss', loss, sync_dist=True)
        self.log_dict(self.test_classification_metrics(torch.sigmoid(y_pred[:, :8]), is_object.long()))
        self.log_dict(self.test_regression_metrics(y_pred[:, 16:].view(-1, 8, 2), y))
        self.log_dict(self.train_var_metrics(y_pred_var.sqrt(),
                                             torch.abs(y_pred[:, 16:].view(-1, 8, 2) - y).mean(dim=-1)))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                                          patience=3, min_lr=1e-6, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': reduce_lr_on_plateau,
            'monitor': 'train_loss_epoch'
        }
