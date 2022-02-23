import os
from pathlib import Path
from typing import Optional

import click
import pytorch_lightning as pl
import pytorch_lightning.loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary, QuantizationAwareTraining
from pytorch_lightning.plugins import DDPPlugin

from landing_pad_matcher.datamodules.density import DensityDataModule
from landing_pad_matcher.models.density_estimator import DensityEstimator


@click.command()
@click.option('--data-path', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--batch-size', type=int, default=256)
@click.option('--validation-batch-size', type=int, default=256)
@click.option('--epochs', type=int, default=1000)
@click.option('--lr', type=float, default=1e-3)
@click.option('--number-of-workers', type=int, default=os.getenv('WORKERS', 4))
def train(data_path: Path, batch_size: int, validation_batch_size: int,
          epochs: int, lr: float, number_of_workers: Optional[int]):
    pl.seed_everything(42)

    data_module = DensityDataModule(data_path, batch_size, validation_batch_size, number_of_workers)

    model = DensityEstimator(lr=lr)

    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.5f}', monitor='val_loss', verbose=True)
    early_stop_callback = EarlyStopping(monitor='train_loss', patience=50)
    model_summary_callback = ModelSummary(max_depth=-1)
    # quantization_callback = QuantizationAwareTraining(qconfig='qnnpack')
    logger = pl.loggers.NeptuneLogger(project='Vision/LandingDensityEstimation')

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[model_summary_callback, checkpoint_callback, early_stop_callback],
        gpus=-1,
        strategy=DDPPlugin(
            find_unused_parameters=False,
        ),
        sync_batchnorm=True,
        precision=32,
        max_epochs=epochs,
        benchmark=True,
        accumulate_grad_batches=1
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(ckpt_path='best', datamodule=data_module)


if __name__ == '__main__':
    train()
