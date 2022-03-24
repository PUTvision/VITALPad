import os
from pathlib import Path
from typing import Optional

import click
import pytorch_lightning as pl
import pytorch_lightning.loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from pytorch_lightning.plugins import DDPPlugin

from landing_pad_matcher.datamodules.rotation import RotationDataModule
from landing_pad_matcher.models.rotation_estimator import RotationEstimator


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

    data_module = RotationDataModule(data_path, batch_size, validation_batch_size, number_of_workers)

    model = RotationEstimator(lr=lr)

    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.5f}', monitor='val_loss', verbose=True)
    early_stop_callback = EarlyStopping(monitor='train_loss', patience=50)
    model_summary_callback = ModelSummary(max_depth=-1)
    # quantization_callback = QuantizationAwareTraining(qconfig='qnnpack',
    #                                                   modules_to_fuse=training_fusable)
    logger = pl.loggers.NeptuneLogger(project='Vision/LandingPadMatcher')

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[model_summary_callback, checkpoint_callback, early_stop_callback],
        gpus=-1,
        strategy=DDPPlugin(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True
        ),
        sync_batchnorm=True,
        precision=16,
        max_epochs=epochs,
        benchmark=True,
        accumulate_grad_batches=1
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, ckpt_path=checkpoint_callback.best_model_path, datamodule=data_module)


if __name__ == '__main__':
    train()
