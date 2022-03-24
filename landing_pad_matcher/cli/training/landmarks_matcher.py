import os
from pathlib import Path
from typing import Optional

import click
import pytorch_lightning as pl
import pytorch_lightning.loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary, QuantizationAwareTraining
from pytorch_lightning.plugins import DDPPlugin

from landing_pad_matcher.datamodules.landmarks import LandmarksDataModule
from landing_pad_matcher.models.landmarks_regressor import LandmarksRegressor


@click.command()
@click.option('--data-path', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--textures-path', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('--photos-path', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--batch-size', type=int, default=256)
@click.option('--validation-batch-size', type=int, default=256)
@click.option('--epochs', type=int, default=1000)
@click.option('--lr', type=float, default=1e-3)
@click.option('--number-of-workers', type=int, default=os.getenv('WORKERS', 4))
def train(data_path: Path, textures_path: Path, photos_path: Optional[Path], batch_size: int,
          validation_batch_size: int, epochs: int, lr: float, number_of_workers: Optional[int]):
    pl.seed_everything(42)

    data_module = LandmarksDataModule(data_path, textures_path, batch_size, validation_batch_size, photos_path,
                                      number_of_workers)

    model = LandmarksRegressor(lr=lr)

    # blocks_residuals = [1, 2, 2, 2, 4, 2]
    # fusable = [('network.conv_stem', 'network.bn1', 'network.act1'),
    #            ('network.conv_head', 'network.act2')]
    # for block in range(6):
    #     for residual in range(blocks_residuals[block]):
    #         fusable.append((f'network.blocks.{block}.{residual}.conv_dw',
    #                         f'network.blocks.{block}.{residual}.bn1',
    #                         f'network.blocks.{block}.{residual}.act1'))
    #         fusable.append((f'network.blocks.{block}.{residual}.conv_pw',
    #                         f'network.blocks.{block}.{residual}.bn2',
    #                         f'network.blocks.{block}.{residual}.act2'))
    #         if block == 5:
    #             fusable.append((f'network.blocks.{block}.{residual}.se.conv_reduce',
    #                             f'network.blocks.{block}.{residual}.se.act1',
    #                             f'network.blocks.{block}.{residual}.se.conv_expand',
    #                             f'network.blocks.{block}.{residual}.se.gate'))

    # training_fusable = [('network.conv_head', 'network.act2')]
    # for residual in range(blocks_residuals[5]):
    #     training_fusable.append((f'network.blocks.5.{residual}.se.conv_reduce',
    #                              f'network.blocks.5.{residual}.se.act1'))

    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.5f}', monitor='val_loss', verbose=True)
    early_stop_callback = EarlyStopping(monitor='train_loss', patience=50)
    model_summary_callback = ModelSummary(max_depth=-1)
    # quantization_callback = QuantizationAwareTraining(qconfig='fbgemm')
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
