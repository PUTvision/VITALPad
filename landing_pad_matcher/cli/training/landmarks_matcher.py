from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import pytorch_lightning.loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from pytorch_lightning.strategies import DDPStrategy

from landing_pad_matcher.datamodules.landmarks import LandmarksDataModule
from landing_pad_matcher.models.landmarks_regressor import LandmarksRegressor


def train(data_path: str, textures_path: str, model_name: str, batch_size: int,
          validation_batch_size: int, epochs: int, lr: float, number_of_workers: int,
          photos_path: Optional[str] = None):
    pl.seed_everything(42)

    data_path = Path(hydra.utils.to_absolute_path(data_path))
    textures_path = Path(hydra.utils.to_absolute_path(textures_path))
    photos_path = Path(hydra.utils.to_absolute_path(photos_path))
    data_module = LandmarksDataModule(data_path, textures_path, batch_size, validation_batch_size, photos_path,
                                      number_of_workers)

    model = LandmarksRegressor(model_name=model_name, lr=lr)

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
    early_stop_callback = EarlyStopping(monitor='train_loss', patience=10)
    model_summary_callback = ModelSummary(max_depth=-1)
    # quantization_callback = QuantizationAwareTraining(qconfig='fbgemm')
    logger = pl.loggers.NeptuneLogger(project='Vision/LandingPadMatcher')

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[model_summary_callback, checkpoint_callback, early_stop_callback],
        gpus=-1,
        strategy=DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True
        ),
        sync_batchnorm=True,
        precision=32,
        max_epochs=epochs,
        benchmark=True,
        accumulate_grad_batches=1
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, ckpt_path=checkpoint_callback.best_model_path, datamodule=data_module)

    logger.experiment.stop()
