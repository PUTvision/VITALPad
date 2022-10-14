from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import pytorch_lightning.loggers
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from pytorch_lightning.strategies import DDPStrategy

from landing_pad_matcher.datamodules.density import DensityDataModule
from landing_pad_matcher.models.density_estimator import DensityEstimator


def train(model_name: str, encoder_name: str, data_path: str, batch_size: int, validation_batch_size: int,
          epochs: int, lr: float, number_of_workers: Optional[int]):
    pl.seed_everything(42)

    data_path = Path(hydra.utils.to_absolute_path(data_path))
    data_module = DensityDataModule(data_path, batch_size, validation_batch_size, number_of_workers)

    # classes_weights = data_module.compute_class_weights()
    classes_weights = torch.tensor([0.3472, 67.1499, 9.5648], dtype=torch.float64)
    model = DensityEstimator(model_name=model_name, encoder_name=encoder_name, lr=lr, classes_weights=classes_weights)

    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.5f}', monitor='val_loss', verbose=True)
    early_stop_callback = EarlyStopping(monitor='train_loss', patience=50)
    model_summary_callback = ModelSummary(max_depth=-1)
    logger = pl.loggers.NeptuneLogger(project='Vision/LandingDensityEstimation')

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
        precision=16,
        max_epochs=epochs,
        benchmark=True,
        accumulate_grad_batches=1
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, ckpt_path=checkpoint_callback.best_model_path, datamodule=data_module)

    logger.experiment.stop()
