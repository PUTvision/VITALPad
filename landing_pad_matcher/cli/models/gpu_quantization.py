import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, Dict

import click
import numpy as np
from onnxruntime.quantization import CalibrationDataReader, create_calibrator, write_calibration_table

from landing_pad_matcher.datamodules.density import DensityDataModule
from landing_pad_matcher.datasets.density import DensityDataset
from landing_pad_matcher.datasets.landmarks import LandmarksDataset


class DensityEstimatorDataReader(CalibrationDataReader):
    def __init__(self, data_path: Path):
        paths = DensityDataModule.get_paths(data_path)
        self._dataset = DensityDataset(paths)
        self._iterations = min(len(self._dataset) - 1, 100)

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        iteration = self._iterations
        if iteration >= 0:
            self._iterations -= 1
            return {'input': self._dataset[iteration][0][None, ...].numpy()}

        return None


class LandingPadDataReader(CalibrationDataReader):
    def __init__(self, data_path: Path):
        self._iterations = 100
        self._dataset = LandmarksDataset(data_path, num_samples=self._iterations)

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        iteration = self._iterations
        if iteration:
            self._iterations -= 1
            return {'input': self._dataset[iteration][0][None, ...].numpy()}

        return None


@click.command()
@click.argument('model-type', type=click.Choice(('DensityEstimator', 'KeypointsDetector')))
@click.argument('model-path', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument('data-path', type=click.Path(exists=True, path_type=Path))
@click.argument('output-dir', type=click.Path(file_okay=False, path_type=Path))
def export(model_type: str, model_path: Path, data_path: Path, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    output_dir = output_dir.absolute()

    # TensorRT EP INT8 settings
    os.environ['ORT_TENSORRT_FP16_ENABLE'] = '1'  # Enable FP16 precision
    os.environ['ORT_TENSORRT_INT8_ENABLE'] = '1'  # Enable INT8 precision
    os.environ['ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME'] = 'calibration.flatbuffers'
    os.environ['ORT_TENSORRT_ENGINE_CACHE_ENABLE'] = '1'
    os.environ['ORT_TENSORRT_ENGINE_CACHE_PATH'] = str(output_dir)
    os.environ['ORT_TENSORRT_CACHE_PATH'] = str(output_dir)

    if model_type == 'DensityEstimator':
        data_reader = DensityEstimatorDataReader(data_path)
    else:
        data_reader = LandingPadDataReader(data_path)

    # Generate INT8 calibration table
    with NamedTemporaryFile() as augmented_model_file:
        calibrator = create_calibrator(str(model_path), [], augmented_model_path=augmented_model_file.name)
        calibrator.set_execution_providers(['CUDAExecutionProvider'])
        calibrator.collect_data(data_reader)

        shutil.copy(model_path, output_dir)

        os.chdir(output_dir)
        write_calibration_table(calibrator.compute_range())


if __name__ == '__main__':
    export()
