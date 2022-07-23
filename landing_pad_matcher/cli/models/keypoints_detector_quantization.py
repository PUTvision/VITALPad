import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, Dict

import click
import numpy as np
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, create_calibrator, write_calibration_table

from landing_pad_matcher.datasets.landmarks import LandmarksDataset


class LandingPadDataReader(CalibrationDataReader):
    def __init__(self):
        self._iterations = 10000
        self._dataset = LandmarksDataset(Path('../../../data/pad.png'), num_samples=self._iterations)

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        iteration = self._iterations
        if iteration:
            self._iterations -= 1
            return {'input': self._dataset[iteration][0][None, ...].numpy()}

        return None


@click.argument('model-path')
@click.argument('output-dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
def export(model_path: Path, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    output_dir = output_dir.absolute()

    # TensorRT EP INT8 settings
    os.environ['ORT_TENSORRT_FP16_ENABLE'] = '1'  # Enable FP16 precision
    os.environ['ORT_TENSORRT_INT8_ENABLE'] = '1'  # Enable INT8 precision
    os.environ['ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME'] = 'calibration.flatbuffers'
    os.environ['ORT_TENSORRT_ENGINE_CACHE_ENABLE'] = '1'
    os.environ['ORT_TENSORRT_ENGINE_CACHE_PATH'] = str(output_dir)
    os.environ['ORT_TENSORRT_CACHE_PATH'] = str(output_dir)

    # Generate INT8 calibration table
    with NamedTemporaryFile() as augmented_model_file:
        calibrator = create_calibrator(str(model_path), [], augmented_model_path=augmented_model_file.name)
        calibrator.set_execution_providers(['CUDAExecutionProvider'])
        data_reader = LandingPadDataReader()
        calibrator.collect_data(data_reader)
        write_calibration_table(calibrator.compute_range())


if __name__ == '__main__':
    export()
