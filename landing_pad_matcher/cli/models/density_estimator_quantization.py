import os
from pathlib import Path
from typing import Optional, Dict

import numpy as np
from onnxruntime.quantization import CalibrationDataReader, create_calibrator, write_calibration_table

from landing_pad_matcher.datamodules.density import DensityDataModule
from landing_pad_matcher.datasets.density import DensityDataset
from landing_pad_matcher.datasets.landmarks import LandmarksDataset


class DensityEstimatorDataReader(CalibrationDataReader):
    def __init__(self):
        paths = DensityDataModule.get_paths(Path('/home/rivi/Datasets/Landing'))
        self._dataset = DensityDataset(paths)
        self._iterations = len(self._dataset) - 1

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        iteration = self._iterations
        if iteration >= 0:
            self._iterations -= 1
            return {'input': self._dataset[iteration][0][None, ...].numpy()}

        return None


def main():
    # Dataset settings
    model_path = "../../density_estimator.onnx"
    augmented_model_path = "../../quantized_density_estimator.onnx"

    # TensorRT EP INT8 settings
    os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"  # Enable FP16 precision
    os.environ["ORT_TENSORRT_INT8_ENABLE"] = "1"  # Enable INT8 precision
    os.environ["ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME"] = "calibration.flatbuffers"  # Calibration table name
    os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"  # Enable engine caching
    os.environ["ORT_TENSORRT_ENGINE_CACHE_PATH"] = "../../density_estimator_cache"

    # Generate INT8 calibration table
    calibrator = create_calibrator(model_path, [], augmented_model_path=augmented_model_path)
    calibrator.set_execution_providers(["CUDAExecutionProvider"])
    data_reader = DensityEstimatorDataReader()
    calibrator.collect_data(data_reader)
    write_calibration_table(calibrator.compute_range())


if __name__ == '__main__':
    main()
