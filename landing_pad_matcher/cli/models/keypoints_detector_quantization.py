import os
from pathlib import Path
from typing import Optional, Dict

import numpy as np
from onnxruntime.quantization import CalibrationDataReader, create_calibrator, write_calibration_table

from landing_pad_matcher.datasets.landmarks import LandmarksDataset


class LandingPadDataReader(CalibrationDataReader):
    def __init__(self):
        self._iterations = 10000
        self._dataset = LandmarksDataset(Path('../../data/pad.png'), num_samples=self._iterations)

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        iteration = self._iterations
        if iteration:
            self._iterations -= 1
            return {'input': self._dataset[iteration][0][None, ...].numpy()}

        return None


def main():
    '''
    TensorRT EP INT8 Inference on Resnet model
    The script is using ILSVRC2012 ImageNet dataset for calibration and prediction.
    Please prepare the dataset as below,
    1. Create dataset folder 'ILSVRC2012' in workspace.
    2. Download ILSVRC2012 validation dataset and development kit from http://www.image-net.org/challenges/LSVRC/2012/downloads.
    3. Extract validation dataset JPEG files to 'ILSVRC2012/val'.
    4. Extract development kit to 'ILSVRC2012/devkit'. Two files in the development kit are used, 'ILSVRC2012_validation_ground_truth.txt' and 'meta.mat'.
    5. Download 'synset_words.txt' from https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt into 'ILSVRC2012/'.

    Please download Resnet50 model from ONNX model zoo https://github.com/onnx/models/blob/master/vision/classification/resnet/model/resnet50-v2-7.tar.gz
    Untar the model into the workspace
    '''

    # Dataset settings
    model_path = "../../keypoints_detector.onnx"
    augmented_model_path = "../../quantized_keypoints_detector.onnx"

    # TensorRT EP INT8 settings
    os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"  # Enable FP16 precision
    os.environ["ORT_TENSORRT_INT8_ENABLE"] = "1"  # Enable INT8 precision
    os.environ["ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME"] = "calibration.flatbuffers"  # Calibration table name
    os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"  # Enable engine caching
    os.environ["ORT_TENSORRT_ENGINE_CACHE_PATH"] = "../../keypoints_detector_cache"

    # Generate INT8 calibration table
    calibrator = create_calibrator(model_path, [], augmented_model_path=augmented_model_path)
    calibrator.set_execution_providers(["CUDAExecutionProvider"])
    data_reader = LandingPadDataReader()
    calibrator.collect_data(data_reader)
    write_calibration_table(calibrator.compute_range())


if __name__ == '__main__':
    main()
