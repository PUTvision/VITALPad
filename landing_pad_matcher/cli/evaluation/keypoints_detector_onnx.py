import random
from pathlib import Path

import click
import numpy as np
import onnxruntime
import torch
import torchmetrics
from torchmetrics import MetricCollection

from landing_pad_matcher.datasets.landmarks import LandmarksDataset


@click.command()
@click.argument('model-dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('data-path', type=click.Path(exists=True, dir_okay=False, path_type=Path))
def evaluate(model_dir: Path, data_path: Path):
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(str(model_dir / 'keypoints_detector.onnx'), options, providers=[
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_fp16_enable': True,
            'trt_int8_enable': True,
            'trt_engine_cache_enable': True,
            'trt_int8_calibration_table_name': 'calibration.flatbuffers',
            'trt_engine_cache_path': str(model_dir)
        })
    ])

    torch.random.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    dataset = LandmarksDataset(data_path, 10000)

    classification_metrics = MetricCollection([
        torchmetrics.Accuracy(),
    ])
    regression_metrics = MetricCollection([
        torchmetrics.MeanAbsoluteError(),
        torchmetrics.MeanAbsolutePercentageError(),
        torchmetrics.MeanSquaredError()
    ])
    for index in range(len(dataset)):
        inputs, labels = dataset[index]
        inputs = inputs[None, ...].numpy()
        confidence, std, keypoints = session.run(None, {'input': inputs})

        classification_metrics.update(torch.tensor(confidence[0])[None, ...], labels[0][None, ...].type(torch.int64))
        regression_metrics.update(torch.from_numpy(keypoints.squeeze()), labels[1:])

    print(f'Classification: {classification_metrics.compute()}')
    print(f'Regression: {regression_metrics.compute()}')


if __name__ == '__main__':
    evaluate()
