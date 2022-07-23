from pathlib import Path

import click
import onnxruntime
import torch
import torchmetrics
from sklearn.model_selection import train_test_split

from landing_pad_matcher.datamodules.density import DensityDataModule
from landing_pad_matcher.datasets.density import DensityDataset


@click.command()
@click.argument('model-dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument('data-path', type=click.Path(exists=True, file_okay=False, path_type=Path))
def evaluate(model_dir: Path, data_path: Path):
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(str(model_dir / 'density_estimator.yaml.onnx'), options, providers=[
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_fp16_enable': True,
            'trt_int8_enable': True,
            'trt_engine_cache_enable': True,
            'trt_int8_calibration_table_name': 'calibration.flatbuffers',
            'trt_engine_cache_path': str(model_dir)
        })
    ])

    paths = DensityDataModule.get_paths(data_path)
    _, val_paths = train_test_split(paths, test_size=0.15, random_state=42)
    dataset = DensityDataset(val_paths)

    metrics = torchmetrics.MeanAbsoluteError()
    for i, (inputs, labels) in enumerate(dataset):
        if i > 99:
            break

        inputs = inputs[None, ...].numpy()
        outputs = torch.from_numpy(session.run(None, {'input': inputs})[0].squeeze())
        metrics.update(outputs, labels)

    print(metrics.compute())


if __name__ == '__main__':
    evaluate()
