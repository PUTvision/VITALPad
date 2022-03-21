from onnxruntime.quantization import quantize_static, \
    QuantFormat, QuantType

from landing_pad_matcher.cli.models.keypoints_detector_quantization import LandingPadDataReader


def main():
    model_path = "../../keypoints_detector.onnx"
    quantized_model_path = "../../quantized_keypoints_detector.onnx"

    data_reader = LandingPadDataReader()
    quantize_static(model_path, quantized_model_path, data_reader,
                    quant_format=QuantFormat.QOperator, weight_type=QuantType.QInt8)


if __name__ == '__main__':
    main()
