import time

import numpy as np
import onnxruntime

options = onnxruntime.SessionOptions()
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session = onnxruntime.InferenceSession('keypoints_detector.onnx', options)

inputs = np.random.rand(1, 3, 128, 128).astype(np.float32)

iterations = 5000

start = time.perf_counter()
for _ in range(iterations):
    outs = session.run(None, {'input': inputs})
end = time.perf_counter()

time_per_sample = (end - start) / iterations

print(1 / time_per_sample)
