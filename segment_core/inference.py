import argparse
import os
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", required=True)
parser.add_argument("--output_path", required=True)
parser.add_argument("--model_path", required=True)
parser.add_argument("--patch_size", required=True)
parser.add_argument("--pad_size", type=int, default=32)
args = parser.parse_args()

def report_progress(current, total):
    print(f"PROGRESS:{current}:{total}", flush=True)

module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

from CoreSeg import CoreSegInferenceBackend

resolutionMap = {
    "1:2": 256,
    "1:4": 512,
    "1:8": 1024,
    "1:16": 2048,
}

patch_size = resolutionMap[args.patch_size]
volume = np.load(args.input_path)

backend = CoreSegInferenceBackend()

prediction = backend.predictVolume(
    volume,
    args.model_path,
    patch_size,
    args.pad_size,
    progressCallback=report_progress,
)

np.save(args.output_path, prediction)