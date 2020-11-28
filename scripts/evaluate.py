import torch

import argparse
import os

from collections import OrderedDict

import dalib
from dalib.models import Detector
from dalib.datasets import DetectionDatasetsCollection
from dalib.metrics import apply_detector, DetectionDatasetEvaluator


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a face detection model."
    )

    parser.add_argument(
        "-c", "--config",
        type=str, default=None,
        help="Path to detector configuration file."
    )

    parser.add_argument(
        "-w", "--weights",
        type=str, required=True,
        help="Path to detector weights file."
    )

    parser.add_argument(
        "-d", "--data-dir",
        type=str, required=True,
        help="Root directory with datasets."
    )

    return parser.parse_args()


def main(args):
    collection = DetectionDatasetsCollection(args.data_dir)
    dataset_names = collection.get_descriptions().keys()

    detector = Detector(args.config)
    weights = torch.load(args.weights)
    try:
        detector.load_state_dict(weights)
    except:
        weights = OrderedDict([
            ('.'.join(key.split('.')[1:]), value) for key, value in weights["state_dict"].items()
        ])
        detector.load_state_dict(weights)


    for name in dataset_names:
        print(f"{name} AP@50:")
        dataset = collection.get_dataset(name, split="val")
        detection_data = apply_detector(detector, dataset)
        ev = DetectionDatasetEvaluator(config={
            "dataset": name,
            "min_height": 10,
            "max_height": "inf"
        })
        print("\tFace height in range [10 .. inf]:")
        metrics = ev(detection_data)
        for category, value in metrics.items():
            print(f"\t\t{category}: {value:.4f}")
        ev = DetectionDatasetEvaluator(config={
            "dataset": name,
            "min_height": 20,
            "max_height": 200,
        })
        print("\tFace height in range[20 .. 200]:")
        metrics = ev(detection_data)
        for category, value in metrics.items():
            print(f"\t\t{category}: {value:.4f}")
        print()

if __name__ == "__main__":
    main(get_args())
