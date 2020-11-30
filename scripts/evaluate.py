import torch

import argparse
import os

from collections import OrderedDict, defaultdict

from rich.console import Console
from rich.table import Table

import dalib
from dalib.models import Detector
from dalib.datasets import DetectionDatasetsCollection
from dalib.metrics import apply_detector, DetectionDatasetEvaluator

SIZE_BINS = [
    [10, float("inf")],
    [20, 200],
    [10, 20],
    [20, 50],
    [50, 100],
    [100, 200],
    [200, float("inf")],
]

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
            ('.'.join(key.split('.')[1:]), value) for key, value in weights["state_dict"].items()\
                    if key.split('.')[0] == "detector"
        ])
        detector.load_state_dict(weights)

    console = Console()
    for name in dataset_names:
        dataset = collection.get_dataset(name, split="val")
        detection_data = apply_detector(detector, dataset)

        console.print()
        table = Table("", title=f"{name} AP@50")
        table_data = defaultdict(list)
        for min_height, max_height in SIZE_BINS:
            table.add_column(f"{min_height}..{max_height} px")
            ev = DetectionDatasetEvaluator({
                "dataset": name,
                "min_height": min_height,
                "max_height": max_height,
            })
            metrics = ev(detection_data)
            for category, ap_score in metrics.items():
                table_data[category].append(f"{ap_score:.3f}")

        for category, ap_scores in table_data.items():
            table.add_row(category, *ap_scores)
        
        console.print(table)
        console.print()


if __name__ == "__main__":
    main(get_args())
