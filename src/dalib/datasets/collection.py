import os
from .widerface import WIDERFACEDataset
from .facemask import FaceMaskDataset
from .fddb import FDDBDataset

from collections import OrderedDict

DATASETS = OrderedDict([
    ("widerface", WIDERFACEDataset),
    ("facemask", FaceMaskDataset),
    ("fddb", FDDBDataset),
])


class DetectionDatasetsCollection:
    """Detection datasets collection.

    Args:
        data_dir: path to the datasets directory.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_dataset(self, name, split="train"):
        dataset = DATASETS[name](
            root=os.path.join(self.data_dir, name), split=split
        )
        dataset.name = name
        return dataset

    @staticmethod
    def get_descriptions():
        return OrderedDict([
            (name, {
                "description": dataset.__doc__,
            })
            for name, dataset in DATASETS.items()
        ])

    def __repr__(self):
        repr_str = ""
        for name, desc in self.get_descriptions().items():
            repr_str += f"{name}:\n"
            if desc["description"]:
                repr_str += desc["description"]
            repr_str += "\n-------------\n"
        return repr_str
