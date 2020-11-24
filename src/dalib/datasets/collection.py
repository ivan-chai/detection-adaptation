import os
from .widerface import WIDERFACEDataset
from .facemask import FaceMaskDataset
from .fddb import FDDBDataset

datasets_dict = {
    "WIDERFACE": WIDERFACEDataset,
    "FaceMask": FaceMaskDataset,
    "FDDB": FDDBDataset,
}

environ_dict = {
    "WIDERFACE": "WIDERFACE_DIR",
    "FaceMask": "FACEMASK_DIR",
    "FDDB": "FDDB_DIR",
}


class Collection:
    """Datasets collection.

    Args:
        path_dict: a dict of dataset paths. If path is provided for a given dataset,
            it overrides the path inferred from environment variables. Default: {}

    """
    def __init__(self, path_dict={}):
        self.datasets_dict = datasets_dict
        self.path_dict = {}
        for key, var in environ_dict.items():
            path = os.environ.get(var)
            if path:
                self.path_dict[key] = path
        self.path_dict.update(path_dict)

    def get_dataset(self, name, split="train"):
        dataset = self.datasets_dict[name](
            root=self.path_dict.get(name), split=split
        )
        dataset.name = name
        return dataset

    def get_descriptions(self):
        return {
            name: {
                "path": self.path_dict.get(name),
                "path_env_var": environ_dict.get(name),
                "description": dataset.__doc__,
            }
            for name, dataset in self.datasets_dict.items()
        }

    def __repr__(self):
        repr_str = ""
        for name, desc in self.get_descriptions().items():
            repr_str += f"{name}:\n"
            repr_str += "path: " + str(desc["path"]) + "\n"
            repr_str += "path_env_var: " + str(desc["path_env_var"]) + "\n"
            if desc["description"]:
                repr_str += desc["description"]
            repr_str += "\n-------------\n"
        return repr_str
