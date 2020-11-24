import torch
import numpy as np

from torchvision.transforms.functional import to_tensor

from tqdm import tqdm

from . import AveragePrecisionCalculator


def apply_detector(detector, dataset, device="cuda:0"):
    """Args:
        detector: torch.nn.Module that has a :meth:`predict` that accepts a
            tensor of shape :math:`(1,3,H,W)` and returns a list of len 1 of
            shape [
                {
                    "scores": Numpy array :math:`(N,)`,
                    "bboxes": Numpy array :math:`(N, 4)`,
                    ...
                }
            ]
        dataset: An instance of torch.utils.data.Dataset. A sample should
            have form (image, target), where target is a dictionary.
        device: device on which prediction generation is done. Default: "cuda:0".
    """
    detector = detector.eval().to(device)
    data = []
    for image, target\
            in tqdm(dataset, desc="Generating validation predictions..."):
        with torch.no_grad():
            pred, = detector.predict(to_tensor(image)[None,...].to(device),
                    score_threshold=0.01)
            for key in pred.keys():
                if isinstance(pred[key], torch.Tensor):
                    pred[key] = pred[key].numpy()
        item = target.copy()
        item["scores"] = pred["scores"]
        item["bboxes_pr"] = pred["bboxes_pr"]
        item["bboxes_gt"] = pred["bboxes_gt"]
        data.append(item)

    return data


class DetectionDatasetEvaluator:
    """Utility class for evaluating detection quality over a dataset."""

    discriminators = {
        "WIDERFACE": _widerface_discriminator,
        "FaceMask": _facemask_discriminator,
        "FDDB": _fddb_discriminator,
    }

    @staticmethod
    def get_default_config():
        return OrderedDict([
            ("dataset", "WIDERFACE"),
            ("min_height", 10),
            ("max_height", float("inf")),
            ("resolution", 100),
            ("iou_threshold", 0.5),
        ])

    def __init__(self, config=None):
        config = prepare_config(self, config)
        config["max_height"] = float(config["max_height"])
        self.discriminator = lambda item: self.discriminators[config["dataset"]](item, config["min_height"], config["max_height"])
        self.config = config

    @staticmethod
    def _widerface_discriminator(item, min_height=10, max_height=float("inf")):
        do_count = item["difficulty"] is not None
        bboxes = item["bboxes_gt"]
        heights = bboxes[:,3] - bboxes[:,1]
        all_subset, = np.where(np.logical_and(heights < max_height, heights >= min_height))
        subsets = {"all": all_subset, item["difficulty"]: all_subset}
        return do_count, subsets

    @staticmethod
    def _facemask_discriminator(item, min_height=10, max_height=float("inf")):
        do_count = True
        bboxes = item["bboxes_gt"]
        heights = bboxes[:,3] - bboxes[:,1]
        proper_sized = np.logical_and(heights < max_height, heights >= min_height)
        all_subset, = np.where(proper_sized)
        masked_subset, = np.where(np.logical_and(proper_sized, item["with_mask"]))
        nonmasked_subset, = np.where(np.logical_and(proper_sized,\
                np.logical_not(item["with_mask"])))
        subsets = {
            "all": all_subset,
            "with_mask": masked_subset,
            "without_mask": nonmasked_subset,
        }
        return do_count, subsets

    @staticmethod
    def _fddb_discriminator(item, min_height=10, max_height=float("inf")):
        do_count = True
        bboxes = item["bboxes_gt"]
        heights = bboxes[:,3] - bboxes[:,1]
        all_subset, = np.where(np.logical_and(heights < max_height, heights >= min_height))
        subsets = {"all": all_subset}
        return do_count, subsets


    def __call__(self, evaluation_data, per_image=False):
        """Args:
            evaluation_data: [
                {
                    "scores": Numpy array :math:`(N,)`,
                    "bboxes_pr": Numpy array :math:`(N, 4)`,
                    "bboxes_gt": Numpy array :math:`(N_{gt},  4)`,
                    ...
                }
            ]
            per_image: whether to calculate AP score for each image separately, or for the dataset
                as a whole. Default: False.

        Returns:
           Numpy array of AP scores with shape :math:`(N_images,)` if per_image is True,
           dictionary of AP scores by subset name otherwise.
        """

        ap = AveragePrecisionCalculator(discriminator, resolution, iou_threshold)
        if per_image:
            AP_array = np.array([ap([item])["all"] for item in self.data])
            return AP_array
        else:
            ap(self.data)
            self.AP = ap.AP
            self.precision_recall = ap.precision_recall
            return self.AP
