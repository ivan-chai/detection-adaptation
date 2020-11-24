import torch
import numpy as np

from torchvision.transforms.functional import to_tensor

from tqdm import tqdm

from . import AveragePrecisionCalculator


class DetectionDatasetEvaluator:
    """Utility class for evaluating detection quality over a dataset."""
    def get_dataset(self):
        raise NotImplementedError()

    @staticmethod
    def enrich(target):
        return {}

    def setup(self, detector, device="cuda:0"):
        """Args:
            detector: torch.nn.Module that has a :meth:`predict` that accepts a
                tensor of shape :math:`(1,3,H,W)` and returns a list of len 1 of
                shape [
                    {
                        "scores": Numpy array :math:`(N,)`,
                        "bboxes": Numpy array :math:`(N,4)`,
                        ...
                    }
                ]
            device: device on which prediction generation is done. Default: "cuda:0".

        Children should implement :meth:`get_default_discriminator` or provide custom discriminator
        function on call (see :class:`AveragePrecisionCalculator` docs for signature). 

        Children should implement :meth:`get_dataset` to get the evaluation dataset.

        Children can override :meth:`enrich` static method to enrich ground truth labels for discriminator to use.
        """
        detector = detector.eval().to(device)
        dataset = self.get_dataset()

        data = []

        for image, target\
                in tqdm(dataset, desc="Generating validation predictions..."):
            with torch.no_grad():
                pred, = detector.predict(to_tensor(image)[None,...].to(device),
                        score_threshold=0.01)
                for key in pred.keys():
                    if isinstance(pred[key], torch.Tensor):
                        pred[key] = pred[key].numpy()
            item = {
                    "scores": pred["scores"],
                    "bboxes_pr": pred["bboxes"],
                    "bboxes_gt": target["bboxes"],
            }
            item.update(self.specific_data(target))
            data.append(item)

        self.data = data

    @staticmethod
    def get_default_discriminator(*args, **kwargs):
        raise NotImplementedError()


    def __call__(self, custom_discriminator=None, min_height=10, max_height=float("inf"),
            resolution=100, iou_threshold=0.5, per_image=False):
        """Args:
            custom_discriminator: a custom discriminator function. See :class:`AveragePrecisionCalculator`
            docs for signature. Default: None.
            min_height: Float, minimal ground truth bbox height to be evaluated. Ignored if
                custom discriminator is not None. Default: 10.
            max_height: Float, maximal ground truth bbox height to be evaluated. Ignored if
                custom discriminator is not None. Default: inf.
            resolution: Int, score resolution to use for AP calculation. Default: 100.
            iou_threshold: Float, IoU threshold for discriminating between true positives and
                false positives. Default: 0.5.
            per_image: whether to calculate AP score for each image separately, or for the dataset
                as a whole. Default: False.

        Returns:
           Numpy array of AP scores with shape :math:`(N_images,)` if per_image is True,
           dictionary of AP scores by subset name otherwise.
        """

        discriminator = custom_discriminator if custom_discriminator is not None\
                else lambda x: self.default_discriminator(x, min_height, max_height)

        ap = AveragePrecisionCalculator(discriminator, resolution, iou_threshold)
        if per_image:
            AP_array = np.array([ap([item])["all"] for item in self.data])
            return AP_array
        else:
            ap(self.data)
            self.AP = ap.AP
            self.precision_recall = ap.precision_recall
            return self.AP
