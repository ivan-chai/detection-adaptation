import torch
import numpy as np

from torchvision.transforms.functional import to_tensor

from tqdm import tqdm

from . import AveragePrecisionCalculator

class DatasetEvaluator:
    """Utility class for evaluating detection quality over a dataset."""
    def get_dataset(self):
        raise NotImplementedError()

    @staticmethod
    def specific_data(target):
        return {}

    def fit(self, detector, device="cuda:0"):
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

    def predict(self, custom_discriminator=None, min_height=10, max_height=float("inf"),
            resolution=100, iou_threshold=0.5, per_image=False):

        discriminator = custom_discriminator if custom_discriminator is not None\
                else lambda x: self.default_discriminator(x, min_height, max_height)

        ap = AveragePrecisionCalculator(discriminator, resolution, iou_threshold)
        if per_image:
            AP_array = np.array([ap.fit_predict([item])["all"] for item in self.data])
            return AP_array
        else:
            ap.fit_predict(self.data)
            self.AP = ap.AP
            self.precision_recall = ap.precision_recall
            return self.AP
