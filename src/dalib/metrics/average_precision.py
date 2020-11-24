import torch
import numpy as np


def iou(boxes_a, boxes_b):
    """Args:
        boxes_a: Numpy array of boxes with shape :math:`(N_a, 4)`.
        boxes_b: Numpy array of boxes with shape :math:`(N_b, 4)`.

    Returns:
        Numpy matrix with IoU of each pair of boxes  with shape :math:`(N_a, N_b)`.
    """
    boxes_a = boxes_a[:,None,:]
    boxes_b = boxes_b[None,:,:]

    ix = np.maximum(0, np.minimum(boxes_a[...,2], boxes_b[...,2]) - np.maximum(boxes_a[...,0], boxes_b[...,0]))
    iy = np.maximum(0, np.minimum(boxes_a[...,3], boxes_b[...,3]) - np.maximum(boxes_a[...,1], boxes_b[...,1]))
    inter = ix*iy

    ar_a = np.maximum(0, (boxes_a[...,2:] - boxes_a[...,:2])).prod(axis=-1)
    ar_b = np.maximum(0, (boxes_b[...,2:] - boxes_b[...,:2])).prod(axis=-1)

    iou_mat = inter / (ar_a + ar_b - inter + 1e-10)

    return iou_mat


def count_gt_tp_fp(scores, bboxes_pr, bboxes_gt, subsets=None, resolution=100, iou_threshold=0.5):
    """Args:
        scores: Numpy array of scores with shape :math:`(N_pr,)`.
        bboxes_pr: Numpy array of predicted bboxes with shape :math:`(N_pr, 4)`.
        bboxes_gt: Numpy array of ground truth bboxes with shape :math:`(N_gt, 4)`.
        subsets: None or list of arrays of ground truth bboxes indices. Default: None.
        resolution: Int, score resolution. Scores thresholds are generated
            via np.linspace(0,1,resolution). Default: 100.
        iou_threshold: Float, threshold for IoU of predicted and ground truth bboxes for
            separating hits and misses.

    Returns:
        Numpy array of ground truth, true positive and false positive counts 
        for different values of IoU thresholds. Has shape :math:`(resolution, 3)` if subsets is None,
        and :math:`(len(subsets), resolution, 3)` otherwise.
    """
    if isinstance(scores, torch.Tensor): scores = scores.numpy()
    if isinstance(bboxes_pr, torch.Tensor): bboxes_pr = bboxes_pr.numpy()
    if isinstance(bboxes_gt, torch.Tensor): bboxes_gt = bboxes_gt.numpy()
    scores_grid = np.linspace(0,1,resolution)
    precision_tables, recall_tables = {}, {}

    if subsets is None:
        single_result = True
        subsets = [np.arange(len(bboxes_gt))]
    else:
        single_result = False

    if len(bboxes_gt) == 0:
        N_fp = (scores[None,:] >= scores_grid[:,None]).sum(axis=1) if len(scores) > 0 else np.zeros(resolution)
        N_fp = N_fp.astype(np.int)
        N_tp = np.zeros(resolution).astype(np.int)
        N_gt = np.zeros(resolution).astype(np.int)
        result = np.stack([N_gt, N_tp, N_fp], axis=-1)
        if single_result:
            return result
        else:
            return np.tile(result[None,:,:], (len(subsets), 1, 1))

    if len(bboxes_pr) == 0:
        N_fp = np.zeros(resolution).astype(np.int)
        N_tp = np.zeros(resolution).astype(np.int)
        N_gt = np.ones(resolution).astype(np.int)
        N_gt = np.array([len(s) for s in subsets])[:,None] * N_gt[None,:]
        result = np.stack([
            N_gt,
            np.tile(N_tp[None,:], (len(subsets), 1)),
            np.tile(N_fp[None,:], (len(subsets), 1)),
        ], axis=-1)
        if single_result:
            return result.squeeze(0)
        else:
            return result

    iou_mat = iou(bboxes_pr, bboxes_gt)

    best_match = iou_mat.argmax(axis=1)
    best_iou   = iou_mat[np.arange(len(bboxes_pr)), best_match]

    matched = (best_iou >= iou_threshold)

    missed_inds,  = np.where(np.logical_not(matched))
    miss_scores = scores[missed_inds]

    best_match = best_match[matched]
    scores = scores[matched]

    hit_scores = np.zeros(len(bboxes_gt))
    for i, s in zip(best_match, scores):
        threshold = hit_scores[i]
        if s > threshold:
            hit_scores[i] = s

    misses_table = (miss_scores[None,:] >= scores_grid[:,None]).sum(axis=1)

    result = []
    for subset in subsets:
        _hit_scores = hit_scores[subset]
        _matched_scores = scores[np.isin(best_match, subset)]

        N_tp = (_hit_scores[None,:] > scores_grid[:,None]).sum(axis=1)
        N_fp = misses_table + (_matched_scores[None,:] >= scores_grid[:,None]).sum(axis=1) - N_tp
        N_gt = len(subset)*np.ones(resolution).astype(np.int)

        result.append(np.stack([N_gt, N_tp, N_fp], axis=-1))

    result = np.stack(result)

    if single_result:
        return result.squeeze(0)
    else:
        return result


def calculate_PR(gt_tp_fp_table):
    """Args:
        gt_tp_fp_table: Numpy array of ground truth, true positives and false positives counts
            of shape :math:`(resolution, 3)` or :math:`(n_subsets, resolution, 3)`.

    Returns:
        Numpy array of precision and recall scores of shape :math:`(resolution, 2)`.
    """
    n_pred = gt_tp_fp_table[..., 1] + gt_tp_fp_table[..., 2]
    precision = gt_tp_fp_table[..., 1] / np.maximum(1, n_pred)
    precision[n_pred == 0] = 1
    recall = gt_tp_fp_table[..., 1] / np.maximum(1, gt_tp_fp_table[..., 0])
    recall[gt_tp_fp_table[...,0] == 0] = 1

    for i in range(len(precision) - 1):
        precision[i + 1] = max(precision[i], precision[i+1])

    return np.stack([precision, recall], axis=-1)


def calculate_AP(PR_table):
    """Args:
        PR_table: Numpy array of precision and recall scores of shape :math:`(resolution, 2)`.

    Returns:
        Float, average precision score. 
    """
    PR_table = np.concatenate([PR_table, np.array([[1,0]])], axis=0)
    heights = PR_table[:-1,0]
    widths = PR_table[:-1,1] - PR_table[1:,1]
    area = (heights*widths).sum()
    return area


class AveragePrecisionCalculator:
    """Utility class for calculation of average precision (AP)
    and generation of precision-recall curves.

    Args:
        discriminator: a callable that decides whether to include a given
            sample in statistics and over which subsets of targets
            to calculate the metrics.
                Args:
                    item: {
                        "scores":  :math:`(N,)`,
                        "bboxes_pr": :math:`(N,4)`,
                        "bboxes_gt": :math:'(N_{gt},4)',
                      ...
                    }
                Returns:
                (
                        bool, True if include the sample else False
                    {
                        subset_name: Numpy array :math:`(N_{sub},)` of ground truth labels indices
                            for sub in subsets
                    }
                )
        resolution: Int, number of score thresholds.
        iou_threshold: Float, true positive threshold.
    """
    def __init__(self, discriminator=None, resolution=100, iou_threshold=0.5):
        self.discriminator = discriminator
        self.resolution = resolution
        self.iou_threshold = iou_threshold

    @staticmethod
    def default_discriminator(item):
        do_count = True
        subsets = {"all": np.arange(len(item["bboxes_gt"]))}
        return do_count, subsets

    def __call__(self, data):
        """Args:
            data: [
                {
                   "scores": :math:(N_{i},),
                   "bboxes_pr": :math:(N_{i},4),
                   "bboxes_gt": :math:(N_gt_{i},4),
                   ...
                }
                for i in range(number_of_samples)
            ]
        Returns: {
            A dictionary of AP scores by subset name.
        }
        """
        discriminator = self.discriminator if self.discriminator is not None\
                        else self.default_discriminator
        gt_tp_fp_accumulator = {}
        for item in data:
            do_count, subsets = discriminator(item)
            if not do_count:
                continue
            gt_tp_fp = count_gt_tp_fp(item["scores"], item["bboxes_pr"], item["bboxes_gt"],
                                subsets.values(), self.resolution, self.iou_threshold)
            for key, value in zip(subsets.keys(), gt_tp_fp):
                try:
                    gt_tp_fp_accumulator[key] += value
                except:
                    gt_tp_fp_accumulator[key] = value

        precision_recall = {key: calculate_PR(value) for key, value in gt_tp_fp_accumulator.items()}
        AP = {key: calculate_AP(value) for key, value in precision_recall.items()}

        self.precision_recall = precision_recall
        self.AP = AP

        return AP
