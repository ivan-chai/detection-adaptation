import torch
import numpy as np

def iou(boxes_a, boxes_b):
    """
    boxes_a: array<N_a, 4>
    boxes_b: array<N_b, 4>

    returns: array<N_a, N_b>
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

def calculate_PR(scores, bboxes_pr, bboxes_gt, subsets=None, resolution=100, iou_threshold=0.5):
    if isinstance(scores, torch.Tensor): scores = scores.numpy()
    if isinstance(bboxes_pr, torch.Tensor): bboxes_pr = bboxes_pr.numpy()
    if isinstance(bboxes_gt, torch.Tensor): bboxes_gt = bboxes_gt.numpy()
    grid = np.linspace(0,1,resolution)
    precision_tables, recall_tables = {}, {}

    if subsets is None:
        single_result = True
        subsets = [np.arange(len(bboxes_gt))]
    else:
        single_result = False

    if len(bboxes_gt) == 0:
        misses_table = (scores[None,:] >= grid[:,None]).sum(axis=1) if len(scores) > 0 else np.zeros(resolution)
        precision_table = np.ones(resolution)
        precision_table[misses_table > 0] = 0
        recall_table = np.ones(resolution)
        PR_table = np.stack([precision_table, recall_table], axis=1)
        if single_result:
            return PR_table
        else:
            return np.stack([PR_table.copy() for _ in subsets])

    if len(bboxes_pr) == 0:
        precision_table = np.ones(resolution)
        recall_table = np.zeros(resolution)
        PR_table = np.stack([precision_table, recall_table], axis=1)
        if single_result:
            return PR_table
        else:
            return np.stack([PR_table.copy() for _ in subsets])


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

    misses_table = (miss_scores[None,:] >= grid[:,None]).sum(axis=1)

    PR_tables = []
    for subset in subsets:
        _hit_scores = hit_scores[subset]
        _matched_scores = scores[np.isin(best_match, subset)]

        _hits_table = (_hit_scores[None,:] >= grid[:,None]).sum(axis=1)
        _shots_table = misses_table + (_matched_scores[None,:] >= grid[:,None]).sum(axis=1)

        _precision_table = np.ones(resolution)
        np.true_divide(_hits_table, _shots_table, out=_precision_table, where=(_shots_table>0))
        _recall_table = _hits_table / len(subset) if len(subset) > 0 else np.ones(resolution)

        _PR_table = np.stack([_precision_table, _recall_table], axis=1)
        PR_tables.append(_PR_table)

    if single_result:
        return PR_tables[0]
    else:
        return np.stack(PR_tables)

def calculate_AP(PR_table):
    PR_table = np.concatenate([PR_table, np.array([[1,0]])], axis=0)
    traps = np.concatenate([PR_table[:-1], PR_table[1:]], axis=1)
    areas = np.minimum(traps[:,1], traps[:,3])*np.abs(traps[:,0] - traps[:,2])
    return areas.sum()
