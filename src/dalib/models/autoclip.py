import torch
from torch import nn

import numpy as np

from collections import deque

class AutoClip:
    """Track grad norm history and clip gradients to given percentile.
    See arXiv:2007.14469.
    """

    def __init__(self, percentile, history_size=100, min_history_size=10):
        self.history = deque(maxlen=history_size)
        self.percentile = percentile

    def __call__(self, parameters):
        max_norm = float("inf") if len(self.history) < 10 else np.percentile(self.history, self.percentile)
        grad_norm = nn.utils.clip_grad_norm_(parameters, max_norm)
        self.history.append(grad_norm)
        clipped_grad_norm = min(grad_norm, max_norm)
        return grad_norm, clipped_grad_norm
