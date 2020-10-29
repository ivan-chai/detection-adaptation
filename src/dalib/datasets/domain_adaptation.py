import bisect

from torch.utils.data import ConcatDataset


class DomainAdaptationDataset(ConcatDataset):
    """Dataset as a concatenation of multiple datasets with dataset idx as domain label

    This class is useful to assemble datasets from different domains.

    If :meth:`__getitem__` of original datasets returns a tuple
    :class:`dalib.utils.datasets.DomainAdaptationDataset` will return a tuple
    concateneted with domain label based on dataset index in list of datasets.

    If :meth:`__getitem__` of original datasets returns a dict
    :class:`dalib.utils.datasets.DomainAdaptationDataset` will return a dict
    with "domain_label" key.

    Args:
        datasets (list or tuple): List of datasets to be concatenated
    """

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        item = self.__add_domain_label(
            self.datasets[dataset_idx][sample_idx],
            dataset_idx
        )

        return item

    def __add_domain_label(self, item, domain_label):
        if isinstance(item, tuple):
            item = item + (domain_label,)
        elif isinstance(item, dict):
            item["domain_label"] = domain_label
        else:
            raise NotImplementedError
        return item
