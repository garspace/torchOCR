import os
import torch

from .base_dataset import BaseDataSet
from .simple_dataset import SimpleDataSet
from utils.torch_utils import torch_distributed_zero_first

class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def build_dataloader(cfg, mode, batch_size, rank=-1):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    support_dict = [
        'BaseDataSet', 'SimpleDataSet', 'LMDBDataSet', 'PGDataSet', 'PubTabDataSet'
    ]

    with torch_distributed_zero_first(rank):
        assert cfg[mode]['name'] in support_dict
        dataset = eval(cfg[mode]['name'])(cfg[mode], mode=mode)

    workers = cfg[mode]['workers']
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True)
    return dataloader, dataset