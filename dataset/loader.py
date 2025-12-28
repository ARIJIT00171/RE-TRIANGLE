import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.distributed import any_broadcast


class MetaLoader(object):
    """ wraps multiple dataset loaders """

    def __init__(self, loaders, accum_steps=1, distributed=False):
        assert isinstance(loaders, dict)
        self.name2loader = {}
        self.name2iter = {}
        self.sampling_pools = []
        self.name2labelname = {}
        for idx, (n, l) in enumerate(loaders.items()):
            if isinstance(l, tuple):
                l, r = l
            elif isinstance(l, DataLoader):
                r = 1
            else:
                raise ValueError()
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.sampling_pools.extend([n] * r)

        self.accum_steps = accum_steps
        self.distributed = distributed
        self.step = 0
        self.epoch = 0

    def __iter__(self):
        """ this iterator will run indefinitely """
        task = self.sampling_pools[0]
        while True:
            if self.step % self.accum_steps == 0:
                task = random.choice(self.sampling_pools)
                if self.distributed:
                    task = any_broadcast(task, 0)
            self.step += 1
            iter_ = self.name2iter[task]
            try:
                batch = next(iter_)
            except StopIteration:
                self.epoch = self.epoch + 1
                if isinstance(self.name2loader[task].sampler, DistributedSampler):
                    self.name2loader[task].sampler.set_epoch(self.epoch)
                else:
                    pass
                iter_ = iter(self.name2loader[task])
                batch = next(iter_)
                self.name2iter[task] = iter_

            yield task, batch


def move_to_cuda(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t, device) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t, device) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t, device) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def record_cuda_stream(batch):
    if isinstance(batch, torch.Tensor):
        if batch.is_cuda:
            batch.record_stream(torch.cuda.current_stream())
    elif isinstance(batch, list) or isinstance(batch, tuple):
        for t in batch:
            record_cuda_stream(t)
    elif isinstance(batch, dict):
        for t in batch.values():
            record_cuda_stream(t)
    else:
        pass


class PrefetchLoader(object):
    """
    overlap compute and cuda dataset transfer
    (copied and then modified from nvidia apex)
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=self.device)

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return

        with torch.cuda.stream(self.stream):
            self.batch = move_to_cuda(self.batch, self.device)

    def next(self, it):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            record_cuda_stream(batch)
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method
