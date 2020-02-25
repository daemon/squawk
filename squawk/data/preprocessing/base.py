from typing import Sequence

import torch
import torch.nn as nn

from squawk.data.dataset import ClassificationExample, ClassificationBatch


# Written as a class for multiprocessing serialization
class Composition(object):

    def __init__(self, modules):
        self.modules = modules

    def __call__(self, *args):
        for mod in self.modules:
            args = mod(*args)
            args = (args,)
        return args[0]


class IdentityTransform(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def compose(*collate_modules):
    return Composition(collate_modules)


def batchify(examples: Sequence[ClassificationExample], max_length: int = None, pad_silence=True):
    examples = sorted(examples, key=lambda x: x.audio.size()[-1], reverse=True)
    lengths = torch.tensor([ex.audio.size(-1) for ex in examples])
    if pad_silence:
        if max_length is None: max_length = max(ex.audio.size()[-1] for ex in examples)
        examples = [ClassificationExample(torch.cat((ex.audio.squeeze(), torch.zeros(max_length - ex.audio.size(-1))), -1),
                                          ex.label) for ex in examples]
    batch = ClassificationBatch(torch.stack([ex.audio for ex in examples]),
                                torch.tensor([ex.label for ex in examples]),
                                lengths)
    return batch


def move_cuda(batch: ClassificationBatch):
    batch.audio = batch.audio.cuda()
    batch.label = batch.label.cuda()
    return batch


def identity(x):
    return x


class ZmuvTransform(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('total', torch.zeros(1))
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('mean2', torch.zeros(1))

    def update(self, data):
        self.mean = (data.sum() + self.mean * self.total) / (self.total + data.numel())
        self.mean2 = ((data ** 2).sum() + self.mean2 * self.total) / (self.total + data.numel())
        self.total += data.numel()

    @property
    def std(self):
        return (self.mean2 - self.mean ** 2).sqrt()

    def forward(self, x):
        return (x - self.mean) / self.std
