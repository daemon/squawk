from dataclasses import dataclass
from typing import Sequence
import random

from torchaudio.transforms import MelSpectrogram, ComputeDeltas
import torch
import torch.nn as nn

from .dataset import ClassificationExample, ClassificationBatch
from squawk.data.spec_augment_tensorflow import sparse_warp


# Written as a class for multiprocessing serialization
class Composition(object):
    def __init__(self, modules):
        self.modules = modules

    def __call__(self, *args):
        for mod in self.modules:
            args = mod(*args)
            args = (args,)
        return args[0]


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


class IdentityTransform(nn.Module):
    def forward(self, x):
        return x


@dataclass
class SpecAugmentConfig(object):
    W: int = 80
    F: int = 40
    mF: int = 2
    T: int = 100
    p: float = 1.0
    mT: int = 2

class SpecAugmentTransform(nn.Module):

    def __init__(self, config: SpecAugmentConfig = SpecAugmentConfig()):
        super().__init__()
        self.config = config

    def timewarp(self, x):
        x = x.permute(0, 2, 1).contiguous().unsqueeze(-1)
        x = torch.from_numpy(sparse_warp(x.cpu().numpy(), self.config.W)).to(x.device).squeeze(-1).permute(0, 2, 1).contiguous()
        return x

    def tmask(self, x):
        for idx in range(x.size(0)):
            t = random.randrange(0, self.config.T)
            t0 = random.randrange(0, x.size(2) - t)
            x[idx, :, t0:t0 + t] = 0
        return x

    def fmask(self, x):
        for idx in range(x.size(0)):
            f = random.randrange(0, self.config.F)
            f0 = random.randrange(0, x.size(1) - f)
            x[idx, f0:f0 + f] = 0
        return x

    def forward(self, x):
        with torch.no_grad():
            x = torch.cat([self.timewarp(y) for y in x.squeeze(1).split(1, 0)], 0)
            x = self.tmask(x)
            x = self.fmask(x)
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


class StandardAudioTransform(nn.Module):

    def __init__(self, sample_rate=44100, n_fft=int(400 / 16000 * 44100)):
        super().__init__()
        self.spec_transform = MelSpectrogram(n_mels=80, sample_rate=sample_rate, n_fft=n_fft)
        self.delta_transform = ComputeDeltas()

    def forward(self, audio: torch.Tensor, mels_only=False, deltas_only=False):
        with torch.no_grad():
            log_mels = audio if deltas_only else self.spec_transform(audio).add_(1e-7).log_()
            if mels_only:
                return log_mels
            deltas = self.delta_transform(log_mels)
            accels = self.delta_transform(deltas)
            return torch.stack((log_mels, deltas, accels), 1)

    def compute_length(self, length: int):
        return int((length - self.spec_transform.win_length) / self.spec_transform.hop_length + 1)
