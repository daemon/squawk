from dataclasses import dataclass
from typing import Sequence
import math
import random

from torchaudio.transforms import MelSpectrogram, ComputeDeltas
import torch
import torch.nn as nn

from squawk.data.dataset import ClassificationExample
from squawk.data.preprocessing.spec_augment_tensorflow import sparse_warp


def timeshift(examples: Sequence[ClassificationExample], W=0.8, sr=44100, p=0.5):
    new_examples = []
    for example in examples:
        label = example.label
        w = min(int(random.random() * W * sr), int(p * example.audio.size(1)))
        audio = example.audio[:, w:] if random.random() < 0.5 else example.audio[:, :example.audio.size(1) - w]
        new_examples.append(ClassificationExample(audio, label))
    return new_examples


@dataclass
class SpecAugmentConfig(object):
    W: int = 80
    F: int = 40
    mF: int = 2
    T: int = 100
    p: float = 1.0
    mT: int = 2
    use_timewarp: bool = True


class StandardAudioTransform(nn.Module):

    def __init__(self, sample_rate=44100, n_fft=int(400 / 16000 * 44100), use_vtlp=False):
        super().__init__()
        self.spec_transform = MelSpectrogram(n_mels=80, sample_rate=sample_rate, n_fft=n_fft)
        if use_vtlp:
            self.spec_transform = apply_vtlp(self.spec_transform)
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
            if self.config.use_timewarp:
                x = torch.cat([self.timewarp(y) for y in x.squeeze(1).split(1, 0)], 0)
            x = self.tmask(x)
            x = self.fmask(x)
        return x


def create_vtlp_fb_matrix(n_freqs, f_min, f_max, n_mels, sample_rate, alpha, f_hi=4800 / 16000, training=True):
    # type: (int, float, float, int, int, float, int, bool) -> torch.Tensor
    # freq bins
    # Equivalent filterbank construction by Librosa
    S = sample_rate
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    # hertz to mel(f) is 2595. * math.log10(1. + (f / 700.))
    m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
    m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    # mel to hertz(mel) is 700. * (10**(mel / 2595.) - 1.)
    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    if training:
        f_hi = int(f_hi * sample_rate)
        f_pts[f_pts <= f_hi * min(alpha, 1) / alpha] *= alpha
        f = f_pts[f_pts > f_hi * min(alpha, 1) / alpha]
        f_pts[f_pts > f_hi * min(alpha, 1) / alpha] = S / 2 - ((S / 2 - f_hi * min(alpha, 1)) /
                                                               (S / 2 - f_hi * min(alpha, 1) / alpha)) * (S / 2 - f)
    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))
    return fb


class VtlpMelScale(nn.Module):

    __constants__ = ['n_mels', 'sample_rate', 'f_min', 'f_max']

    def __init__(self, n_mels=128, sample_rate=16000, f_min=0., f_max=None, n_stft=None):
        super().__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min

        assert f_min <= self.f_max, 'Require f_min: %f < f_max: %f' % (f_min, self.f_max)

    def forward(self, specgram):
        # pack batch
        shape = specgram.size()
        specgram = specgram.reshape(-1, shape[-2], shape[-1])

        fb = create_vtlp_fb_matrix(specgram.size(1), self.f_min, self.f_max, self.n_mels, self.sample_rate,
                                   random.random() * 0.2 + 0.9, training=self.training).to(specgram.device)
        # (channel, frequency, time).transpose(...) dot (frequency, n_mels)
        # -> (channel, time, n_mels).transpose(...)
        mel_specgram = torch.matmul(specgram.transpose(1, 2), fb).transpose(1, 2)
        # unpack batch
        mel_specgram = mel_specgram.reshape(shape[:-2] + mel_specgram.shape[-2:])
        return mel_specgram


def apply_vtlp(mel_spectrogram: MelSpectrogram):
    s = mel_spectrogram
    s.mel_scale = VtlpMelScale(s.n_mels, s.sample_rate, s.f_min, s.f_max, s.n_fft // 2 + 1)
    return s
