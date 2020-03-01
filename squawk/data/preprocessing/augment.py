from dataclasses import dataclass
from typing import Sequence, Mapping
import math
import random

from torchaudio.transforms import MelSpectrogram, ComputeDeltas
import numpy as np
import librosa
import torch
import torch.nn as nn

from squawk.data.dataset import ClassificationExample
from squawk.data.preprocessing.spec_augment_tensorflow import sparse_warp


@dataclass
class AugmentationParameter(object):
    domain: Sequence[float]
    name: str
    current_value_idx: int = None
    prob: float = 0.2
    enabled: bool = True

    @property
    def magnitude(self):
        return self.domain[self.current_value_idx]

    @classmethod
    def from_dict(cls, data_dict):
        return cls(data_dict['domain'], data_dict['name'], data_dict['current_value_idx'], data_dict['prob'])


class AugmentModule(nn.Module):

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> Sequence[AugmentationParameter]:
        raise NotImplementedError

    def augment(self, param: AugmentationParameter, x, **kwargs):
        raise NotImplementedError

    def passthrough(self, x, **kwargs):
        return x

    def forward(self, x, augment_params: Sequence[AugmentationParameter] = None, **kwargs):
        if augment_params is None:
            augment_params = self.default_params
        for param in augment_params:
            x = self.augment(param, x, **kwargs) if param.enabled else self.passthrough(x, **kwargs)
        return x


class TimeshiftTransform(AugmentModule):

    def __init__(self, sr=44100):
        super().__init__()
        self.sr = sr

    @property
    def default_params(self):
        return AugmentationParameter([0.25, 0.5, 0.75, 1], 'timeshift', 2),

    def augment(self, param: AugmentationParameter, examples: Sequence[ClassificationExample], **kwargs):
        if not self.training:
            return examples
        new_examples = []
        for example in examples:
            label = example.label
            w = min(int(random.random() * param.magnitude * self.sr), int(0.5 * example.audio.size(1)))
            audio = example.audio[:, w:] if random.random() < 0.5 else example.audio[:, :example.audio.size(1) - w]
            new_examples.append(ClassificationExample(audio, label))
        return new_examples


class TimestretchTransform(AugmentModule):

    def __init__(self):
        super().__init__()

    @property
    def default_params(self):
        return AugmentationParameter([0.01, 0.025, 0.05, 0.075, 0.1], 'timestretch', 2),

    def augment(self, param, examples: Sequence[ClassificationExample], **kwargs):
        if not self.training:
            return examples
        rate = np.clip(np.random.normal(1, param.magnitude), 0.75, 1.25)
        new_examples = []
        for example in examples:
            audio = torch.from_numpy(librosa.effects.time_stretch(example.audio.squeeze().cpu().numpy(), rate))
            audio = audio.unsqueeze(0)
            new_examples.append(ClassificationExample(audio, example.label))
        return new_examples


class NoiseTransform(AugmentModule):

    def __init__(self):
        super().__init__()

    @property
    def default_params(self):
        return AugmentationParameter([0.0001, 0.00025, 0.0005, 0.001, 0.002], 'white', 3),\
               AugmentationParameter([1 / 60000, 1 / 45000, 1 / 30000, 1 / 15000, 1 / 7500], 'salt_pepper', 2)

    def augment(self, param, waveform, lengths=None):
        if not self.training:
            return waveform
        if param.name == 'white':
            strength = param.magnitude * random.random()
            noise_mask = torch.empty_like(waveform).normal_(0, strength)
        else:
            prob = param.magnitude * random.random()
            noise_mask = torch.empty_like(waveform).bernoulli_(prob / 2) - torch.empty_like(waveform).bernoulli_(prob / 2)
        noise_mask.clamp_(-1, 1)
        for idx, length in enumerate(lengths.tolist()):
            waveform[idx, :length] = (waveform[idx, :length] + noise_mask[idx, :length]).clamp_(-1, 1)
        return waveform


@dataclass
class SpecAugmentConfig(object):
    W: int = 80
    F: int = 40
    mF: int = 2
    T: int = 100
    p: float = 1.0
    mT: int = 2
    use_timewarp: bool = True


class StandardAudioTransform(AugmentModule):

    def __init__(self, sample_rate=44100, n_fft=int(400 / 16000 * 44100)):
        super().__init__()
        self.spec_transform = MelSpectrogram(n_mels=80, sample_rate=sample_rate, n_fft=n_fft)
        self.vtlp_transform = apply_vtlp(MelSpectrogram(n_mels=80, sample_rate=sample_rate, n_fft=n_fft))
        self.delta_transform = ComputeDeltas()

    @property
    def default_params(self) -> Sequence[AugmentationParameter]:
        return AugmentationParameter([0], 'vtlp', 0),

    def _execute_op(self, op, audio, mels_only=False, deltas_only=False):
        with torch.no_grad():
            log_mels = audio if deltas_only else op(audio).add_(1e-7).log_()
            if mels_only:
                return log_mels
            deltas = self.delta_transform(log_mels)
            accels = self.delta_transform(deltas)
            return torch.stack((log_mels, deltas, accels), 1)

    def augment(self, param, audio: torch.Tensor, **kwargs):
        return self._execute_op(self.vtlp_transform, audio, **kwargs)

    def passthrough(self, audio: torch.Tensor, **kwargs):
        return self._execute_op(self.spec_transform, audio, **kwargs)

    def compute_length(self, length: int):
        return int((length - self.spec_transform.win_length) / self.spec_transform.hop_length + 1)


class SpecAugmentTransform(AugmentModule):

    def __init__(self, use_timewarp=False):
        super().__init__()
        self.use_timewarp = use_timewarp

    @property
    def default_params(self) -> Sequence[AugmentationParameter]:
        return AugmentationParameter([40, 80, 160, 240, 320], 'sa_warp', 1),\
               AugmentationParameter([20, 30, 40, 50, 60], 'sa_freq', 2),\
               AugmentationParameter([50, 75, 100, 125, 150], 'sa_time', 2)

    def timewarp(self, x, W):
        x = x.permute(0, 2, 1).contiguous().unsqueeze(-1)
        x = torch.from_numpy(sparse_warp(x.cpu().numpy(), W)).to(x.device).squeeze(-1).permute(0, 2, 1).contiguous()
        return x

    def tmask(self, x, T):
        for idx in range(x.size(0)):
            t = random.randrange(0, T)
            t0 = random.randrange(0, x.size(2) - t)
            x[idx, :, t0:t0 + t] = 0
        return x

    def fmask(self, x, F):
        for idx in range(x.size(0)):
            f = random.randrange(0, F)
            f0 = random.randrange(0, x.size(1) - f)
            x[idx, f0:f0 + f] = 0
        return x

    def augment(self, param, x, **kwargs):
        with torch.no_grad():
            if self.use_timewarp and param.name == 'sa_warp':
                return torch.cat([self.timewarp(y, param.magnitude) for y in x.squeeze(1).split(1, 0)], 0)
            if param.name == 'sa_freq':
                return self.fmask(x, param.magnitude)
            elif param.name == 'sa_time':
                return self.tmask(x, param.magnitude)
        return x


def create_vtlp_fb_matrix(n_freqs, f_min, f_max, n_mels, sample_rate, alpha, f_hi=4800, training=True):
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
