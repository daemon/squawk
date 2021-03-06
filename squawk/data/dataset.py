from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.utils.data as tud
import torchaudio

from squawk.ctqdm import ctqdm


@dataclass
class DatasetInfo(object):
    name: str
    sample_rate: int
    label_map: Mapping[str, int]

    def __post_init__(self):
        self.num_labels = len(self.label_map)


class LruCache(object):

    def __init__(self, capacity=np.inf, load_fn=None):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.load_fn = load_fn

    def __getitem__(self, key):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
        except KeyError:
            if self.load_fn is not None:
                self[key] = value = self.load_fn(key)
            else:
                raise KeyError
        return value

    def __setitem__(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value


@dataclass
class ClassificationExample(object):
    audio: torch.Tensor
    label: int


@dataclass
class ClassificationBatch(object):
    audio: torch.Tensor
    labels: torch.LongTensor
    lengths: torch.LongTensor = None

    def to(self, device):
        self.audio = self.audio.to(device)
        self.labels = self.labels.to(device)
        self.lengths = self.lengths.to(device)

    def pin_memory(self):
        self.audio = self.audio.pin_memory()
        self.labels = self.labels.pin_memory()
        if self.lengths is not None: self.lengths = self.lengths.pin_memory()
        return self


@dataclass
class ClassificationDataset(tud.Dataset):
    audio_data: Sequence[str]
    label_data: Sequence[int]
    info: DatasetInfo
    lru_cache: LruCache

    def __post_init__(self):
        self.lru_cache.load_fn = self.load

    def __len__(self):
        return len(self.audio_data)

    def load(self, idx):
        return torchaudio.load(self.audio_data[idx])[0]

    def split(self, proportion):
        proportion = int(proportion * len(self.audio_data))
        audio_data1 = self.audio_data[:proportion]
        audio_data2 = self.audio_data[proportion:]
        label_data1 = self.label_data[:proportion]
        label_data2 = self.label_data[proportion:]
        return ClassificationDataset(audio_data1, label_data1, self.info, LruCache(self.lru_cache.capacity)),\
               ClassificationDataset(audio_data2, label_data2, self.info, LruCache(self.lru_cache.capacity))

    def __getitem__(self, idx):
        return ClassificationExample(self.lru_cache[idx], self.label_data[idx])


def load_gsc(folder: Path, lru_maxsize=np.inf):
    def load_split(name):
        dev_path = folder / 'validation_list.txt'
        test_path = folder / 'testing_list.txt'
        labels = []
        for x in folder.glob('*'):
            if x.is_dir():
                labels.append(x.name)
        labels = sorted(labels)
        l2idx = {x: idx for idx, x in enumerate(labels)}
        with open(dev_path) as f:
            dev_set = list(map(str.strip, f.readlines()))
        with open(test_path) as f:
            test_set = list(map(str.strip, f.readlines()))
        if name == 'dev':
            tgt_set = dev_set
        elif name == 'test':
            tgt_set = test_set
        else:
            all_set = set(f'{str(x.parent.name)}/{str(x.name)}' for x in folder.glob('*/*.wav'))
            dev_set = set(dev_set)
            test_set = set(test_set)
            tgt_set = (all_set - dev_set) - test_set
        return ClassificationDataset([f'{str(folder)}/{x}' for x in tgt_set],
                                     [l2idx[x.split('/')[0]] for x in tgt_set],
                                     DatasetInfo('GSC', 16000, l2idx),
                                     LruCache(lru_maxsize))
    return load_split('training'), load_split('dev'), load_split('test')


def load_freesounds(folder: Path, lru_maxsize=np.inf):
    def load_split(name):
        train_csv_path = folder / 'FSDKaggle2018.meta' / f'train_post_competition.csv'
        labels_csv_path = folder / 'FSDKaggle2018.meta' / f'{name}_post_competition{"_scoring_clips" if name == "test" else ""}.csv'
        df = pd.read_csv(labels_csv_path, quoting=3)
        train_df = pd.read_csv(train_csv_path, quoting=3)
        idx2l = sorted(list(set(train_df['label'].unique())))
        l2idx = {lbl: idx for idx, lbl in enumerate(idx2l)}
        label_map = {}
        for row in df[['fname', 'label']].itertuples():
            label_map[row.fname] = l2idx[row.label]

        data_folder = folder / f'FSDKaggle2018.audio_{name}'
        audio_data = []
        label_data = []
        for wav_file in ctqdm(list(data_folder.glob('*.wav')), desc=f'Preparing {name} dataset'):
            audio_data.append(str(wav_file))
            label_data.append(label_map[wav_file.name])
        _, sr = torchaudio.load(wav_file)
        return ClassificationDataset(audio_data, label_data, DatasetInfo('FreeSounds', sr, l2idx), LruCache(lru_maxsize))
    train_split, dev_split = load_split('train').split(0.9)
    test_split = load_split('test')
    return train_split, dev_split, test_split
