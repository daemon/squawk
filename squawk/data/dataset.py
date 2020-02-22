from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

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


@dataclass
class ClassificationExample(object):
    audio: torch.Tensor
    label: int


@dataclass
class ClassificationBatch(object):
    audio: torch.Tensor
    labels: torch.LongTensor
    lengths: torch.LongTensor = None

    def cuda(self):
        self.audio = self.audio.cuda()
        self.labels = self.labels.cuda()
        self.lengths = self.lengths.cuda()
        return self

    def pin_memory(self):
        self.audio = self.audio.pin_memory()
        self.labels = self.labels.pin_memory()
        if self.lengths is not None: self.lengths = self.lengths.pin_memory()
        return self


@dataclass
class ClassificationDataset(tud.Dataset):
    audio_data: Sequence[torch.Tensor]
    label_data: Sequence[int]
    info: DatasetInfo

    def __len__(self):
        return len(self.audio_data)

    def split(self, proportion):
        proportion = int(proportion * len(self.audio_data))
        audio_data1 = self.audio_data[:proportion]
        audio_data2 = self.audio_data[proportion:]
        label_data1 = self.label_data[:proportion]
        label_data2 = self.label_data[proportion:]
        return ClassificationDataset(audio_data1, label_data1, self.info), ClassificationDataset(audio_data2, label_data2, self.info)

    def to(self, device):
        self.audio_data = [x.to(device) for x in self.audio_data]

    def __getitem__(self, idx):
        return ClassificationExample(self.audio_data[idx], self.label_data[idx])


def load_freesounds(folder: Path):
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
        for wav_file in ctqdm(list(data_folder.glob('*.wav'))):
            audio, sr = torchaudio.load(wav_file)
            audio_data.append(audio)
            label_data.append(label_map[wav_file.name])
        return ClassificationDataset(audio_data, label_data, DatasetInfo('FreeSounds', sr, l2idx))
    train_split, dev_split = load_split('train').split(0.9)
    return train_split, dev_split, load_split('test')
