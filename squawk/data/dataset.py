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
    sample_rate: int
    label_mapper: Mapping[str, int]


@dataclass
class ClassificationExample(object):
    audio: torch.Tensor
    label: int


@dataclass
class ClassificationDataset(tud.Dataset):
    audio_data: Sequence[torch.Tensor]
    label_data: Sequence[int]
    info: DatasetInfo

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        return ClassificationExample(self.audio_data[idx], self.label_data[idx])


def load_freesounds(folder: Path):
    labels_csv_path = folder / 'FSDKaggle2018.meta' / 'train_post_competition.csv'
    df = pd.read_csv(labels_csv_path, quoting=3)
    idx2l = sorted(list(set(df['label'].unique())))
    l2idx = {lbl: idx for idx, lbl in enumerate(idx2l)}
    label_map = {}
    for row in df[['fname', 'label']].itertuples():
        label_map[row.fname] = l2idx[row.label]

    train_csv_folder = folder / 'FSDKaggle2018.audio_train'
    audio_data = []
    label_data = []
    for wav_file in ctqdm(list(train_csv_folder.glob('*.wav'))):
        audio, sr = torchaudio.load(wav_file)
        audio_data.append(audio)
        label_data.append(label_map[wav_file.name])
    return ClassificationDataset(audio_data, label_data, DatasetInfo(sr, l2idx))
