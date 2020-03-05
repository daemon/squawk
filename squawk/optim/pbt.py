from dataclasses import dataclass, field
from typing import Dict, Sequence, List
import json
import pathlib
import os
import random
import time

import numpy as np
import torch

from squawk.utils import FileLock
from squawk.data import AugmentationParameter


class PbaMetaOptimizer(object):

    def __init__(self,
                 weight_path: str,
                 augment_ops: Sequence[AugmentationParameter],
                 resample_prob=0.2,
                 index_amount_perturb=(0, 1, 2, 3),
                 datafile_path='.pba-meta.json',
                 quality=-100000,
                 step_no=0,
                 id=None,
                 exploit_epochs=3,
                 lock=None,
                 metadata=None):
        self.augment_ops = augment_ops
        self.resample_prob = resample_prob
        self.index_amount_perturb = index_amount_perturb
        self.weight_path = weight_path
        for x in augment_ops:
            if x.current_value_idx is None:
                x.current_value_idx = random.choice(list(range(len(x.domain))))
        self.quality = quality
        self.step_no = step_no
        self.exploit_epochs = exploit_epochs
        self.datafile_path = pathlib.Path(datafile_path)
        self.lock_file = self.datafile_path.with_suffix('.lock')
        with self.lock(grab=lock is None) as lock:
            if metadata is None:
                try:
                    self.metadata = PbaMetadata.load(self.datafile_path, lock=lock)
                except FileNotFoundError:
                    self.metadata = PbaMetadata()
            else:
                self.metadata = metadata
            self.id = self.metadata.last_id if id is None else id
            if metadata is None:
                self.save(lock=lock, increment=id is None)

    def lock(self, grab=True):
        return FileLock(self.lock_file, grab=grab)

    def dict(self):
        return dict(quality=self.quality,
                    step_no=self.step_no,
                    datafile_path=str(self.datafile_path),
                    index_amount_perturb=self.index_amount_perturb,
                    resample_prob=self.resample_prob,
                    exploit_epochs=self.exploit_epochs,
                    weight_path=self.weight_path,
                    id=self.id,
                    augment_ops=[x.__dict__ for x in self.augment_ops])

    def fence(self, lock):
        while any(opt.step_no < self.step_no for opt in self.metadata.optimizers):
            lock.unlock()
            time.sleep(5)
            lock.lock()
            self.metadata = PbaMetadata.load(self.datafile_path, lock=lock)

    def load_fence(self, lock):
        self.save(lock=lock, inc_load_fence=True)
        while self.metadata.load_fence < self.metadata.curr_count and self.metadata.load_fence != 0:
            lock.unlock()
            time.sleep(5)
            lock.lock()
            self.metadata = PbaMetadata.load(self.datafile_path, lock=lock)
        if self.metadata.load_fence != 0:
            self.save(lock=lock, clear_load_fence=True)

    def step(self, quality: float, load_callback=None, explore=True):
        self.quality = quality
        self.step_no += 1
        self.save()
        if self.step_no >= self.exploit_epochs:
            with self.lock() as lock:
                self.fence(lock)
                self.metadata = PbaMetadata.load(self.datafile_path, lock=lock)
                opt_perfs = [opt.quality for opt in self.metadata.optimizers]
                if self.quality <= np.quantile(opt_perfs, 0.25):
                    q75 = np.quantile(opt_perfs, 0.75)
                    top25_opt = random.choice(list(filter(lambda opt: opt.quality >= q75, self.metadata.optimizers)))
                    for op1, op2 in zip(self.augment_ops, top25_opt.augment_ops): op1.copy_from(op2)
                    self.quality = top25_opt.quality
                    if load_callback: load_callback(torch.load(top25_opt.weight_path, lambda s, l: s))
                self.load_fence(lock)
        if explore:
            self.explore()

    def explore(self):
        for param in self.augment_ops:
            if random.random() < 0.2:
                param.current_value_idx = random.choice(list(range(len(param.domain))))
            else:
                amount = random.choice([0, 1, 2, 3])
                if random.random() < 0.5:
                    param.current_value_idx += amount
                else:
                    param.current_value_idx -= amount
                param.current_value_idx = max(min(param.current_value_idx, len(param.domain) - 1), 0)
            if random.random() < 0.2:
                param.prob = random.choice(np.linspace(0, 1, 11))
            else:
                param.prob += random.choice(np.linspace(-0.3, 0.3, 7))
                param.prob = max(min(param.prob, 1), 0)

    def sample(self):
        count = np.random.choice([0, 1, 2], p=[0.2, 0.3, 0.5])
        params = self.augment_ops
        for idx in np.random.permutation(list(range(len(params)))):
            param = params[idx]
            param.enabled = False
            if random.random() < param.prob and count > 0:
                param.enabled = True
                count -= 1

    def __enter__(self, *args):
        pass

    def __exit__(self, *args):
        self.cleanup()

    def cleanup(self):
        self.save(decrement=True)
        if self.metadata.curr_count == 0:
            os.remove(str(self.datafile_path))
            os.remove(str(self.lock_file))

    def save(self, lock=None, increment=False, decrement=False, inc_load_fence=False, clear_load_fence=False):
        with self.lock(grab=lock is None) as lock:
            try:
                self.metadata = metadata = PbaMetadata.load(self.datafile_path, lock=lock)
            except FileNotFoundError:
                metadata = self.metadata
            inc = 0
            if increment:
                inc = 1
            if decrement:
                inc = -1
            metadata.curr_count += inc
            metadata.last_id += inc
            if inc_load_fence:
                metadata.load_fence += 1
            if clear_load_fence:
                metadata.load_fence = 0
            idx = next((idx for idx, opt in enumerate(metadata.optimizers) if opt.id == self.id), None)
            if idx is None:
                metadata.optimizers.append(self)
            else:
                metadata.optimizers[idx] = self
            metadata.save(self.datafile_path, lock=lock)

    @classmethod
    def from_dict(cls, data_dict, lock=None, metadata=None):
        return cls(data_dict['weight_path'],
                   [AugmentationParameter.from_dict(x) for x in data_dict['augment_ops']],
                   resample_prob=data_dict['resample_prob'],
                   quality=data_dict['quality'],
                   step_no=data_dict['step_no'],
                   datafile_path=data_dict['datafile_path'],
                   index_amount_perturb=data_dict['index_amount_perturb'],
                   exploit_epochs=data_dict['exploit_epochs'],
                   lock=lock,
                   metadata=metadata,
                   id=data_dict.get('id'))


@dataclass
class PbaMetadata(object):
    last_id: int = 0
    curr_count: int = 0
    load_fence: int = 0
    optimizers: List[PbaMetaOptimizer] = field(default_factory=list)

    def dict(self):
        return dict(last_id=self.last_id,
                    curr_count=self.curr_count,
                    load_fence=self.load_fence,
                    optimizers=[x.dict() for x in self.optimizers])

    @classmethod
    def from_dict(cls, data_dict, lock=None):
        metadata = cls(data_dict['last_id'], data_dict['curr_count'], data_dict['load_fence'])
        metadata.optimizers = [PbaMetaOptimizer.from_dict(v, lock=lock, metadata=metadata) for v in data_dict['optimizers']]
        return metadata

    @classmethod
    def load(cls, path: pathlib.Path, lock=None):
        with FileLock(path.with_suffix('.lock'), grab=lock is None) as lock, open(str(path)) as f:
            return cls.from_dict(json.load(f), lock=lock)

    def save(self, path: pathlib.Path, lock=None):
        with FileLock(path.with_suffix('.lock'), grab=lock is None), open(str(path), 'w') as f:
            json.dump(self.dict(), f)
