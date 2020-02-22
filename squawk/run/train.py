from itertools import chain
from pathlib import Path
import argparse

from tqdm import trange, tqdm
from torch.optim.adamw import AdamW
import torch
import torch.nn as nn
import torch.utils.data as tud

from squawk.data import load_freesounds, batchify, StandardAudioTransform, ZmuvTransform, SpecAugmentTransform
from squawk.model import LASClassifier, LASClassifierConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=Path)
    parser.add_argument('--num-epochs', '-ne', type=int, default=20)
    parser.add_argument('--batch-size', '-bsz', type=int, default=16)
    parser.add_argument('--weight-decay', '-wd', type=float, default=5e-5)
    args = parser.parse_args()
    train_ds, dev_ds, test_ds = load_freesounds(args.dir)
    train_loader = tud.DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, shuffle=True, collate_fn=batchify, num_workers=16, drop_last=True)
    dev_loader = tud.DataLoader(dev_ds, batch_size=10, pin_memory=True, shuffle=False, collate_fn=batchify, num_workers=10)
    spec_transform = StandardAudioTransform()
    spec_transform.cuda()

    model = LASClassifier(LASClassifierConfig(train_ds.info.num_labels)).cuda()
    zmuv_transform = ZmuvTransform().cuda()
    sa_transform = SpecAugmentTransform().cuda()
    criterion = nn.CrossEntropyLoss()
    params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = AdamW(params, 1e-3, weight_decay=args.weight_decay)

    for ex in tqdm(chain(train_ds, dev_ds, test_ds), desc='Constructing ZMUV'):
        zmuv_transform.update(spec_transform(ex.audio.cuda(), mels_only=True))

    for epoch_idx in trange(args.num_epochs, position=0, leave=True):
        model.train()
        pbar = tqdm(train_loader, total=len(train_loader), position=1, desc='Training', leave=True)
        for batch in pbar:
            batch.cuda()
            audio = spec_transform(sa_transform(zmuv_transform(spec_transform(batch.audio, mels_only=True))), deltas_only=True)
            lengths = batch.lengths.clone()
            for idx, x in enumerate(lengths):
                lengths[idx] = spec_transform.compute_length(x)
            scores = model(audio, lengths)
            optimizer.zero_grad()
            loss = criterion(scores, batch.labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(dict(loss=f'{loss.item():.3}'))

        model.eval()
        pbar = tqdm(dev_loader, total=len(dev_loader), position=1, desc='Testing', leave=True)
        num_correct = 0
        num_total = 0
        for batch in pbar:
            batch.cuda()
            lengths = batch.lengths.clone()
            for idx, x in enumerate(lengths):
                lengths[idx] = spec_transform.compute_length(x)
            audio = spec_transform(zmuv_transform(spec_transform(batch.audio, mels_only=True)), deltas_only=True)
            with torch.no_grad():
                scores = model(audio, lengths)
            num_correct += (scores.max(1)[1] == batch.labels).float().sum().item()
            num_total += scores.size(0)
            pbar.set_postfix(dict(acc=f'{num_correct / num_total:.2f}'))
        if epoch_idx == 9:
            optimizer = AdamW(params, 1e-3 / 3, weight_decay=args.weight_decay)
        tqdm.write(f'Accuracy: {num_correct / num_total:.4f}')


if __name__ == '__main__':
    main()
