from copy import deepcopy
from itertools import chain
from pathlib import Path
import argparse

from tqdm import trange, tqdm
from torch.optim.adamw import AdamW
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud

from squawk.data import load_freesounds, batchify, StandardAudioTransform, ZmuvTransform, SpecAugmentTransform,\
    find_metric, TimeshiftTransform, compose, SpecAugmentConfig, IdentityTransform
from squawk.model import LASClassifier, LASClassifierConfig, MobileNetClassifier, MNClassifierConfig
from squawk.utils import prettify_dataclass, Workspace, set_seed, prepare_device


def main():
    def evaluate(data_loader, prefix: str):
        model.eval()
        spec_transform.eval()
        pbar = tqdm(data_loader, total=len(data_loader), position=1, desc=prefix, leave=True)
        for tracker in trackers:
            tracker.reset()
        for batch in pbar:
            batch.to(device)
            lengths = batch.lengths.clone()
            for idx, x in enumerate(lengths):
                lengths[idx] = spec_transform.compute_length(x)
            audio = spec_transform(zmuv_transform(spec_transform(batch.audio, mels_only=True)), deltas_only=True)
            with torch.no_grad():
                scores = model(audio, lengths)
            for tracker in trackers:
                tracker.update(scores, batch.labels)
        for tracker in trackers:
            writer.add_scalar(f'{prefix}/Loss/{tracker.name}', tracker.value, epoch_idx)
            tqdm.write(f'{epoch_idx + 1},{tracker.name},{tracker.value:.4f}')

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('--num-epochs', '-ne', type=int, default=20)
    parser.add_argument('--batch-size', '-bsz', type=int, default=16)
    parser.add_argument('--weight-decay', '-wd', type=float, default=5e-5)
    parser.add_argument('--no-spec-augment', '-nosa', action='store_false', dest='use_spec_augment')
    parser.add_argument('--workspace', '-w', type=str, default=str(Path('workspaces') / 'default'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model', type=str, default='las', choices=['las', 'mn'])
    parser.add_argument('--no-timewarp', '-notw', action='store_false', dest='use_timewarp')
    parser.add_argument('--use-timeshift', '-ts', action='store_true')
    parser.add_argument('--use-vtlp', '-vtlp', action='store_true')
    parser.add_argument('--num-gpu', type=int, default=1)
    parser.add_argument('--lru-maxsize', type=int, default=np.inf)
    parser.add_argument('--use-pba', action='store_true')
    args = parser.parse_args()

    device, gpu_device_ids = prepare_device(args.num_gpu)
    set_seed(args.seed)

    train_ds, dev_ds, test_ds = load_freesounds(Path(args.dir), lru_maxsize=args.lru_maxsize)
    timeshift_transform = TimeshiftTransform(sr=train_ds.info.sample_rate)
    if args.use_timeshift:
        train_collate = compose(deepcopy, timeshift_transform.shift, batchify)
    else:
        train_collate = batchify
    train_loader = tud.DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, shuffle=True, collate_fn=train_collate, num_workers=16, drop_last=True)
    dev_loader = tud.DataLoader(dev_ds, batch_size=16, pin_memory=True, shuffle=False, collate_fn=batchify, num_workers=16)
    test_loader = tud.DataLoader(test_ds, batch_size=16, pin_memory=True, shuffle=False, collate_fn=batchify, num_workers=16)
    spec_transform = StandardAudioTransform(use_vtlp=args.use_vtlp)
    spec_transform.cuda()

    if args.model == 'las':
        config = LASClassifierConfig(train_ds.info.num_labels)
        model = LASClassifier(config)
    else:
        config = MNClassifierConfig(train_ds.info.num_labels)
        model = MobileNetClassifier(config)
    tqdm.write(prettify_dataclass(args))
    tqdm.write(prettify_dataclass(config))

    ws = Workspace(Path(args.workspace))
    ws.write_config(config)
    ws.write_args(args)
    writer = ws.summary_writer

    sa_config = SpecAugmentConfig(use_timewarp=args.use_timewarp)
    zmuv_transform = ZmuvTransform()
    sa_transform = SpecAugmentTransform(sa_config) if args.use_spec_augment else IdentityTransform()
    criterion = nn.CrossEntropyLoss()
    params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = AdamW(params, args.lr, weight_decay=args.weight_decay)
    trackers = (find_metric('accuracy')(), find_metric('map')(precision=3))
    tqdm.write(f'# Parameters: {sum(p.numel() for p in params)}')
    writer.add_scalar('Meta/Parameters', sum(p.numel() for p in params))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Multiple GPU support; to be enabled with further testing
    # if len(gpu_device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=gpu_device_ids)

    spec_transform = spec_transform.to(device)
    model = model.to(device)
    zmuv_transform = zmuv_transform.to(device)
    sa_transform = sa_transform.to(device)

    for batch in tqdm(chain(train_loader, dev_loader), desc='Constructing ZMUV', total=len(train_loader) + len(dev_loader)):
        audio = batch.audio.to(device)
        lengths = batch.lengths.clone()
        for idx, x in enumerate(lengths):
            lengths[idx] = spec_transform.compute_length(x)

        audio = spec_transform(audio, mels_only=True)
        mask = torch.zeros_like(audio)
        for batch_idx, length in enumerate(lengths.tolist()):
            mask[batch_idx, :, :length] = 1
        zmuv_transform.update(audio, mask=mask)

    for epoch_idx in trange(args.num_epochs, position=0, leave=True):
        model.train()
        spec_transform.train()
        pbar = tqdm(train_loader, total=len(train_loader), position=1, desc='Training', leave=True)
        for batch in pbar:
            batch.to(device)
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

        evaluate(dev_loader, 'Dev')
        if epoch_idx == 9:
            optimizer = AdamW(params, args.lr / 3, weight_decay=args.weight_decay)
    evaluate(test_loader, 'Test')


if __name__ == '__main__':
    main()
