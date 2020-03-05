from copy import deepcopy
from itertools import chain
from pathlib import Path
import argparse
import json

from tqdm import trange, tqdm
from torch.optim.adamw import AdamW
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud

from squawk.data import load_gsc, batchify, StandardAudioTransform, ZmuvTransform, SpecAugmentTransform,\
    find_metric, TimeshiftTransform, compose, TimestretchTransform, NoiseTransform
from squawk.model import ResNet, ResNetConfig
from squawk.optim import PbaMetaOptimizer
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
        results = {}
        for tracker in trackers:
            writer.add_scalar(f'{prefix}/Loss/{tracker.name}', tracker.value, epoch_idx)
            results[tracker.name] = tracker.value
            tqdm.write(f'{epoch_idx + 1},{tracker.name},{tracker.value:.4f}')
        if args.use_pba:
            with pba_optimizer.lock():
                ws.increment_model(model, results[args.target_metric])
        else:
            ws.increment_model(model, results[args.target_metric])
        return results

    def load_model(state_dict):
        model.cpu()
        model.load_state_dict(state_dict)
        model.to(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('--num-epochs', '-ne', type=int, default=20)
    parser.add_argument('--batch-size', '-bsz', type=int, default=16)
    parser.add_argument('--weight-decay', '-wd', type=float, default=5e-5)
    parser.add_argument('--no-spec-augment', '-nosa', action='store_false', dest='use_spec_augment')
    parser.add_argument('--workspace', '-w', type=str, default=str(Path('workspaces') / 'default'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--model', type=str, default='las', choices=['las', 'mn'])
    parser.add_argument('--no-timewarp', '-notw', action='store_false', dest='use_timewarp')
    parser.add_argument('--use-timeshift', '-ts', action='store_true')
    parser.add_argument('--use-vtlp', '-vtlp', action='store_true')
    parser.add_argument('--num-gpu', type=int, default=1)
    parser.add_argument('--lru-maxsize', type=int, default=np.inf)
    parser.add_argument('--use-pba', action='store_true')
    parser.add_argument('--use-timestretch', action='store_true')
    parser.add_argument('--use-noise', action='store_true')
    parser.add_argument('--target-metric', '-tm', type=str, default='MAP@3')
    parser.add_argument('--use-all', action='store_true')
    parser.add_argument('--pba-init', type=str, choices=['random', 'default'], default='default')
    parser.add_argument('--seed-only', action='store_true')
    parser.add_argument('--no-cleanup', action='store_false', dest='cleanup_pba')
    args = parser.parse_args()

    if args.use_all:
        args.use_timeshift = True
        args.use_timestretch = True
        args.use_spec_augment = True
        args.use_noise = True
        args.use_vtlp = True
    if args.seed_only:
        args.use_spec_augment = False
        args.use_timestretch = False
        args.use_spec_augment = False
        args.use_noise = False
        args.use_vtlp = False

    device, gpu_device_ids = prepare_device(args.num_gpu)
    set_seed(args.seed)

    data_config = {
        "data_dir": args.dir,
        "sample_rate": 16000,
        "group_speakers_by_id": True,
        "dev_pct": 0.1,
        "test_pct": 0.1,
        "noise_pct": 0.1,
        "target_class": ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"],
        "unknown_class": False,
        "silence_class": False
    }

    train_ds, dev_ds, test_ds = load_gsc(data_config, lru_maxsize=args.lru_maxsize)
    timeshift_transform = TimeshiftTransform(sr=train_ds.info.sample_rate)
    timestretch_transform = TimestretchTransform()
    noise_transform = NoiseTransform()
    train_compose = [deepcopy]
    if args.use_timeshift:
        train_compose.append(timeshift_transform)
    if args.use_timestretch:
        train_compose.append(timestretch_transform)
    train_compose.append(batchify)
    if len(train_compose) > 2:
        train_collate = compose(*train_compose)
    else:
        train_collate = batchify
    train_loader = tud.DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, shuffle=True, collate_fn=train_collate, num_workers=16, drop_last=True)
    dev_loader = tud.DataLoader(dev_ds, batch_size=16, pin_memory=True, shuffle=False, collate_fn=batchify, num_workers=16)
    test_loader = tud.DataLoader(test_ds, batch_size=16, pin_memory=True, shuffle=False, collate_fn=batchify, num_workers=16)
    spec_transform = StandardAudioTransform()
    spec_transform.to(device)

    # use args.model if necessary
    model_config = ResNetConfig(len(data_config["target_class"]))
    model = ResNet(model_config)

    tqdm.write(prettify_dataclass(args))
    tqdm.write(prettify_dataclass(model_config))

    ws = Workspace(Path(args.workspace))
    ws.write_config(model_config)
    ws.write_args(args)
    writer = ws.summary_writer

    zmuv_transform = ZmuvTransform()
    sa_transform = SpecAugmentTransform(use_timewarp=args.use_timewarp)
    criterion = nn.CrossEntropyLoss()
    params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = AdamW(params, args.lr, weight_decay=args.weight_decay)
    trackers = (find_metric('accuracy')(), find_metric('map')(precision=3))
    tqdm.write(f'# Parameters: {sum(p.numel() for p in params)}')
    writer.add_scalar('Meta/Parameters', sum(p.numel() for p in params))

    augment_modules = (timestretch_transform, timeshift_transform, noise_transform, sa_transform, spec_transform)
    if args.pba_init == 'random':
        for mod in augment_modules:
            for param in mod.augment_params:
                param.current_value_idx = None
    if args.use_pba:
        augment_params = list(chain(*[mod.augment_params for mod in augment_modules]))
        pba_optimizer = PbaMetaOptimizer(ws.model_path(), augment_params)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Multiple GPU support; to be enabled with further testing
    # if len(gpu_device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=gpu_device_ids)

    spec_transform = spec_transform.to(device)
    if not args.use_vtlp:
        spec_transform.eval()
    model = model.to(device)
    zmuv_transform = zmuv_transform.to(device)
    sa_transform = sa_transform.to(device)
    noise_transform = noise_transform.to(device)

    timeshift_transform.eval()
    timestretch_transform.eval()
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
    timeshift_transform.train()
    timestretch_transform.train()

    for epoch_idx in trange(args.num_epochs, position=0, leave=True):
        model.train()
        if args.use_vtlp: spec_transform.train()
        pbar = tqdm(train_loader, total=len(train_loader), position=1, desc='Training', leave=True)
        for batch in pbar:
            if args.use_pba:
                pba_optimizer.sample()
            batch.to(device)
            audio = batch.audio
            if args.use_noise:
                audio = noise_transform(audio, lengths=batch.lengths)
            audio = spec_transform(audio, mels_only=True)
            audio = zmuv_transform(audio)
            if args.use_spec_augment:
                audio = sa_transform(audio)
            audio = spec_transform(audio, deltas_only=True)
            lengths = batch.lengths.clone()
            for idx, x in enumerate(lengths):
                lengths[idx] = spec_transform.compute_length(x)
            scores = model(audio, lengths)
            optimizer.zero_grad()
            loss = criterion(scores, batch.labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(dict(loss=f'{loss.item():.3}'))

        results = evaluate(dev_loader, 'Dev')
        if args.use_pba:
            pba_optimizer.step(results[args.target_metric], load_model)
        if epoch_idx == 9:
            optimizer = AdamW(params, args.lr / 3, weight_decay=args.weight_decay)
    if args.use_pba and args.cleanup_pba:
        pba_optimizer.cleanup()
    evaluate(test_loader, 'Test')


if __name__ == '__main__':
    main()
