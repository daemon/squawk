from itertools import chain
from pathlib import Path
import argparse

from tqdm import trange, tqdm
from torch.optim.adamw import AdamW
import torch
import torch.nn as nn
import torch.utils.data as tud

from squawk.data import load_freesounds, batchify, StandardAudioTransform, ZmuvTransform, SpecAugmentTransform,\
    find_metric, IdentityTransform
from squawk.model import LASClassifier, LASClassifierConfig, MobileNetClassifier, MNClassifierConfig
from squawk.utils import prettify_dataclass, Workspace, set_seed, prepare_device


def main():
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
    parser.add_argument('--num_gpu', type=int, default=1)
    args = parser.parse_args()

    # prepare hardware accelearation
    device, gpu_device_ids = prepare_device(args.num_gpu)
    set_seed(args.seed, device)

    train_ds, dev_ds, test_ds = load_freesounds(Path(args.dir))
    train_loader = tud.DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, shuffle=True, collate_fn=batchify, num_workers=16, drop_last=True)
    dev_loader = tud.DataLoader(dev_ds, batch_size=10, pin_memory=True, shuffle=False, collate_fn=batchify, num_workers=10)
    spec_transform = StandardAudioTransform()

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

    zmuv_transform = ZmuvTransform()
    sa_transform = SpecAugmentTransform() if args.use_spec_augment else IdentityTransform
    criterion = nn.CrossEntropyLoss()
    params = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = AdamW(params, args.lr, weight_decay=args.weight_decay)
    trackers = (find_metric('accuracy')(), find_metric('map')(precision=3))
    tqdm.write(f'# Parameters: {sum(p.numel() for p in params)}')
    writer.add_scalar('Meta/Parameters', sum(p.numel() for p in params))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # # Multiple GPU supports; To be enabled with further testing
    # if len(gpu_device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=gpu_device_ids)

    spec_transform = spec_transform.to(device)
    model = model.to(device)
    zmuv_transform = zmuv_transform.to(device)
    sa_transform = sa_transform.to(device)

    for batch in tqdm(train_loader, desc='Constructing ZMUV with train dataset'):
        audio = batch.audio.to(device)
        zmuv_transform.update(spec_transform(audio, mels_only=True))

    for batch in tqdm(dev_loader, desc='Constructing ZMUV with dev dataset'):
        audio = batch.audio.to(device)
        zmuv_transform.update(spec_transform(audio, mels_only=True))

    for epoch_idx in trange(args.num_epochs, position=0, leave=True):
        model.train()
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

        model.eval()
        pbar = tqdm(dev_loader, total=len(dev_loader), position=1, desc='Testing', leave=True)
        for tracker in trackers:
            tracker.reset()
        for batch in pbar:
            batch.cuda()
            lengths = batch.lengths.clone()
            for idx, x in enumerate(lengths):
                lengths[idx] = spec_transform.compute_length(x)
            audio = spec_transform(zmuv_transform(spec_transform(batch.audio, mels_only=True)), deltas_only=True)
            with torch.no_grad():
                scores = model(audio, lengths)
            for tracker in trackers:
                tracker.update(scores, batch.labels)
        if epoch_idx == 9:
            optimizer = AdamW(params, args.lr / 3, weight_decay=args.weight_decay)
        for tracker in trackers:
            writer.add_scalar(f'Dev/Loss/{tracker.name}', tracker.value, epoch_idx)
            tqdm.write(f'{epoch_idx + 1},{tracker.name},{tracker.value:.4f}')


if __name__ == '__main__':
    main()
