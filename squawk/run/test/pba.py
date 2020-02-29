import argparse
import time

from squawk.optim import PbaMetaOptimizer, AugmentationParameter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', '-i', type=int, required=True)
    args = parser.parse_args()
    meta_opt = PbaMetaOptimizer(f'fake-{args.id}', dict(test=AugmentationParameter(list(range(20)), 'test', 0)))
    for epoch_idx in range(10):
        params = meta_opt.sample()
        if params:
            quality = params[0].domain[params[0].current_value_idx]
        else:
            quality = 0
        print(dict(epoch_idx=epoch_idx, id=args.id, quality=quality))
        time.sleep(1)
        meta_opt.step(quality)
    meta_opt.cleanup()


if __name__ == '__main__':
    main()
