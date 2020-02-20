from pathlib import Path
import argparse

from torchaudio.transforms import MelSpectrogram

from squawk.data import load_freesounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=Path)
    args = parser.parse_args()
    load_freesounds(args.dir)


if __name__ == '__main__':
    main()
