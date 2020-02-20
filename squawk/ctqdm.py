from dataclasses import dataclass
from tqdm import tqdm, trange


@dataclass
class TqdmContext(object):
    use_tqdm: bool = False

    def __enter__(self, *args):
        _GLOBAL_CONTEXT.append(self.use_tqdm)

    def __exit__(self, *args):
        _GLOBAL_CONTEXT.pop()


_GLOBAL_CONTEXT = [True]


def enable_tqdm():
    return TqdmContext(True)


def disable_tqdm():
    return TqdmContext(False)


def ctqdm(*args, **kwargs):
    kwargs['disable'] = not _GLOBAL_CONTEXT[-1]
    return tqdm(*args, **kwargs)


def ctrange(*args, **kwargs):
    kwargs['disable'] = not _GLOBAL_CONTEXT[-1]
    return trange(*args, **kwargs)