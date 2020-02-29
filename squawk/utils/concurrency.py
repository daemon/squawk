from dataclasses import dataclass
from typing import Union
import os
import pathlib
import fcntl


@dataclass
class FileLock(object):
    path: Union[str, pathlib.Path]
    grab: bool = True
    locked: bool = False

    def __enter__(self, *args):
        if not self.grab:
            return self
        self.lock()
        return self

    def lock(self):
        self.f = open(self.path, 'w')
        fcntl.flock(self.f, fcntl.LOCK_EX)
        self.locked = True

    def unlock(self):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()
        self.locked = False

    def __exit__(self, *args):
        if not self.grab:
            return
        if self.locked:
            self.unlock()

    @staticmethod
    def clear(path: str):
        try:
            os.remove(path)
        except:
            pass
