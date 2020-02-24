from dataclasses import dataclass
from pathlib import Path
import json
import shutil

from torch.utils.tensorboard import SummaryWriter

from .dataclass import gather_dict


@dataclass
class Workspace(object):
    path: Path

    def __post_init__(self):
        self.path.mkdir(parents=True, exist_ok=True)
        log_path = self.path / 'logs'
        self.summary_writer = SummaryWriter(str(log_path))
        try:
            shutil.rmtree(log_path)
        except:
            pass

    def write_config(self, config):
        with open(self.path / 'config.json', 'w') as f:
            json.dump(gather_dict(config), f, indent=2)

    def write_args(self, args):
        with open(self.path / 'cmd-args.json', 'w') as f:
            json.dump(gather_dict(args), f, indent=2)
