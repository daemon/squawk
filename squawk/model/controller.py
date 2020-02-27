from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FFNControllerConfig(object):
    num_outputs: int
    embedding_size: int = 8
    dnn_size: int = 16


class FFNController(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Parameter(torch.Tensor(config.embedding_size).uniform_(-0.25, 0.25), requires_grad=True)
        self.dnn = nn.Sequential(nn.Linear(config.embedding_size, config.dnn_size),
                                 nn.GELU(),
                                 nn.Linear(config.dnn_size, config.num_outputs))

    def forward(self):
        return self.dnn(self.embedding)
