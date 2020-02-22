from dataclasses import dataclass, field

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LASEncoderConfig(object):
    num_spec_channels: int = 3
    num_latent_channels: int = 64
    hidden_size: int = 512
    num_layers: int = 1


@dataclass
class FixedAttentionModuleConfig(object):
    num_heads: int = 4
    hidden_size: int = 512


@dataclass
class LASClassifierConfig(object):
    num_labels: int
    dnn_size: int = 512
    dropout: float = 0.2
    las_config: LASEncoderConfig = field(default_factory=LASEncoderConfig)
    fixed_attn_config: FixedAttentionModuleConfig = field(default_factory=FixedAttentionModuleConfig)


class LASEncoder(nn.Module):

    def __init__(self, config: LASEncoderConfig = LASEncoderConfig()):
        super().__init__()
        out_channels = config.num_latent_channels
        hidden_size = config.hidden_size
        self.conv1 = conv1 = nn.Conv2d(config.num_spec_channels, stride=2, kernel_size=3, out_channels=out_channels)
        self.conv2 = conv2 = nn.Conv2d(out_channels, stride=2, kernel_size=3, out_channels=out_channels)
        self.conv_encoder = nn.Sequential(conv1,
                                          nn.BatchNorm2d(out_channels),
                                          nn.ReLU(),
                                          nn.MaxPool2d((1, 2)),
                                          conv2,
                                          nn.BatchNorm2d(out_channels),
                                          nn.ReLU(),
                                          nn.MaxPool2d((1, 2)))
        self.lstm_encoder = nn.LSTM(out_channels * 19, hidden_size, config.num_layers, bias=True, bidirectional=True)

    def forward(self, x, lengths):
        x = self.conv_encoder(x)
        x = x.permute(3, 0, 1, 2).contiguous()
        x = x.view(-1, x.size(1), x.size(2) * x.size(3))
        lengths = ((lengths.float() - self.conv1.kernel_size[1]) / 2 + 1).floor()
        lengths = (lengths / 2).floor()
        lengths = ((lengths.float() - self.conv2.kernel_size[1]) / 2 + 1).floor()
        lengths = (lengths / 2).floor()
        rnn_seq, (rnn_out, _) = self.lstm_encoder(pack_padded_sequence(x, lengths))
        return rnn_seq, rnn_out


class FixedAttentionModule(nn.Module):

    def __init__(self, config: FixedAttentionModuleConfig = FixedAttentionModuleConfig()):
        super().__init__()
        self.context_vec = nn.Parameter(torch.Tensor(config.hidden_size * 2).uniform_(-0.25, 0.25))
        self.v_proj = nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
        self.k_proj = nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
        self.num_heads = config.num_heads

    def forward(self, rnn_seq, mask=None):
        values = self.v_proj(rnn_seq)
        keys = self.k_proj(rnn_seq)
        rnn_seq = values.view(values.size(0), values.size(1), self.num_heads, values.size(2) // self.num_heads)
        keys = keys.view(values.size(0), keys.size(1), self.num_heads, keys.size(2) // self.num_heads)
        cvec = self.context_vec.view(-1, self.num_heads).unsqueeze(-1).expand(-1, -1, rnn_seq.size(0))
        logits = torch.einsum('ijkl,lki->ijk', rnn_seq, cvec)
        if mask is not None:
            mask = (1 - mask) * -100
            logits += mask.unsqueeze(-1).expand_as(logits)
        scores = F.softmax(logits, 0)
        vec = torch.einsum('ijk,ijkl->jkl', scores, keys)
        return vec.view(vec.size(0), -1)


class LASClassifier(nn.Module):

    def __init__(self, config: LASClassifierConfig):
        super().__init__()
        self.encoder = LASEncoder(config.las_config)
        self.attn = FixedAttentionModule(config.fixed_attn_config)
        self.fc = nn.Sequential(nn.Linear(config.las_config.hidden_size * 2, config.dnn_size),
                                nn.ReLU(),
                                nn.Dropout(config.dropout),
                                nn.Linear(config.dnn_size, config.num_labels))

    def forward(self, x, lengths):
        rnn_seq, rnn_out = self.encoder(x, lengths)
        rnn_seq, lengths = pad_packed_sequence(rnn_seq)

        mask = torch.zeros(rnn_seq.size(0), rnn_seq.size(1))
        for idx, length in enumerate(lengths.tolist()):
            mask[:length, idx] = 1
        mask = mask.to(rnn_seq.device)
        context = self.attn(rnn_seq, mask=mask)
        return self.fc(context)
