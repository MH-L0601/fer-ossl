"""Mapper from Exp-CLIP (4096-dim) features to ResNet50 (2048-dim) semantic space.

Used to map CLIP text prototypes into the ResNet50 feature space before injecting
them into DebGCD's clustering head.

The mapper is trained by ``train_mapper.py`` and saved to
``dev_outputs/mapper_expclip_to_resnet50.pth``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mapper(nn.Module):
    """MLP from in_dim to out_dim, output L2-normalized."""

    def __init__(self, in_dim=2048, out_dim=4096, hidden=2048, n_layers=2):
        super().__init__()
        layers = []
        if n_layers <= 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.GELU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden, hidden))
                layers.append(nn.LayerNorm(hidden))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def load_mapper(path, device='cpu'):
    """Load a trained mapper checkpoint."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    mapper = Mapper(in_dim=cfg['in_dim'], out_dim=cfg['out_dim'],
                    hidden=cfg['hidden'], n_layers=cfg['n_layers'])
    mapper.load_state_dict(ckpt['state_dict'])
    mapper.to(device)
    return mapper


class ResNet50ClipMap(nn.Module):
    """resnet50 (2048-dim) + frozen mapper (-> 4096-dim CLIP space).

    ``forward`` returns the L2-normalized 4096-dim feature, matching the output
    of ``ExpCLIPBackbone`` so DebGCD's head (feat_dim=4096) and text prototypes
    work without any other change.
    """

    def __init__(self, resnet, mapper):
        super().__init__()
        self.resnet = resnet
        self.mapper = mapper
        for p in self.mapper.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.mapper(self.resnet(x))
