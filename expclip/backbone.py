"""Exp-CLIP backbone: frozen CLIP image encoder + Exp-CLIP projection head.

Replaces DebGCD's ViT/ResNet backbone. The CLIP image encoder and the Exp-CLIP
projection head (Linear 512 -> 4096 for ViT-B/16) are both frozen; the DebGCD
head is trained on top of the L2-normalized 4096-dim features.

The projection head weight is loaded from the Exp-CLIP checkpoint (e.g.
``ExpCLIP_B16_model.pth``), whose state_dict key is ``mlp.weight`` of shape
(4096, 512).
"""
import torch
import torch.nn as nn

from .clip import clip


class ExpCLIPBackbone(nn.Module):
    def __init__(self, clip_name='ViT-B/16', head_path=None, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.clip_model, _ = clip.load(clip_name, self.device)

        if head_path is None:
            raise ValueError('head_path (Exp-CLIP projection head checkpoint) is required')

        w = self._load_head_weight(head_path)
        # w: (out_dim, in_dim) = (4096, 512) for ViT-B/16
        self.projection_head = nn.Linear(w.shape[1], w.shape[0], bias=False)
        with torch.no_grad():
            self.projection_head.weight.copy_(w)
        self.projection_head.to(self.device)
        self.feat_dim = w.shape[0]

        # freeze CLIP + projection head entirely
        for p in self.parameters():
            p.requires_grad = False

    @staticmethod
    def _load_head_weight(head_path):
        ckpt = torch.load(head_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            w = None
            for k, v in ckpt.items():
                if 'mlp.weight' in k or k == 'weight':
                    w = v
                    break
            if w is None:
                raise KeyError(
                    f'No projection head weight found in {head_path}; keys={list(ckpt.keys())}'
                )
        else:
            w = ckpt
        return w

    def encode_image(self, x):
        # ensure fp32 (CLIP expects float); autocast-safe
        f = self.clip_model.encode_image(x.float()).float()
        f = self.projection_head(f)
        f = f / f.norm(dim=-1, keepdim=True)
        return f

    @torch.no_grad()
    def encode_text(self, text_list):
        tokens = clip.tokenize(text_list, context_length=77, truncate=True).to(self.device)
        f = self.clip_model.encode_text(tokens).float()
        f = self.projection_head(f)
        f = f / f.norm(dim=-1, keepdim=True)
        return f

    def forward(self, x):
        return self.encode_image(x)
