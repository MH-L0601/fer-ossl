"""Train a mapper from Exp-CLIP (4096-dim) to ResNet50 (2048-dim) space.

The mapper is trained on paired Exp-CLIP and ResNet50 features from the same
RAF-DB images, using a symmetric InfoNCE and cosine-alignment loss. It maps CLIP
text prototypes into the ResNet50 feature space for DebGCD clustering.

Usage:
    python train_mapper.py \
        --resnet_ckpt rafdb_known6_resnet50_pretrain.pth \
        --expclip_head expclip/checkpoint/ExpCLIP_B16_model.pth \
        --out dev_outputs/mapper_expclip_to_resnet50.pth
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from config import rafdb_root
from data.rafdb import RAFDB
from expclip.backbone import ExpCLIPBackbone
from expclip.text_prototypes import build_text_prototypes, RAFDB_ID_TO_EMOTION
from models.mapper import Mapper
from models.resnet_backbone import build_resnet50


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def make_transform(mean, std):
    return transforms.Compose([
        transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def info_nce(a, b, temp=0.07):
    """Symmetric InfoNCE between paired batches of L2-normalized features."""
    sim = a @ b.T / temp
    labels = torch.arange(a.shape[0], device=a.device)
    return F.cross_entropy(sim, labels)


@torch.no_grad()
def extract_features(backbone, loader, device, desc=''):
    """Extract backbone features for a dataset, returning an [N, D] array."""
    feats = []
    for i, (img, _, _) in enumerate(loader):
        feats.append(backbone(img.to(device)).detach().cpu())
        if (i + 1) % 20 == 0:
            print(f'    [{desc}] {i + 1}/{len(loader)} batches')
    return torch.cat(feats, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser(
        description='Train Exp-CLIP (4096) -> ResNet50 (2048) mapper')
    parser.add_argument('--resnet_ckpt', type=str,
                        default='rafdb_known6_resnet50_pretrain.pth',
                        help='ResNet50 backbone weights (fc=Identity compatible).')
    parser.add_argument('--expclip_head', type=str,
                        default='expclip/checkpoint/ExpCLIP_B16_model.pth')
    parser.add_argument('--clip_name', type=str, default='ViT-B/16')
    parser.add_argument('--out', type=str,
                        default='dev_outputs/mapper_expclip_to_resnet50.pth')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--extract_batch', type=int, default=256)
    parser.add_argument('--hidden', type=int, default=4096)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--temp', type=float, default=0.07)
    parser.add_argument('--lambda_align', type=float, default=1.0,
                        help='Weight of the cosine alignment term relative to InfoNCE.')
    parser.add_argument('--known_classes', type=int, nargs='+',
                        default=[0, 1, 2, 4, 5, 6])
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Max RAF-DB samples to use (-1 = all).')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    print('Building ResNet50 backbone...')
    resnet = build_resnet50(pretrained=False)
    ckpt = torch.load(args.resnet_ckpt, map_location='cpu', weights_only=False)
    missing, unexpected = resnet.load_state_dict(ckpt, strict=False)
    print(f'  ResNet50 load: {len(missing)} missing, {len(unexpected)} unexpected keys')
    for p in resnet.parameters():
        p.requires_grad = False
    resnet.eval().to(device)

    print('Building Exp-CLIP backbone...')
    expclip = ExpCLIPBackbone(
        clip_name=args.clip_name, head_path=args.expclip_head, device=device)
    expclip.eval().to(device)
    print(f'  Exp-CLIP feat_dim = {expclip.feat_dim}')

    resnet_tf = make_transform(IMAGENET_MEAN, IMAGENET_STD)
    clip_tf = make_transform(CLIP_MEAN, CLIP_STD)
    ds_resnet = RAFDB(root=rafdb_root, train=True, transform=resnet_tf)
    ds_clip = RAFDB(root=rafdb_root, train=True, transform=clip_tf)

    if args.max_samples > 0:
        n = min(args.max_samples, len(ds_resnet))
        ds_resnet.samples = ds_resnet.samples[:n]
        ds_resnet.uq_idxs = ds_resnet.uq_idxs[:n]
        ds_clip.samples = ds_clip.samples[:n]
        ds_clip.uq_idxs = ds_clip.uq_idxs[:n]
        ds_resnet.target_transform = None
        ds_clip.target_transform = None

    loader_resnet = DataLoader(
        ds_resnet, batch_size=args.extract_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    loader_clip = DataLoader(
        ds_clip, batch_size=args.extract_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print(f'Extracting paired features from {len(ds_resnet)} RAF-DB train images...')
    X = extract_features(resnet, loader_resnet, device, desc='resnet50')
    Y = extract_features(expclip, loader_clip, device, desc='expclip')
    print(f'  X (resnet50) shape = {X.shape}, Y (expclip) shape = {Y.shape}')

    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    X_norm = F.normalize(X, dim=-1)
    N = X.shape[0]

    mapper = Mapper(in_dim=Y.shape[1], out_dim=X.shape[1],
                    hidden=args.hidden, n_layers=args.n_layers).to(device)
    print(f'\nMapper: {sum(p.numel() for p in mapper.parameters()) / 1e6:.2f}M params')

    optimizer = torch.optim.AdamW(
        mapper.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    print(f'\nTraining mapper for {args.epochs} epochs...')
    best_loss = float('inf')
    best_state = None
    for epoch in range(args.epochs):
        mapper.train()
        epoch_loss = 0.0
        nb = 0
        perm = torch.randperm(N)
        for i in range(0, N, args.batch_size):
            bidx = perm[i:i + args.batch_size]
            y_b = Y[bidx].to(device)
            x_b = X_norm[bidx].to(device)
            x_hat = mapper(y_b)

            loss_nce = 0.5 * (
                info_nce(x_hat, x_b, args.temp)
                + info_nce(x_b, x_hat, args.temp))
            loss_align = 1.0 - (x_hat * x_b).sum(dim=1).mean()
            loss = loss_nce + args.lambda_align * loss_align

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1

        scheduler.step()
        avg = epoch_loss / nb
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.cpu().clone()
                          for k, v in mapper.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'  epoch {epoch + 1:3d}/{args.epochs}  loss={avg:.4f}  '
                  f'(nce={loss_nce.item():.4f}, align={loss_align.item():.4f})')

    mapper.load_state_dict(best_state)
    mapper.eval()

    print('\n=== Semantic alignment in ResNet50 space ===')
    known_classes = list(args.known_classes)
    text_protos = build_text_prototypes(expclip, known_classes).cpu()
    with torch.no_grad():
        text_protos = F.normalize(
            mapper(text_protos.to(device)), dim=-1).cpu()

    targets = torch.tensor([t for (_, t) in ds_resnet.samples])
    print(f'{"class":<10}{"emotion":<12}{"cos(center, own proto)":<24}  top-1 class')
    retrieval_correct = 0
    retrieval_total = 0
    for k, c in enumerate(known_classes):
        mask = targets == c
        if mask.sum() == 0:
            continue
        center = F.normalize(X_norm[mask].mean(dim=0, keepdim=True), dim=-1)
        proto = text_protos[k]
        own_cos = (center * proto).sum().item()
        pred = int((center * text_protos).sum(dim=1).argmax().item())
        correct = pred == k
        retrieval_correct += int(correct)
        retrieval_total += 1
        print(f'{c:<10}{RAFDB_ID_TO_EMOTION[c]:<12}{own_cos:<24.4f}  '
              f'{known_classes[pred]} {"OK" if correct else "MISMATCH"}')

    print(f'\n  Prototype retrieval accuracy (known classes): '
          f'{retrieval_correct}/{retrieval_total} = '
          f'{retrieval_correct / retrieval_total:.4f}')

    payload = {
        'state_dict': best_state,
        'config': {
            'in_dim': Y.shape[1],
            'out_dim': X.shape[1],
            'hidden': args.hidden,
            'n_layers': args.n_layers,
            'resnet_ckpt': args.resnet_ckpt,
            'expclip_head': args.expclip_head,
            'clip_name': args.clip_name,
        },
        'best_loss': best_loss,
    }
    torch.save(payload, args.out)
    print(f'\nSaved mapper to {args.out}  (best_loss={best_loss:.4f})')


if __name__ == '__main__':
    main()
