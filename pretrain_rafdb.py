"""
RAF-DB 已知类监督预训练 (为 DebGCD 提供 warmup backbone 权重)。

在 RAF-DB 的【已知类】(默认排除未知类 3 = Happiness) 上微调骨干,
保存 backbone.state_dict(), 之后 DebGCD 通过 --warmup_model_dir 直接加载。

支持两种骨干 (--backbone):
  - vit      : DINO ViT-B (768 维), 从 --warmup_model_dir 的 DINO checkpoint 初始化
  - resnet50 : ResNet50 (2048 维), 默认从 ImageNet 初始化, 也可用 --warmup_model_dir
               加载现有 ResNet50 权重 (如 fer-ossl 的 encoder_pretrained_50.pth)

用法示例 (未知类 3 = Happiness, 已知类 = 0,1,2,4,5,6):
  # ResNet50 骨干 (与 fer-ossl 对齐)
  python pretrain_rafdb.py \
      --backbone resnet50 \
      --unknown_classes 3 \
      --epochs 50 \
      --lr 0.01 \
      --grad_from_block 0 \
      --batch_size 32 \
      --save_path ./rafdb_known6_resnet50_pretrain.pth
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import random
import time
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.rafdb import RAFDB, subsample_classes
from config import rafdb_root
from models import vision_transformer as vits
from models.resnet_backbone import build_resnet50, set_trainable_from_grad, FEATURE_DIMS


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_transforms(image_size=224):
    interpolation = transforms.InterpolationMode.BICUBIC
    crop_pct = 0.875

    train_transform = transforms.Compose([
        transforms.Resize(int(image_size / crop_pct), interpolation),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(int(image_size / crop_pct), interpolation),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    return train_transform, test_transform


def get_known_classes(args):
    """已知类 = 除 unknown_classes 外的所有 7 类表情。"""
    unknown = sorted(set(args.unknown_classes))
    assert all(0 <= c < 7 for c in unknown), f'unknown_classes must be within 0..6, got {unknown}'
    assert len(unknown) < 7, 'unknown_classes cannot include all 7 classes'
    known = [c for c in range(7) if c not in set(unknown)]
    return known, unknown


def build_datasets(known_classes, train_transform, test_transform):
    whole_train = RAFDB(root=rafdb_root, transform=train_transform, train=True)
    whole_test = RAFDB(root=rafdb_root, transform=test_transform, train=False)

    # subsample_classes 会过滤到已知类并把标签重映射到 0..num_known-1
    train_set = subsample_classes(deepcopy(whole_train), include_classes=known_classes)
    test_set = subsample_classes(deepcopy(whole_test), include_classes=known_classes)
    return train_set, test_set


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description='RAF-DB known-class pretraining')
    parser.add_argument('--backbone', type=str, default='vit', choices=['vit', 'resnet50'],
                        help='Backbone to pretrain: vit (DINO ViT-B) or resnet50.')
    parser.add_argument('--warmup_model_dir', type=str, default=None,
                        help='Optional pretrained weights to load as the starting point. '
                             'For vit, pass the DINO ViT-B checkpoint; for resnet50, pass '
                             'an existing ResNet50 backbone (e.g. fer-ossl encoder_pretrained_50.pth) '
                             'or omit to start from ImageNet.')
    parser.add_argument('--unknown_classes', nargs='+', type=int, default=[3],
                        help='Unknown class ids to exclude from pretraining, e.g. "3"')
    parser.add_argument('--save_path', type=str, default='./rafdb_known6_pretrain.pth')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Full finetune (grad_from_block=0) needs ~32 on 11GB GPU; '
                             'grad_from_block>=11 can use 128.')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--grad_from_block', type=int, default=0,
                        help='Finetune blocks >= this index (0 = full backbone).')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_random_seed(args.seed)

    known_classes, unknown = get_known_classes(args)
    num_known = len(known_classes)
    print(f'Known classes: {known_classes} ({num_known}), Unknown classes: {unknown}')

    # ----------------------
    # 数据
    # ----------------------
    train_transform, test_transform = build_transforms(args.image_size)
    train_set, test_set = build_datasets(known_classes, train_transform, test_transform)
    print(f'Train samples (known): {len(train_set)}, Test samples (known): {len(test_set)}')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=args.num_workers, pin_memory=False)

    # ----------------------
    # 模型: 骨干 + 分类头
    # ----------------------
    if args.backbone == 'vit':
        backbone = vits.__dict__['vit_base']()
    elif args.backbone == 'resnet50':
        backbone = build_resnet50(pretrained=True)
    else:
        raise ValueError(f'Unknown backbone {args.backbone}')

    if args.warmup_model_dir is not None and os.path.exists(args.warmup_model_dir):
        print(f'Loading pretrained weights from {args.warmup_model_dir}')
        state = torch.load(args.warmup_model_dir, map_location='cpu')
        backbone.load_state_dict(state)

    feat_dim = FEATURE_DIMS[args.backbone]

    # 冻结全部, 再从 grad_from_block 开始解冻
    set_trainable_from_grad(backbone, args.backbone, args.grad_from_block)

    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.parameters())
    print(f'Backbone trainable params: {trainable}/{total}')

    head = nn.Linear(feat_dim, num_known).to(device)
    model = nn.Sequential(backbone, head).to(device)

    optimizer = SGD([
        {'params': [p for p in backbone.parameters() if p.requires_grad]},
        {'params': head.parameters()},
    ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        start = time.perf_counter()
        for images, labels, _ in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        scheduler.step()
        test_acc = evaluate(model, test_loader, device) * 100.0
        avg_loss = running_loss / len(train_set)
        print(f'Epoch {epoch}/{args.epochs - 1}  loss {avg_loss:.4f}  test_acc {test_acc:.2f}%  '
              f'time {time.perf_counter() - start:.1f}s')

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(backbone.state_dict(), args.save_path)
            print(f'  -> saved best backbone to {args.save_path} (test_acc {best_acc:.2f}%)')

    print(f'Pretraining done. Best known-class test acc: {best_acc:.2f}%')
    print(f'Backbone saved to {args.save_path}')


if __name__ == '__main__':
    main()
