#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
解析 DebGCD 训练日志 (train_DebGCD.py 的三分支输出), 三个分支分别绘图。

日志每个 epoch (由 "Train Epoch: <epoch> Avg Loss" 确定) 会输出:

  [Branch1 GCD     ] Train ACC: All X | Old Y | New Z
  [Branch1 GCD     ] Test  ACC: All X | Old Y | New Z
  [Branch1 GCD     ] AUROC (main-head unknown prob): Train X | Test Y

  [Branch2 OOD     ] AUROC (OVA unknown score): Train X | Test Y
  [Branch2 OOD     ] Det Acc@0.5 (known/unknown): Train X | Test Y
  [Branch2 OOD     ] Det Recall@0.5 (unknown): Train X | Test Y
  [Branch2 OOD     ] Det Spec@0.5 (known): Train X | Test Y
  [Branch2 OOD     ] Det Best Acc (Youden thr Train X | Test Y): Train X | Test Y

  [Branch3 Debiased] Train ACC: All X | Old Y | New Z
  [Branch3 Debiased] Test  ACC: All X | Old Y | New Z

输出三张图:
  branch1_gcd.png        GCD 主聚类分支 (聚类准确率 + AUROC)
  branch2_ood.png        OOD/OVA 分支 (AUROC + 已知/未知筛选指标)
  branch3_debiased.png   去偏辅助分支 (聚类准确率)

用法: python plot_debgcd_log.py <log_path> [out_dir]
"""
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- 解析 ----------
PAT_EPOCH = re.compile(r'Train Epoch:\s*(\d+)\s+Avg Loss')

# 分支 -> 指标名 -> 正则 (捕获 2 或 3 个 float, 按 (train, test) 或 (all, old, new) 排列)
BRANCH_PATTERNS = {
    1: {
        'train_acc': re.compile(r'\[Branch1 GCD\s*\]\s*Train ACC:\s*All\s+([\d.]+)\s*\|\s*Old\s+([\d.]+)\s*\|\s*New\s+([\d.]+)'),
        'test_acc':  re.compile(r'\[Branch1 GCD\s*\]\s*Test\s+ACC:\s*All\s+([\d.]+)\s*\|\s*Old\s+([\d.]+)\s*\|\s*New\s+([\d.]+)'),
        'auroc':     re.compile(r'\[Branch1 GCD\s*\]\s*AUROC.*Train\s+([\d.]+)\s*\|\s*Test\s+([\d.]+)'),
    },
    2: {
        'auroc':         re.compile(r'\[Branch2 OOD\s*\]\s*AUROC.*Train\s+([\d.]+)\s*\|\s*Test\s+([\d.]+)'),
        'det_acc':       re.compile(r'\[Branch2 OOD\s*\]\s*Det Acc@0.5.*Train\s+([\d.]+)\s*\|\s*Test\s+([\d.]+)'),
        'det_recall':    re.compile(r'\[Branch2 OOD\s*\]\s*Det Recall@0.5.*Train\s+([\d.]+)\s*\|\s*Test\s+([\d.]+)'),
        'det_spec':      re.compile(r'\[Branch2 OOD\s*\]\s*Det Spec@0.5.*Train\s+([\d.]+)\s*\|\s*Test\s+([\d.]+)'),
        'det_best_thr':  re.compile(r'\[Branch2 OOD\s*\]\s*Det Best Acc.*Youden thr\s+Train\s+([\d.]+)\s*\|\s*Test\s+([\d.]+)'),
        'det_best_acc':  re.compile(r'\[Branch2 OOD\s*\]\s*Det Best Acc.*Train\s+([\d.]+)\s*\|\s*Test\s+([\d.]+)'),
    },
    3: {
        'train_acc': re.compile(r'\[Branch3 Debiased\]\s*Train ACC:\s*All\s+([\d.]+)\s*\|\s*Old\s+([\d.]+)\s*\|\s*New\s+([\d.]+)'),
        'test_acc':  re.compile(r'\[Branch3 Debiased\]\s*Test\s+ACC:\s*All\s+([\d.]+)\s*\|\s*Old\s+([\d.]+)\s*\|\s*New\s+([\d.]+)'),
    },
}

COLOR_TRAIN = '#2c7fb8'
COLOR_TEST = '#d95f0e'
ACC_COLORS = ['#2c7fb8', '#d95f0e', '#31a354']
ACC_LABELS = ['All', 'Old (known)', 'New (unknown)']


def parse_log(path):
    """返回 {branch: {epoch: {metric: (floats...)}}}"""
    data = {b: {} for b in BRANCH_PATTERNS}
    cur_epoch = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = PAT_EPOCH.search(line)
            if m:
                cur_epoch = int(m.group(1))
                continue
            if cur_epoch is None:
                continue
            for branch, pats in BRANCH_PATTERNS.items():
                for key, pat in pats.items():
                    m = pat.search(line)
                    if m:
                        vals = tuple(float(x) for x in m.groups())
                        data[branch].setdefault(cur_epoch, {})[key] = vals
                        break
    return data


def series(d, key, idx):
    """从 {epoch: {metric: tuple}} 中取出某指标的某分量, 返回 (epochs, values)。"""
    eps = sorted(e for e, v in d.items() if key in v)
    return eps, [d[e][key][idx] for e in eps]


def _style_ax(ax, xlabel='Epoch'):
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def _save(fig, out_dir, name):
    out = os.path.join(out_dir, name)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved -> {out}')
    return out


def plot_branch1(d, out_dir):
    """GCD 主聚类分支: 聚类准确率 (Train/Test) + AUROC。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for i, split in enumerate(['train_acc', 'test_acc']):
        ax = axes[i]
        for j in range(3):
            eps, vals = series(d, split, j)
            ax.plot(eps, vals, label=ACC_LABELS[j], color=ACC_COLORS[j], lw=2)
        ax.set_ylabel('Clustering Accuracy')
        ax.set_title('GCD Acc - ' + ('Train' if split == 'train_acc' else 'Test'))
        _style_ax(ax)

    ax = axes[2]
    eps, tv = series(d, 'auroc', 0)
    ax.plot(eps, tv, label='Train', color=COLOR_TRAIN, lw=2)
    eps, sv = series(d, 'auroc', 1)
    ax.plot(eps, sv, label='Test', color=COLOR_TEST, lw=2)
    ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.7)
    ax.set_ylabel('AUROC')
    ax.set_ylim(0, 1)
    ax.set_title('GCD main-head unknown-prob AUROC')
    _style_ax(ax)

    fig.suptitle('Branch1 - GCD Main Clustering Head', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir, 'branch1_gcd.png')

    eps, allv = series(d, 'test_acc', 0)
    if allv:
        best = max(range(len(allv)), key=lambda i: allv[i])
        print(f'[Branch1 GCD] 最佳 Test All acc = {allv[best]:.4f} @ epoch {eps[best]}')


def plot_branch2(d, out_dir):
    """OOD/OVA 分支: AUROC + 已知/未知筛选指标。"""
    metrics = [
        ('auroc',        'AUROC (OVA unknown score)', 0.5),
        ('det_acc',      'Det Acc@0.5 (known/unknown)', None),
        ('det_recall',   'Det Recall@0.5 (unknown)', None),
        ('det_spec',     'Det Spec@0.5 (known)', None),
        ('det_best_acc', 'Det Best Acc (Youden)', None),
        ('det_best_thr', 'Det Best Thr (Youden)', None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for i, (key, title, ref) in enumerate(metrics):
        ax = axes[i]
        eps, tv = series(d, key, 0)
        ax.plot(eps, tv, label='Train', color=COLOR_TRAIN, lw=2)
        eps, sv = series(d, key, 1)
        ax.plot(eps, sv, label='Test', color=COLOR_TEST, lw=2)
        if ref is not None:
            ax.axhline(ref, color='gray', ls='--', lw=1, alpha=0.7)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        _style_ax(ax)

    fig.suptitle('Branch2 - OOD/OVA Head (known/unknown detection)', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir, 'branch2_ood.png')

    eps, tv = series(d, 'auroc', 1)
    if tv:
        best = max(range(len(tv)), key=lambda i: tv[i])
        print(f'[Branch2 OOD] 最佳 Test AUROC = {tv[best]:.4f} @ epoch {eps[best]}')


def plot_branch3(d, out_dir):
    """去偏辅助分支: 聚类准确率。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for i, split in enumerate(['train_acc', 'test_acc']):
        ax = axes[i]
        for j in range(3):
            eps, vals = series(d, split, j)
            ax.plot(eps, vals, label=ACC_LABELS[j], color=ACC_COLORS[j], lw=2)
        ax.set_ylabel('Clustering Accuracy')
        ax.set_title('Debiased Acc - ' + ('Train' if split == 'train_acc' else 'Test'))
        _style_ax(ax)

    fig.suptitle('Branch3 - Debiased Auxiliary Head', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir, 'branch3_debiased.png')

    eps, allv = series(d, 'test_acc', 0)
    if allv:
        best = max(range(len(allv)), key=lambda i: allv[i])
        print(f'[Branch3 Debiased] 最佳 Test All acc = {allv[best]:.4f} @ epoch {eps[best]}')


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else 'train_expclip_0.1_v2.log'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'dev_outputs'
    os.makedirs(out_dir, exist_ok=True)

    data = parse_log(log_path)

    for b in (1, 2, 3):
        print(f'branch{b}: {len(data[b])} epochs')
    if not data[1]:
        print('[!] 未解析到任何 Branch 记录, 请检查日志路径')
        return

    plot_branch1(data[1], out_dir)
    plot_branch2(data[2], out_dir)
    plot_branch3(data[3], out_dir)


if __name__ == '__main__':
    main()
