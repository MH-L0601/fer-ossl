#!/bin/bash
# DebGCD + ResNet50: 用 ResNet50 骨干 (feat_dim=2048) 替换 Exp-CLIP。
# 未知类 = 3 (Happiness), 已知类 = [0,1,2,4,5,6]。
# warmup 权重: rafdb_known6_resnet50_pretrain.pth (已知6类监督预训练)。
# 说明: resnet50 不使用 text_prototypes, transform 用 imagenet。
#
# 用法:
#   bash scripts/train_rafdb_resnet50.sh
# 或防终端清理挂后台:
#   nohup bash scripts/train_rafdb_resnet50.sh > train_rafdb_resnet50.log 2>&1 &
cd "$(dirname "$0")/.."

/root/miniconda3/envs/debgcd/bin/python train_DebGCD.py --config config_resnet50.json "$@"
