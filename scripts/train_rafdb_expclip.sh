#!/bin/bash
# DebGCD + Exp-CLIP: frozen CLIP(B/16)+head as backbone, text prototypes as
# known-class clustering centers. Unknown class = 3 (Happiness), original RAF-DB
# numbering — automatic alignment handled inside expclip/text_prototypes.py.
cd "$(dirname "$0")/.."

/root/miniconda3/envs/debgcd/bin/python train_DebGCD.py --config config_expclip.json "$@"
