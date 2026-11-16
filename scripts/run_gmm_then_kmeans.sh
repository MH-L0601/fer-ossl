#!/usr/bin/env bash
# ============================================================
# 串联训练: 等待当前训练结束 -> GMM聚类训练 -> kmeans聚类训练
#
# 启动:  nohup bash scripts/run_gmm_then_kmeans.sh > train_chain.log 2>&1 &
# 查看:  tail -f train_chain.log train_expclip_gmm.log train_expclip_kmeans.log
# 停止:  pkill -f run_gmm_then_kmeans.sh ; pkill -f train_DebGCD.py
# ============================================================
set -u

cd /root/autodl-tmp/DebGCD-expclip
PY=/root/miniconda3/envs/debgcd/bin/python
CFG=config_expclip.json

# ---------- 1. 等待当前训练结束 ----------
# 匹配: train_DebGCD.py 进程中, 命令行不带 --cluster_method 的(即当前正在跑的训练)
echo "[chain] $(date '+%F %T') 等待当前训练(不带 cluster_method)结束..."
while pgrep -af "train_DebGCD.py" 2>/dev/null | grep -v "cluster_method" | grep -q "train_DebGCD.py"; do
    sleep 60
done
echo "[chain] $(date '+%F %T') 当前训练已结束, 开始串联训练。"

# ---------- 2. GMM 聚类训练 ----------
# 说明: gmm(full) 每 epoch 评估时聚类开销较大, 这里用 --graph_n_init 1 控制开销。
#       若想用对角协方差(更快), 把下面的 --cluster_method gmm 改成 gmm_diag。
echo "[chain] $(date '+%F %T') === 启动 GMM 聚类训练 ==="
${PY} train_DebGCD.py --config ${CFG} --cluster_method gmm --graph_n_init 1 \
    > train_expclip_gmm.log 2>&1
echo "[chain] $(date '+%F %T') GMM 训练结束 exit=$?"

# ---------- 3. kmeans 聚类训练 ----------
echo "[chain] $(date '+%F %T') === 启动 kmeans 聚类训练 ==="
${PY} train_DebGCD.py --config ${CFG} --cluster_method kmeans \
    > train_expclip_kmeans.log 2>&1
echo "[chain] $(date '+%F %T') kmeans 训练结束 exit=$?"

echo "[chain] $(date '+%F %T') 全部训练完成。"
