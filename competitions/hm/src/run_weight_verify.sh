#!/bin/bash
# 重み探索(exp_23)の完走を待ってから、第2段での検証(exp_24)に進む
set -u
cd /c/kaggle_nvidia/kaggle_hm/competitions/hm/src
EXP=/c/kaggle_nvidia/kaggle_hm/competitions/hm/outputs/experiments
export PYTHONIOENCODING=utf-8

echo "探索の完了を待機中... $(date '+%H:%M:%S')"
while [ ! -f "$EXP/best_weights_opt.json" ]; do sleep 60; done
# ファイル生成直後は書き込み途中の可能性があるので少し置く
sleep 20
echo "探索完了を検知 $(date '+%H:%M:%S')"
cat "$EXP/best_weights_opt.json"

echo ""
echo "===== 第2段での検証 $(date '+%H:%M:%S') ====="
python -u exp_24_weight_rerank.py
echo "WEIGHT VERIFY DONE $(date '+%H:%M:%S')"
