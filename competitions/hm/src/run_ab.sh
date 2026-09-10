#!/bin/bash
# A: 最適化重みで提出ファイルを作る
# B: u2tag込み13戦略で重みを再探索し、第2段で検証する
set -u
cd /c/kaggle_nvidia/kaggle_hm/competitions/hm/src
EXP=/c/kaggle_nvidia/kaggle_hm/competitions/hm/outputs/experiments
export PYTHONIOENCODING=utf-8

echo "########## A: 最適化重みで提出ファイル作成 $(date '+%H:%M:%S') ##########"
HM_WEIGHTS="$EXP/best_weights_opt.json" \
  python -u make_submission.py optweights_w6_k200 6 200 128 300 "$EXP/night_best_params.json"
echo "A DONE $(date '+%H:%M:%S')"

echo ""
echo "########## B-1: 13戦略で重み再探索（120試行・4週平均） $(date '+%H:%M:%S') ##########"
python -u exp_23_weight_optuna.py 120 true
echo "B-1 DONE $(date '+%H:%M:%S')"

echo ""
echo "########## B-2: 13戦略の第2段検証 $(date '+%H:%M:%S') ##########"
python -u exp_25_u2tag_weighted.py
echo "ALL DONE $(date '+%H:%M:%S')"
