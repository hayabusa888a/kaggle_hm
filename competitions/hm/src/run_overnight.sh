#!/bin/bash
# 夜間の無人実行。各ステージを直列に回す（同時実行はメモリを奪い合うため厳禁）。
# 各段階の数値は outputs/experiments/experiment_log.csv に追記される。
set -u
cd /c/kaggle_nvidia/kaggle_hm/competitions/hm/src
EXP=/c/kaggle_nvidia/kaggle_hm/competitions/hm/outputs/experiments
export PYTHONIOENCODING=utf-8
PY=python

# stage <名前> <タイムアウト秒> <ログパス> <コマンド...>
# 進捗はマスターログへ、コマンドの出力は個別ログへ分ける。
# （最初の版は呼び出し側で > を書いたため、進捗表示まで個別ログに流れてしまった）
stage () {
  local name=$1; shift
  local tmo=$1; shift
  local log=$1; shift
  echo ""
  echo "==================== [$name] 開始 $(date '+%H:%M:%S') ===================="
  timeout "$tmo" "$@" >> "$log" 2>&1
  local rc=$?
  if [ $rc -eq 124 ]; then
    echo "[$name] タイムアウト(${tmo}s)で打ち切り"
  elif [ $rc -ne 0 ]; then
    echo "[$name] 異常終了 rc=$rc"
  else
    echo "[$name] 正常終了"
  fi
  echo "==================== [$name] 終了 $(date '+%H:%M:%S') ===================="
}

echo "夜間実行 開始 $(date '+%Y-%m-%d %H:%M:%S')"

# --- A: Optuna 再探索（全パラメータをログに出す修正版で20試行）
stage "A_optuna" 9000 $EXP/night_A_optuna.log $PY -u exp_16_optuna.py 20 2 200

# --- B: 上位パラメータの再現性検証（Aのログから上位3件を拾って測り直す）
stage "B_verify" 2400 $EXP/night_B_verify.log $PY -u night_verify.py

# --- C: 学習週を10週に伸ばすための候補生成とデータセット構築
for C in 2020-07-28 2020-07-21 2020-07-14 2020-07-07; do
  stage "C_cand_$C" 1800 $EXP/night_C_build.log $PY -u build_candidates.py $C
done
for C in 2020-08-04 2020-07-28 2020-07-21 2020-07-14 2020-07-07; do
  stage "C_k200_$C" 1800 $EXP/night_C_build.log $PY -u build_k200.py $C
done

# --- D: 学習週 6 -> 8 -> 10
stage "D_weeks" 5400 $EXP/night_D_weeks.log $PY -u exp_18_train_weeks_long.py 6 8 10

# --- E: 最良構成で提出ファイル作成（Bで確定したパラメータ、Dで確定した週数を使う）
stage "E_submit" 12000 $EXP/night_E_submit.log bash night_submit.sh

echo ""
echo "夜間実行 終了 $(date '+%Y-%m-%d %H:%M:%S')"
