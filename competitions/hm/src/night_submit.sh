#!/bin/bash
# 夜間ステージE: B（パラメータ）とD（週数）の結果を読んで提出ファイルを作る。
set -u
cd /c/kaggle_nvidia/kaggle_hm/competitions/hm/src
EXP=/c/kaggle_nvidia/kaggle_hm/competitions/hm/outputs/experiments
export PYTHONIOENCODING=utf-8

# --- B の結果から lr / leaves / rounds を取る（無ければ今日の最良）
read LR LEAVES ROUNDS < <(python - <<'PY'
import json
from pathlib import Path
p = Path(r'C:\kaggle_nvidia\kaggle_hm\competitions\hm\outputs\experiments\night_best_params.json')
d = {'params': {'learning_rate': 0.0188, 'num_leaves': 188}, 'rounds': 796}
if p.exists():
    try:
        d = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        pass
print(d['params'].get('learning_rate', 0.0188),
      d['params'].get('num_leaves', 188),
      d.get('rounds', 796))
PY
)

# --- D の結果から最良の週数を取る（無ければ6）
WEEKS=$(python - <<'PY'
import re
from pathlib import Path
p = Path(r'C:\kaggle_nvidia\kaggle_hm\competitions\hm\outputs\experiments\night_D_weeks.log')
best_w, best_m = 6, -1.0
if p.exists():
    for line in p.read_text(encoding='utf-8', errors='replace').splitlines():
        m = re.match(r'\s*(\d+)\s+[\d,]+\s+[\d.]+\s+([\d.]+)\s+\d+\s*$', line)
        if m and float(m.group(2)) > best_m:
            best_w, best_m = int(m.group(1)), float(m.group(2))
print(best_w)
PY
)

echo "採用構成: 学習${WEEKS}週 / 候補200 / BPR(128,300) / LGB(lr=$LR, leaves=$LEAVES, rounds=$ROUNDS)"
python -u make_submission.py night_best "$WEEKS" 200 128 300 "$LR" "$LEAVES" "$ROUNDS"
