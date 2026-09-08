"""夜間ステージB: Optuna再探索の上位パラメータが再現するか検証する。

今日 trial 10 が探索時0.04141 -> 再現時0.04096 と落ちた（ログに全パラメータを
出していなかったため）。修正版 exp_16 は params= 行を出すので、そこから
完全なパラメータを復元して測り直し、**再現できたものだけ**を採用候補にする。

結果は outputs/experiments/night_best_params.json に書く。
ステージEの提出作成がこれを読む。
"""
from __future__ import annotations

import ast
import gc
import json
import re
from datetime import timedelta
from pathlib import Path

import polars as pl

from hm.config import VALID_CUTOFF, EXPERIMENT_DIR, load_converted
from hm.metrics import get_actuals, evaluate_ranking
from hm.exp_log import log_experiment, timer
from hm.pipeline import dataset_path, feature_columns
from hm.embeddings import add_bpr_similarity, add_user2item_similarity
from hm.rerank import train_lgb_binary, predict

TOP_K, BPR_DIM, BPR_ITERS, N_WEEKS = 200, 128, 300, 2
VALID = VALID_CUTOFF
CUTS = [VALID - timedelta(days=7 * i) for i in range(1, N_WEEKS + 1)]
LOG = EXPERIMENT_DIR / 'night_A_optuna.log'
OUT = EXPERIMENT_DIR / 'night_best_params.json'

# 今日確定している最良（これを下回るなら採用しない）
FALLBACK = {'params': {'learning_rate': 0.0188, 'num_leaves': 188},
            'rounds': 796, 'map': 0.04128, 'source': 'exp_17 optuna_t3'}

# --- Aのログから (MAP, rounds, params) を拾う
head = re.compile(r'trial\s+(\d+)\s+MAP@12=([\d.]+).*rounds=(\d+)')
trials = []
if LOG.exists():
    lines = LOG.read_text(encoding='utf-8', errors='replace').splitlines()
    for i, line in enumerate(lines):
        m = head.search(line)
        if not m:
            continue
        params = None
        for j in range(i + 1, min(i + 3, len(lines))):
            if 'params=' in lines[j]:
                try:
                    params = ast.literal_eval(lines[j].split('params=', 1)[1].strip())
                except Exception:
                    params = None
                break
        if params:
            params.pop('seed', None)
            trials.append({'trial': int(m.group(1)), 'map': float(m.group(2)),
                           'rounds': int(m.group(3)), 'params': params})
trials.sort(key=lambda r: -r['map'])
top = trials[:3]
print('Aのログから {} 試行を読み、上位{}件を検証する'.format(len(trials), len(top)), flush=True)
for t in top:
    print('  trial {} 探索時MAP={:.5f}'.format(t['trial'], t['map']), flush=True)

if not top:
    print('検証対象なし。今日の最良をそのまま使う', flush=True)
    OUT.write_text(json.dumps(FALLBACK, ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(0)

trans, customers, articles = load_converted()
n_items = int(articles['article_id'].max()) + 1
va = get_actuals(trans, VALID)


def load(cutoff, tag):
    df = pl.read_parquet(dataset_path(cutoff, TOP_K, tag))
    df = add_user2item_similarity(df, trans, cutoff, n_items)
    return add_bpr_similarity(df, trans, cutoff, n_items, dim=BPR_DIM, iterations=BPR_ITERS)


valid = load(VALID, 'valid')
cols = feature_columns(valid)
train = pl.concat([load(c, 'train_neg30').select(cols + ['label']) for c in CUTS],
                  how='vertical')
del articles, customers
gc.collect()

best = dict(FALLBACK)
print("\n{:>7}{:>12}{:>12}{:>12}{:>7}".format('trial', '探索時', '再現', '差', 'sec'), flush=True)
for t in top:
    with timer() as tm:
        model, _ = train_lgb_binary(train, cols, categorical=[], params=t['params'],
                                    num_boost_round=t['rounds'])
        pred = predict(model, valid, cols)
        m = evaluate_ranking(pred, va, ks=[12])['map'][0]
    print('{:>7}{:>12.5f}{:>12.5f}{:>+12.5f}{:>7.0f}'.format(
        t['trial'], t['map'], m, m - t['map'], tm()), flush=True)
    log_experiment(name='night_verify/trial{}'.format(t['trial']), stage='rerank',
                   map_at_12=m, k=12, runtime_sec=tm(), n_candidates_per_user=TOP_K,
                   note='再現検証 探索時={:.5f} params={}'.format(t['map'], t['params']))
    if m > best['map']:
        best = {'params': t['params'], 'rounds': t['rounds'], 'map': m,
                'source': 'night_A trial {}'.format(t['trial'])}
    del model, pred
    gc.collect()

print('\n採用: {} (MAP@12={:.5f})'.format(best['source'], best['map']))
print(json.dumps(best, ensure_ascii=False, indent=2))
OUT.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding='utf-8')
