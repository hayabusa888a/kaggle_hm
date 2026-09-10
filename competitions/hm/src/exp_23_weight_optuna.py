"""実験23: 候補結合の重みを Optuna で最適化する（複数週の平均で評価）。

いまの combine() は全戦略 1.0 の等重み。実験21/22 で、弱い戦略（u2tag）を
1つ足すだけで候補段階の MAP@12 が 0.0243 -> 0.0196 に崩れることが分かった。
min-max 正規化後のスコアを等しく足しているため、スコア分布が平坦な戦略が
既存の強い候補の相対順位を壊す。重みで抑えられるはず。

旧ノートブック実装では手動チューニングした重みを使っていた
（timedecay 3.5 / repurchase 1.5 など）。src/hm を書き起こしたときに
等重みへ戻し、以降検証していなかった。

**複数週の平均を目的にする。**
重みは候補スコアを直接いじるパラメータで、しかも自由度が戦略数ぶんある。
1週だけで合わせるとその週のノイズを拾い、valid週に二重に過学習してしまう
（valid週は他のすべての意思決定にも使っている）。
候補段階の評価は1週5秒程度と安いので、4週平均にして週をまたいで
安定して効く重みを選ぶ。週ごとのばらつきも記録する。

最適化する指標: recall@200 の4週平均
  第2段は候補200件を並べ替えるだけなので、その200件に正解が何個入っているかが
  第2段の性能の母数になる。候補段階の MAP@12 を目的にすると確実な戦略へ重みが
  集中して候補の多様性が削れるため、主目的には使わない（記録はする）。

使い方: python exp_23_weight_optuna.py [試行数] [with_u2tag]
"""
from __future__ import annotations

import gc
import json
import sys
from datetime import timedelta

import numpy as np
import optuna
import polars as pl

from hm.config import VALID_CUTOFF, EXPERIMENT_DIR, load_converted
from hm.metrics import get_actuals, evaluate_candidates
from hm.exp_log import log_experiment, timer
from hm.pipeline import build_sources, U2TAG_TAGS
from hm.combine import combine
import hm.candidates as C

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
WITH_U2TAG = len(sys.argv) > 2 and sys.argv[2].lower() in ('1', 'true', 'yes')
TOP_N = 100
KS = [12, 200, 300]
# valid週とその手前3週。すべて候補ファイルが揃っている cutoff。
CUTOFFS = [VALID_CUTOFF - timedelta(days=7 * i) for i in range(0, 4)]

trans, customers, articles = load_converted()

weeks = []   # (cutoff, actuals, sources)
for cut in CUTOFFS:
    acts = get_actuals(trans, cut)
    ids = pl.Series('customer_id', list(acts.keys()), dtype=pl.Int32)
    srcs = build_sources(trans, customers, articles, cut, TOP_N,
                         customer_filter=ids, build_missing=False)
    if WITH_U2TAG:
        for t in U2TAG_TAGS:
            nm = f'u2tag_{t}'
            df = C.load_cached(nm, cut, TOP_N, customer_filter=ids)
            if df is not None:
                srcs[nm] = df
    weeks.append((cut, acts, srcs))
    print('{}: 顧客{:,} 戦略{}'.format(cut, len(acts), len(srcs)), flush=True)

names = list(weeks[0][2])
del trans, customers, articles
gc.collect()
print('\n戦略({}): {}'.format(len(names), names), flush=True)


def evaluate(weights: dict) -> dict:
    """4週それぞれで測り、平均と週ごとの値を返す。"""
    per_week = []
    for _cut, acts, srcs in weeks:
        pool = combine(srcs, weights=weights, with_meta=False)
        res = {r['k']: r for r in evaluate_candidates(pool, acts, ks=KS).to_dicts()}
        per_week.append(res)
        del pool
        gc.collect()
    return {
        'r200': float(np.mean([r[200]['recall'] for r in per_week])),
        'r200_std': float(np.std([r[200]['recall'] for r in per_week])),
        'map12': float(np.mean([r[12]['map'] for r in per_week])),
        'r12': float(np.mean([r[12]['recall'] for r in per_week])),
        'r300': float(np.mean([r[300]['recall'] for r in per_week])),
        'weekly': [round(r[200]['recall'], 4) for r in per_week],
    }


print('\n=== 基準（全戦略1.0） ===', flush=True)
with timer() as t:
    baseline = evaluate({n: 1.0 for n in names})
print('R@200 平均={:.4f} (週ごと {}) 標準偏差={:.4f}'.format(
    baseline['r200'], baseline['weekly'], baseline['r200_std']), flush=True)
print('MAP@12 平均={:.5f}  R@12 平均={:.4f}  ({:.0f}s/試行)'.format(
    baseline['map12'], baseline['r12'], t()), flush=True)
BASE_R200, BASE_MAP = baseline['r200'], baseline['map12']


def objective(trial: optuna.Trial) -> float:
    w = {n: trial.suggest_float(n, 0.0, 3.0) for n in names}
    if sum(w.values()) <= 0:
        return 0.0
    m = evaluate(w)
    for k in ('map12', 'r12', 'r300', 'r200_std'):
        trial.set_user_attr(k, m[k])
    trial.set_user_attr('weekly', m['weekly'])
    return m['r200']


optuna.logging.set_verbosity(optuna.logging.WARNING)
# study を SQLite に置く。途中で落ちても全試行が残り、後から上位を取り出せる。
# （夜間の exp_16 を打ち切ったとき、ログに出していないパラメータが失われた反省）
_db = EXPERIMENT_DIR / ('weights_u2tag.db' if WITH_U2TAG else 'weights.db')
study = optuna.create_study(
    direction='maximize', study_name='weights', load_if_exists=True,
    storage='sqlite:///{}'.format(_db.as_posix()),
    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=15))
study.enqueue_trial({n: 1.0 for n in names})
legacy = {'timedecay': 3.5, 'repurchase': 1.5, 'popular_by_age': 1.5}
study.enqueue_trial({n: legacy.get(n, 1.0) for n in names})

best = [0.0]
def cb(st, tr):
    if tr.value is not None and tr.value > best[0]:
        best[0] = tr.value
        print('  trial {:>3}  R@200={:.4f}({:+.4f})  MAP@12={:.5f}({:+.5f})  週ごと{}  <- 更新'.format(
            tr.number, tr.value, tr.value - BASE_R200,
            tr.user_attrs.get('map12', 0), tr.user_attrs.get('map12', 0) - BASE_MAP,
            tr.user_attrs.get('weekly')), flush=True)
    elif tr.number % 20 == 0:
        print('  trial {:>3}  R@200={:.4f}'.format(tr.number, tr.value or 0), flush=True)

with timer() as t:
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[cb])

b = study.best_trial
print('\n=== 最良（trial {}） ==='.format(b.number), flush=True)
print('R@200  平均 {:.4f}  (基準 {:.4f} から {:+.4f})'.format(
    b.value, BASE_R200, b.value - BASE_R200), flush=True)
print('  週ごと {}  標準偏差 {:.4f}'.format(
    b.user_attrs['weekly'], b.user_attrs['r200_std']), flush=True)
print('MAP@12 平均 {:.5f} (基準 {:.5f} から {:+.5f})'.format(
    b.user_attrs['map12'], BASE_MAP, b.user_attrs['map12'] - BASE_MAP), flush=True)
print('R@12   平均 {:.4f}'.format(b.user_attrs['r12']), flush=True)
print('\n重み:', flush=True)
for n in names:
    print('  {:<24}{:.3f}'.format(n, b.params[n]), flush=True)

print('\n=== 上位8試行（recall@200 順） ===', flush=True)
_done = [tr for tr in study.trials if tr.value is not None]
print('  {:>6}{:>10}{:>11}{:>10}'.format('trial', 'R@200', 'MAP@12', 'R@12'), flush=True)
for tr in sorted(_done, key=lambda x: -x.value)[:8]:
    print('  {:>6}{:>10.4f}{:>11.5f}{:>10.4f}'.format(
        tr.number, tr.value, tr.user_attrs.get('map12', 0),
        tr.user_attrs.get('r12', 0)), flush=True)

print('\n=== 候補MAP@12 が高い上位8試行 ===', flush=True)
print('  {:>6}{:>10}{:>11}{:>10}'.format('trial', 'R@200', 'MAP@12', 'R@12'), flush=True)
for tr in sorted(_done, key=lambda x: -x.user_attrs.get('map12', 0))[:8]:
    print('  {:>6}{:>10.4f}{:>11.5f}{:>10.4f}'.format(
        tr.number, tr.value, tr.user_attrs.get('map12', 0),
        tr.user_attrs.get('r12', 0)), flush=True)
    print('        重み={}'.format(
        {k: round(v, 2) for k, v in tr.params.items()}), flush=True)

out = {'cutoffs': [str(c) for c in CUTOFFS], 'with_u2tag': WITH_U2TAG,
       'n_trials': N_TRIALS, 'weights': {n: b.params[n] for n in names},
       'recall_200_mean': b.value, 'recall_200_weekly': b.user_attrs['weekly'],
       'map_12_mean': b.user_attrs['map12'],
       'baseline': {'recall_200_mean': BASE_R200, 'map_12_mean': BASE_MAP}}
path = EXPERIMENT_DIR / ('best_weights_u2tag.json' if WITH_U2TAG else 'best_weights_opt.json')
path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print('\n保存: {}  ({:.0f}s)'.format(path, t()), flush=True)

log_experiment(name='weight_optuna4w/{}strategies'.format(len(names)), stage='candidate',
               recall_at_k=b.value, k=200, map_at_12=b.user_attrs['map12'],
               runtime_sec=t(), note='Optuna {}試行 4週平均 u2tag={} 重み={}'.format(
                   N_TRIALS, WITH_U2TAG, {n: round(b.params[n], 3) for n in names}))
