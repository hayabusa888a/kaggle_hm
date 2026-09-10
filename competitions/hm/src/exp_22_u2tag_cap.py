"""実験22: u2tag の候補数上限を振って、押し出しが起きない点を探す。

実験21で u2tag を top100（実効95件/人）で足したところ、
  プール上限   0.2689 -> 0.3047 (+0.0358)  … 新しい正解は持ち込めている
  候補MAP@12   0.02433 -> 0.01961 (-0.00472) … 既存の良質な候補を押し出した
  再rank MAP@12 0.04153 -> 0.04128 (-0.00025) … 第2段でも回収しきれず

原因は「u2tag のスコアが弱いのに候補を出しすぎた」こと。
上限を絞れば、独自貢献（department_no で2,152ペア）だけ残せるはず。

候補ファイルの再生成は不要。_finish がスコア順に rank を振っているので、
top100 のファイルを rank<=N で切れば top_n=N で作ったものと同一になる。
"""
from __future__ import annotations

import gc

import polars as pl

from hm.config import VALID_CUTOFF, load_converted
from hm.metrics import get_actuals, evaluate_candidates, hits_at_k
from hm.exp_log import log_experiment, timer
from hm.pipeline import build_sources
from hm.combine import combine
import hm.candidates as C

CUTOFF = VALID_CUTOFF
TOP_N = 100
KS = [12, 100, 200, 300, 1000]
U2TAG = ['u2tag_section_no', 'u2tag_department_no', 'u2tag_product_type_no']
CAPS = [0, 5, 10, 20, 30, 50, 100]   # 0 = u2tag を使わない（10戦略）

trans, customers, articles = load_converted()
actuals = get_actuals(trans, CUTOFF)
valid_ids = pl.Series('customer_id', list(actuals.keys()), dtype=pl.Int32)
print('valid顧客 {:,}'.format(len(actuals)), flush=True)

base = {k: v for k, v in build_sources(
    trans, customers, articles, CUTOFF, TOP_N,
    customer_filter=valid_ids, build_missing=False).items() if k not in U2TAG}
print('既存戦略 {} 個'.format(len(base)), flush=True)

raw = {}
for name in U2TAG:
    raw[name] = C.load_cached(name, CUTOFF, TOP_N, customer_filter=valid_ids)

print('\n{:>5}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}'.format(
    'cap', 'n/user', 'R@12', 'MAP@12', 'R@200', '上限', 'Hit@200'), flush=True)
print('  基準(10戦略): R@12=0.0638 MAP@12=0.02433 R@200=0.2226 上限=0.2689', flush=True)
for cap in CAPS:
    srcs = dict(base)
    if cap > 0:
        for name, df in raw.items():
            srcs[name] = df.filter(pl.col('rank') <= cap)
    with timer() as t:
        pool = combine(srcs, with_meta=False)
        res = {r['k']: r for r in evaluate_candidates(pool, actuals, ks=KS).to_dicts()}
        h200 = hits_at_k(pool, actuals, 200)
    per_user = len(pool) / pool['customer_id'].n_unique()
    print('{:>5}{:>10.1f}{:>10.4f}{:>10.5f}{:>10.4f}{:>10.4f}{:>10,}'.format(
        cap, per_user, res[12]['recall'], res[12]['map'],
        res[200]['recall'], res[1000]['recall'], h200), flush=True)
    log_experiment(name='u2tag_cap/{}'.format(cap), stage='candidate',
                   recall_at_k=res[200]['recall'], k=200, map_at_12=res[12]['map'],
                   runtime_sec=t(), n_candidates_per_user=per_user,
                   note='u2tag上限={} 上限recall={:.4f} Hit@200={:,}'.format(
                       cap, res[1000]['recall'], h200))
    del pool, srcs
    gc.collect()
