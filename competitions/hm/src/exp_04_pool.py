"""実験04: 拡張後の候補プールを結合して recall@K と上限を測る。

比較対象は実験01のベースライン（現行7戦略・128件/人）:
   recall@12=0.0577  recall@100=0.1330  K無制限の上限=0.1433
目標は3位の recall ~18%。
"""
from __future__ import annotations

import sys
import polars as pl

from hm.config import VALID_CUTOFF, load_converted, CANDIDATE_DIR
from hm.metrics import get_actuals, evaluate_candidates
from hm.exp_log import log_experiment, timer
from hm.combine import combine
import hm.candidates as C

CUTOFF = VALID_CUTOFF
TOP_N = 100
KS = [12, 50, 100, 200, 300, 500, 1000]

trans, customers, articles = load_converted()
actuals = get_actuals(trans, CUTOFF)
valid_ids = pl.Series('customer_id', list(actuals.keys()), dtype=pl.Int32)
print(f'cutoff={CUTOFF}  valid顧客={len(actuals):,}')

PER_CUSTOMER = ['repurchase', 'timedecay', 'same_product_code', 'purchase_interval',
                'user_cf', 'also_bought', 'item2item_cf']
KEYED = {'popular': 'dummy', 'popular_by_age': 'age_bin', 'popular_by_channel': 'sales_channel_id'}

sources: dict[str, pl.DataFrame] = {}
for name in PER_CUSTOMER:
    df = C.load_cached(name, CUTOFF, TOP_N)
    if df is None:
        print(f'  [skip] {name}: 未生成')
        continue
    sources[name] = df.filter(pl.col('customer_id').is_in(valid_ids))

keymaps = {
    'dummy': customers.select(['customer_id', pl.lit(0, dtype=pl.Int32).alias('dummy')]),
    'age_bin': customers.select(['customer_id', pl.col('age_bin').fill_null(-1)]),
    'sales_channel_id': C.customer_channel_key(trans, CUTOFF),
}
for name, key in KEYED.items():
    df = C.load_cached(name, CUTOFF, TOP_N)
    if df is None:
        print(f'  [skip] {name}: 未生成')
        continue
    km = keymaps[key].filter(pl.col('customer_id').is_in(valid_ids))
    sources[name] = C.expand_keyed(df, km, key)

print(f'\n結合する戦略: {list(sources)}')

with timer() as t:
    pool = combine(sources, with_meta=False)
    res = evaluate_candidates(pool, actuals, ks=KS)
per_user = len(pool) / pool['customer_id'].n_unique()
print(f'\n候補/人={per_user:.1f}  総数={len(pool):,}行  ({t():.0f}s)')
print(f"{'K':>7}{'recall':>10}{'MAP':>10}   ベースライン(現行7戦略)")
base = {12: 0.0577, 100: 0.1330, 200: 0.1428, 300: 0.1433}
for row in res.to_dicts():
    b = base.get(row['k'])
    delta = f"  {b:.4f} -> {row['recall']:.4f}  ({row['recall']-b:+.4f})" if b else ''
    print(f"{row['k']:>7}{row['recall']:>10.4f}{row['map']:>10.5f}{delta}")

for k in (100, 300):
    r = [x for x in res.to_dicts() if x['k'] == k][0]
    log_experiment(name=f'pool_expanded/{len(sources)}strategies', stage='candidate',
                   recall_at_k=r['recall'], k=k, map_at_12=r['map'], runtime_sec=t(),
                   n_candidates_per_user=per_user,
                   note=f'cutoff={CUTOFF} top_n={TOP_N} 戦略={",".join(sources)}')

# 各戦略を1つずつ抜いたときの recall@100 の落ち込み（寄与の分解）
print('\n=== leave-one-out: 各戦略を抜いたときの recall@100 ===')
full = [x for x in res.to_dicts() if x['k'] == 100][0]['recall']
print(f"{'除外した戦略':<22}{'R@100':>9}{'差分':>10}")
for name in list(sources):
    sub = {k: v for k, v in sources.items() if k != name}
    r = evaluate_candidates(combine(sub, with_meta=False), actuals, ks=[100])['recall'][0]
    print(f'{name:<22}{r:>9.4f}{r-full:>10.4f}')
