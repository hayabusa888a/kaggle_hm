"""実験02: 候補プールの拡張。

根拠（第3部原文）
 - 3位: 「popular items 30->100, itemcf 20->50 のように各戦略の recall 数を増やし、
   1顧客あたりの候補が数百になった結果 recall rate が ~10% から ~18% に上がった」
 - 8位: 「I select top 100 candidates for all candidate type」「重複は除去する」
 - 8位コメント: user based CF は各集団の直近1週売上を**単純加算**して上位を取る
 - 9位: valid週の商品の92%は直近30日にも出現 / 顧客の47%のみ直近30日に取引あり

現状 recall@100=0.1330、K無制限でも 0.1433 が上限だったので、まず上限を上げる。
"""
from __future__ import annotations

import sys
from datetime import date

import polars as pl

from hm.config import CONVERTED_DIR, VALID_CUTOFF, load_converted
from hm.metrics import get_actuals, evaluate_candidates
from hm.exp_log import log_experiment, timer
import hm.candidates as C

CUTOFF = VALID_CUTOFF
TOP_N = 100
KS = [12, 100, 200, 300]

trans, customers, articles = load_converted()
actuals = get_actuals(trans, CUTOFF)
valid_ids = pl.Series('customer_id', list(actuals.keys()), dtype=pl.Int32)
print(f'cutoff={CUTOFF}  valid顧客={len(actuals):,}')

customer_age = customers.select(['customer_id', pl.col('age_bin').fill_null(-1)])
customer_dummy = customers.select(['customer_id', pl.lit(0, dtype=pl.Int32).alias('dummy')])

BUILDERS = {
    'repurchase':        lambda: C.build_repurchase(trans, CUTOFF, TOP_N),
    'timedecay':         lambda: C.build_timedecay(trans, CUTOFF, TOP_N),
    'same_product_code': lambda: C.build_same_product_code(trans, articles, CUTOFF, TOP_N),
    'purchase_interval': lambda: C.build_purchase_interval(trans, CUTOFF, TOP_N),
    'user_cf':           lambda: C.build_user_cf(trans, CUTOFF, TOP_N),
    'also_bought':       lambda: C.build_also_bought(trans, CUTOFF, TOP_N),
    'item2item_cf':      lambda: C.build_item2item_cf(trans, CUTOFF, TOP_N),
    'popular':           lambda: C.build_popular(trans, CUTOFF, TOP_N),
    'popular_by_age':    lambda: C.build_popular_by_age(trans, customers, CUTOFF, TOP_N),
    'popular_by_channel':lambda: C.build_popular_by_channel(trans, CUTOFF, TOP_N),
}

KEYS = {
    'popular': (customer_dummy, 'dummy'),
    'popular_by_age': (customer_age, 'age_bin'),
    'popular_by_channel': (None, 'sales_channel_id'),   # 遅延構築
}

only = sys.argv[1:] or list(BUILDERS)
print(f"\n{'strategy':<20}{'n/user':>9}{'cov%':>7}{'R@12':>9}{'R@100':>9}{'MAP@12':>9}{'build_s':>9}")
for name in only:
    with timer() as t:
        df = BUILDERS[name]()
    build_sec = t()
    if name in KEYS:
        keys, col = KEYS[name]
        if keys is None:
            keys = C.customer_channel_key(trans, CUTOFF)
        df = C.expand_keyed(df, keys.filter(pl.col('customer_id').is_in(valid_ids)), col)
    else:
        df = df.filter(pl.col('customer_id').is_in(valid_ids))
    res = {r['k']: r for r in evaluate_candidates(df, actuals, ks=KS, score_col='score').to_dicts()}
    n_user = df['customer_id'].n_unique()
    per_user = len(df) / max(n_user, 1)
    print(f'{name:<20}{per_user:>9.1f}{n_user/len(actuals)*100:>7.1f}'
          f'{res[12]["recall"]:>9.4f}{res[100]["recall"]:>9.4f}{res[12]["map"]:>9.5f}{build_sec:>9.1f}')
    log_experiment(name=f'single_top{TOP_N}/{name}', stage='candidate',
                   recall_at_k=res[100]['recall'], k=100, map_at_12=res[12]['map'],
                   runtime_sec=build_sec, n_candidates_per_user=per_user,
                   note=f'cutoff={CUTOFF} top_n={TOP_N} 単体')
