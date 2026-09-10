"""実験20: u2tag2i（属性経由の個人化人気）の単体性能。

6位「u2tag2i: tag can be product_code, product_type_no, department_no, section_no」
11位「department_no ごとの人気を、その顧客が同じ department を買っていれば追加」

現状 tag=product_code は same_product_code として実装済み。残る属性を測る。
比較の目安（valid週・単体・top100）:
  popular_by_age  R@100 0.1267 / MAP@12 0.0094
  user_cf         R@100 0.1210 / MAP@12 0.0229
  same_product_code R@100 0.0452 / MAP@12 0.0069
  item2item_cf    R@100 0.0443 / MAP@12 0.0073
"""
from __future__ import annotations

import gc
import sys

import polars as pl

from hm.config import VALID_CUTOFF, load_converted
from hm.metrics import get_actuals, evaluate_candidates
from hm.exp_log import log_experiment, timer
import hm.candidates as C

CUTOFF = VALID_CUTOFF
TOP_N = 100
TAGS = sys.argv[1:] or ['department_no', 'product_type_no', 'section_no']

trans, customers, articles = load_converted()
actuals = get_actuals(trans, CUTOFF)
valid_ids = pl.Series('customer_id', list(actuals.keys()), dtype=pl.Int32)
print('valid顧客 {:,}'.format(len(actuals)), flush=True)

print('\n{:<20}{:>9}{:>7}{:>9}{:>9}{:>10}{:>8}'.format(
    'tag', 'n/user', 'cov%', 'R@12', 'R@100', 'MAP@12', 'sec'), flush=True)
for tag in TAGS:
    with timer() as t:
        df = C.build_u2tag2i(trans, articles, CUTOFF, tag, top_n=TOP_N)
    sub = df.filter(pl.col('customer_id').is_in(valid_ids))
    res = {r['k']: r for r in
           evaluate_candidates(sub, actuals, ks=[12, 100], score_col='score').to_dicts()}
    nu = sub['customer_id'].n_unique()
    print('{:<20}{:>9.1f}{:>7.1f}{:>9.4f}{:>9.4f}{:>10.5f}{:>8.0f}'.format(
        tag, len(sub) / max(nu, 1), nu / len(actuals) * 100,
        res[12]['recall'], res[100]['recall'], res[12]['map'], t()), flush=True)
    log_experiment(name='single_top100/u2tag_{}'.format(tag), stage='candidate',
                   recall_at_k=res[100]['recall'], k=100, map_at_12=res[12]['map'],
                   runtime_sec=t(), n_candidates_per_user=len(sub) / max(nu, 1),
                   note='6位のu2tag2i cutoff={} cov={:.1f}%'.format(CUTOFF, nu / len(actuals) * 100))
    del df, sub
    gc.collect()
