"""実験03: user_cf の履歴窓を1変数だけ振る。

8位の原文には「顧客Xが A,B,C を購入」としか書かれておらず、履歴の長さは指定がない。
既定の30日だと被覆46.5%（9位の「顧客の47%のみ直近30日に取引あり」と一致）なので、
窓を伸ばして被覆と recall がどう動くかを実測する。
"""
from datetime import date
import polars as pl

from hm.config import VALID_CUTOFF, load_converted
from hm.metrics import get_actuals, evaluate_candidates
from hm.exp_log import log_experiment, timer
import hm.candidates as C

CUTOFF = VALID_CUTOFF
trans, customers, articles = load_converted()
actuals = get_actuals(trans, CUTOFF)
valid_ids = pl.Series('customer_id', list(actuals.keys()), dtype=pl.Int32)

print(f"{'history_days':>13}{'group_top':>11}{'n/user':>9}{'cov%':>7}{'R@12':>9}{'R@100':>9}{'MAP@12':>9}{'sec':>7}")
for history_days, group_top in [(30, 50), (90, 50), (180, 50), (None, 50), (30, 100), (90, 100)]:
    with timer() as t:
        df = C.build_user_cf(trans, CUTOFF, top_n=100, history_days=history_days,
                             group_top=group_top, cache=False)
    df = df.filter(pl.col('customer_id').is_in(valid_ids))
    res = {r['k']: r for r in evaluate_candidates(df, actuals, ks=[12, 100], score_col='score').to_dicts()}
    n_user = df['customer_id'].n_unique()
    per_user = len(df) / max(n_user, 1)
    label = 'all' if history_days is None else str(history_days)
    print(f'{label:>13}{group_top:>11}{per_user:>9.1f}{n_user/len(actuals)*100:>7.1f}'
          f'{res[12]["recall"]:>9.4f}{res[100]["recall"]:>9.4f}{res[12]["map"]:>9.5f}{t():>7.1f}')
    log_experiment(name=f'user_cf/hist={label},group_top={group_top}', stage='candidate',
                   recall_at_k=res[100]['recall'], k=100, map_at_12=res[12]['map'],
                   runtime_sec=t(), n_candidates_per_user=per_user,
                   note=f'cutoff={CUTOFF} 8位user based CF 履歴窓sweep cov={n_user/len(actuals)*100:.1f}%')
