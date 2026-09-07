"""実験06: also_bought / item2item_cf の履歴窓。user_cf で効いた1変数を横展開する。

user_cf では history_days 30->90 で R@100 0.086->0.121（被覆46.5%->73.2%）だった。
i2i 系も既定30日で被覆が44%前後に留まっているので、同じ操作の効果を確認する。
"""
import polars as pl

from hm.config import VALID_CUTOFF, load_converted
from hm.metrics import get_actuals, evaluate_candidates
from hm.exp_log import log_experiment, timer
import hm.candidates as C

CUTOFF = VALID_CUTOFF
trans, customers, articles = load_converted()
actuals = get_actuals(trans, CUTOFF)
valid_ids = pl.Series('customer_id', list(actuals.keys()), dtype=pl.Int32)

print(f"{'strategy':<16}{'hist_days':>10}{'n/user':>9}{'cov%':>7}{'R@12':>9}{'R@100':>9}{'MAP@12':>9}{'sec':>7}")
for name, fn in [('also_bought', C.build_also_bought), ('item2item_cf', C.build_item2item_cf)]:
    for hd in (30, 90):
        with timer() as t:
            df = fn(trans, CUTOFF, top_n=100, history_days=hd, cache=False)
        df = df.filter(pl.col('customer_id').is_in(valid_ids))
        res = {r['k']: r for r in evaluate_candidates(df, actuals, ks=[12, 100], score_col='score').to_dicts()}
        n_user = df['customer_id'].n_unique()
        print(f'{name:<16}{hd:>10}{len(df)/n_user:>9.1f}{n_user/len(actuals)*100:>7.1f}'
              f'{res[12]["recall"]:>9.4f}{res[100]["recall"]:>9.4f}{res[12]["map"]:>9.5f}{t():>7.1f}')
        log_experiment(name=f'{name}/hist={hd}', stage='candidate',
                       recall_at_k=res[100]['recall'], k=100, map_at_12=res[12]['map'],
                       runtime_sec=t(), n_candidates_per_user=len(df)/n_user,
                       note=f'cutoff={CUTOFF} i2i履歴窓sweep cov={n_user/len(actuals)*100:.1f}%')
