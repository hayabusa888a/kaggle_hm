"""実験08: 学習週数を1変数として振る。

根拠（第3部原文）
 - 3位「18weeks vs 2weeks is about 0.0005-0.0010 up」「More data can reduce overfitting」
 - 1位「We use 6 weeks data as train, last week as valid, retrieve 100 candidates for
   each user, it has stable cv-lb correlation」cv 0.0430 / lb 0.0362
 - 6位「We use last 9 weeks for training(last week for validation)」
 - 9位「LightGBM trained with 20 weeks of data」
 - 一方 11位「changing the training weeks gave a negative correlation
   (worse cv score, better public/private score)」、13位「増やすと Model2/3 は悪化」
   -> 週数だけはCVを鵜呑みにできない。増やした構成は必ず提出して実測すること。

現状は1週しか使っていないので、ここが最大の伸びしろ。
実験05の最良構成（meta=rank, categorical明示なし）を固定し、週数だけ変える。
"""
from __future__ import annotations

import gc
import sys
from datetime import date, timedelta

import polars as pl

from hm.config import VALID_CUTOFF, load_converted
from hm.metrics import get_actuals, evaluate_ranking
from hm.exp_log import log_experiment, timer
from hm.pipeline import build_dataset, feature_columns
from hm.rerank import downsample_negatives, train_lgb_binary, predict

VALID = VALID_CUTOFF
TOP_K = 100
NEG_RATIO = 30.0
# valid の1週前から遡る
ALL_CUTOFFS = [VALID - timedelta(days=7 * i) for i in range(1, 7)]
WEEK_COUNTS = [int(x) for x in (sys.argv[1:] or ['1', '2', '4', '6'])]

trans, customers, articles = load_converted()

# valid は実験05でキャッシュ済み（cache_tag='valid'）なので即座に返る
va = get_actuals(trans, VALID)
valid = build_dataset(trans, customers, articles, VALID, actuals=va, top_k=TOP_K,
                      customer_filter=pl.Series('customer_id', list(va.keys()), dtype=pl.Int32),
                      n_chunks=8, cache_tag='valid')
print(f'valid: {len(valid):,}行 正例{valid["label"].sum():,}', flush=True)
cols = feature_columns(valid)

folds: dict[date, pl.DataFrame] = {}
def get_fold(cutoff: date) -> pl.DataFrame:
    if cutoff not in folds:
        a = get_actuals(trans, cutoff)
        folds[cutoff] = build_dataset(
            trans, customers, articles, cutoff, actuals=a, top_k=TOP_K,
            customer_filter=pl.Series('customer_id', list(a.keys()), dtype=pl.Int32),
            sample_fn=lambda d: downsample_negatives(d, NEG_RATIO),
            n_chunks=4, cache_tag=f'train_neg{NEG_RATIO:.0f}')
    return folds[cutoff]

print(f"\n{'weeks':>6}{'rows':>12}{'pos':>9}{'R@12':>9}{'MAP@12':>10}{'LB換算':>9}{'sec':>7}")
for n in WEEK_COUNTS:
    cuts = ALL_CUTOFFS[:n]
    with timer() as t:
        train = pl.concat([get_fold(c).select(cols + ['label']) for c in cuts], how='vertical')
        model, cats = train_lgb_binary(train, cols, categorical=[])   # 実験05で明示指定は悪化
        pred = predict(model, valid, cols)
        res = {r['k']: r for r in evaluate_ranking(pred, va, ks=[12]).to_dicts()}
    m = res[12]['map']
    print(f'{n:>6}{len(train):>12,}{train["label"].sum():>9,}'
          f'{res[12]["recall"]:>9.4f}{m:>10.5f}{t():>7.0f}', flush=True)
    log_experiment(name=f'rerank/train_weeks={n}', stage='rerank',
                   recall_at_k=res[12]['recall'], k=12, map_at_12=m, runtime_sec=t(),
                   n_candidates_per_user=TOP_K,
                   note=f'LGB binary meta=rank cat=none neg{NEG_RATIO:.0f}x '
                        f'cutoffs={[str(c) for c in cuts]} valid={VALID}')
    del train, model, pred; gc.collect()
