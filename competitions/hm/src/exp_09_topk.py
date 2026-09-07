"""実験09: 1顧客あたりの候補数 top_k を振る。

根拠（第3部原文）
 - 3位「I tried to increase the recall num of each candidate strategy, such as popular
   items 30->100, itemcf 20->50 and so on. As a result, the total count of candidates
   for each user increased to hundreds and the recall rates increased from ~10% to ~18%」
   ただし「if I only increase the recall num and don't adding the recall features,
   the CV score is very very poor」-> rank メタ特徴は既に入れてあるので条件を満たす
 - 9位「Generate 200 candidates for each customer(学習) / 400 candidates(推論)」
 - 6位は逆に候補120 vs 220 でほぼ差がない（catboost binary 0.0403 vs 0.0402）
   -> チームで割れているので自分のデータで確認する

実測済みの recall（valid週・10戦略）:
   @100 = 0.1627 / @200 = 0.2226 / @300 = 0.2541 / 上限 = 0.2689
候補100では recall の6割しか使えていない。

使い方: python exp_09_topk.py <週数> <top_k> [<top_k> ...]
"""
from __future__ import annotations

import gc
import sys
from datetime import timedelta

import polars as pl

from hm.config import VALID_CUTOFF, load_converted
from hm.metrics import get_actuals, evaluate_ranking
from hm.exp_log import log_experiment, timer
from hm.pipeline import build_dataset, feature_columns
from hm.rerank import downsample_negatives, train_lgb_binary, predict

VALID = VALID_CUTOFF
NEG_RATIO = 30.0
N_WEEKS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TOP_KS = [int(x) for x in (sys.argv[2:] or ['200'])]
CUTS = [VALID - timedelta(days=7 * i) for i in range(1, N_WEEKS + 1)]

trans, customers, articles = load_converted()
va = get_actuals(trans, VALID)
valid_ids = pl.Series('customer_id', list(va.keys()), dtype=pl.Int32)

# 比較の基準（実測済み）: 学習1週 top_k=100 -> CV 0.03656 -> public 0.03037（比 0.831）
print('学習週: ' + ', '.join(str(c) for c in CUTS), flush=True)
print("\n{:>7}{:>12}{:>9}{:>9}{:>10}{:>7}".format(
    'top_k', 'train行', '正例', 'R@12', 'MAP@12', 'sec'), flush=True)

for k in TOP_KS:
    with timer() as t:
        valid = build_dataset(trans, customers, articles, VALID, actuals=va, top_k=k,
                              customer_filter=valid_ids, n_chunks=16, cache_tag='valid')
        cols = feature_columns(valid)
        parts = []
        for c in CUTS:
            a = get_actuals(trans, c)
            parts.append(build_dataset(
                trans, customers, articles, c, actuals=a, top_k=k,
                customer_filter=pl.Series('customer_id', list(a.keys()), dtype=pl.Int32),
                sample_fn=lambda d: downsample_negatives(d, NEG_RATIO),
                n_chunks=8, cache_tag='train_neg30').select(cols + ['label']))
        train = pl.concat(parts, how='vertical')
        del parts; gc.collect()
        model, _ = train_lgb_binary(train, cols, categorical=[])
        pred = predict(model, valid, cols)
        res = {r['k']: r for r in evaluate_ranking(pred, va, ks=[12]).to_dicts()}
    m = res[12]['map']
    print('{:>7}{:>12,}{:>9,}{:>9.4f}{:>10.5f}{:>7.0f}'.format(
        k, len(train), train['label'].sum(), res[12]['recall'], m, t()), flush=True)
    log_experiment(name='rerank/top_k={},weeks={}'.format(k, N_WEEKS), stage='rerank',
                   recall_at_k=res[12]['recall'], k=12, map_at_12=m, runtime_sec=t(),
                   n_candidates_per_user=k,
                   note='LGB binary meta=rank cat=none neg30x 学習{}週 valid={}'.format(
                       N_WEEKS, VALID))
    del train, valid, model, pred; gc.collect()
