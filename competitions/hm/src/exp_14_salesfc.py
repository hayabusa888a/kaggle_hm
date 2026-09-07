"""実験14: 来週売上予測を1特徴だけ足したときの効果（10位）。

事前確認（exp_14_salesfc_check）:
  翌週実売との相関 予測 0.9399 > 直近7日実績 0.9178
  失速商品の検出: 実績比と予測比の相関 0.3416
機能はしているので、MAP@12 に効くかを1変数で測る。

比較基準（実験11、学習2週・候補100・BPR+SVD）: 0.03986

使い方: python exp_14_salesfc.py [週数]
"""
from __future__ import annotations

import gc
import sys
from datetime import timedelta

import polars as pl

from hm.config import VALID_CUTOFF, load_converted
from hm.metrics import get_actuals, evaluate_ranking
from hm.exp_log import log_experiment, timer
from hm.pipeline import dataset_path, feature_columns
from hm.embeddings import add_bpr_similarity, add_user2item_similarity
from hm.sales_forecast import add_sales_forecast
from hm.rerank import train_lgb_binary, predict

N_WEEKS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TOP_K = 100
VALID = VALID_CUTOFF
CUTS = [VALID - timedelta(days=7 * i) for i in range(1, N_WEEKS + 1)]

trans, customers, articles = load_converted()
n_items = int(articles['article_id'].max()) + 1
va = get_actuals(trans, VALID)

def load(cutoff, tag):
    df = pl.read_parquet(dataset_path(cutoff, TOP_K, tag))
    df = add_user2item_similarity(df, trans, cutoff, n_items)
    df = add_bpr_similarity(df, trans, cutoff, n_items, dim=64)
    return add_sales_forecast(df, trans, cutoff)

valid = load(VALID, 'valid')
cols = feature_columns(valid)
without = [c for c in cols if c not in ('a_pred_next_sales', 'a_pred_vs_recent')]
train = pl.concat([load(c, 'train_neg30').select(cols + ['label']) for c in CUTS],
                  how='vertical')
print('train {:,}行 / valid {:,}行'.format(len(train), len(valid)), flush=True)

print("\n{:<18}{:>8}{:>9}{:>10}{:>7}".format('variant', 'n_feat', 'R@12', 'MAP@12', 'sec'),
      flush=True)
for name, cs in [('without_salesfc', without), ('with_salesfc', cols)]:
    with timer() as t:
        model, _ = train_lgb_binary(train, cs, categorical=[])
        pred = predict(model, valid, cs)
        res = {r['k']: r for r in evaluate_ranking(pred, va, ks=[12]).to_dicts()}
    m = res[12]['map']
    print('{:<18}{:>8}{:>9.4f}{:>10.5f}{:>7.0f}'.format(
        name, len(cs), res[12]['recall'], m, t()), flush=True)
    log_experiment(name='rerank/{},weeks={}'.format(name, N_WEEKS), stage='rerank',
                   recall_at_k=res[12]['recall'], k=12, map_at_12=m, runtime_sec=t(),
                   n_candidates_per_user=TOP_K,
                   note='10位の来週売上予測 BPR+SVD併用 valid={}'.format(VALID))
    del model, pred
    gc.collect()
