"""実験15: 提出構成の事前確認。候補200 x BPR(128,300) が実際に効くか。

候補200の効果 +0.00029 は BPRなしで測った値。BPR を入れると
「顧客x商品の紐付け」が強化されるので、候補を増やす効果が食われる可能性がある。
提出作成に2時間かけるので、その前に学習2週で確認する。

比較（学習2週・BPR+SVD）:
  候補100 x BPR(64,100)  = 0.03986
  候補100 x BPR(128,300) = 0.04017
  候補200 x BPR(128,300) = ?
"""
from __future__ import annotations

import gc
from datetime import timedelta

import polars as pl

from hm.config import VALID_CUTOFF, load_converted
from hm.metrics import get_actuals, evaluate_ranking
from hm.exp_log import log_experiment, timer
from hm.pipeline import dataset_path, feature_columns
from hm.embeddings import add_bpr_similarity, add_user2item_similarity
from hm.rerank import train_lgb_binary, predict

VALID = VALID_CUTOFF
N_WEEKS = 2
CUTS = [VALID - timedelta(days=7 * i) for i in range(1, N_WEEKS + 1)]

trans, customers, articles = load_converted()
n_items = int(articles['article_id'].max()) + 1
va = get_actuals(trans, VALID)

print("{:>7}{:>7}{:>8}{:>12}{:>9}{:>10}{:>7}".format(
    'top_k', 'dim', 'iters', 'train行', 'R@12', 'MAP@12', 'sec'), flush=True)

for top_k, dim, iters in [(200, 128, 300), (200, 64, 100)]:
    with timer() as t:
        def load(cutoff, tag):
            df = pl.read_parquet(dataset_path(cutoff, top_k, tag))
            df = add_user2item_similarity(df, trans, cutoff, n_items)
            return add_bpr_similarity(df, trans, cutoff, n_items, dim=dim, iterations=iters)

        valid = load(VALID, 'valid')
        cols = feature_columns(valid)
        train = pl.concat([load(c, 'train_neg30').select(cols + ['label']) for c in CUTS],
                          how='vertical')
        model, _ = train_lgb_binary(train, cols, categorical=[])
        pred = predict(model, valid, cols)
        res = {r['k']: r for r in evaluate_ranking(pred, va, ks=[12]).to_dicts()}
    m = res[12]['map']
    print("{:>7}{:>7}{:>8}{:>12,}{:>9.4f}{:>10.5f}{:>7.0f}".format(
        top_k, dim, iters, len(train), res[12]['recall'], m, t()), flush=True)
    log_experiment(name='final_check/k={},bpr={}_{}'.format(top_k, dim, iters), stage='rerank',
                   recall_at_k=res[12]['recall'], k=12, map_at_12=m, runtime_sec=t(),
                   n_candidates_per_user=top_k,
                   note='提出構成の事前確認 学習{}週 valid={}'.format(N_WEEKS, VALID))
    del train, valid, model, pred
    gc.collect()
