"""実験18: 学習週を 6->8->10 週に伸ばす（最良パラメータで）。

実験08（旧パラメータ・候補100・BPRなし）では 1->6週で +0.00080、
増分は減速しつつも単調増加だった:
  1週 0.03656 / 2週 0.03673 / 4週 0.03713 / 6週 0.03736
上位陣は 1位6週 / 6位9週 / 3位18週 / 9位20週 を使っており、まだ余地がある。

今回は最良構成（候補200 / BPR(128,300) / lr=0.0188 leaves=188 796本）で
6・8・10週を測る。11位が「週数を変えたときだけ CV と LB が負相関」と
報告しているので、CVが伸びても提出して実測すること。

使い方: python exp_18_train_weeks_long.py [週数...]
"""
from __future__ import annotations

import gc
import sys
from datetime import timedelta

import polars as pl

from hm.config import VALID_CUTOFF, load_converted
from hm.metrics import get_actuals, evaluate_ranking
from hm.exp_log import log_experiment, timer
from hm.pipeline import build_dataset, dataset_path, feature_columns
from hm.embeddings import add_bpr_similarity, add_user2item_similarity
from hm.rerank import downsample_negatives, train_lgb_binary, predict

TOP_K, BPR_DIM, BPR_ITERS = 200, 128, 300
LGB = {'learning_rate': 0.0188, 'num_leaves': 188}
ROUNDS = 796
VALID = VALID_CUTOFF
WEEKS = [int(x) for x in (sys.argv[1:] or ['6', '8', '10'])]

trans, customers, articles = load_converted()
n_items = int(articles['article_id'].max()) + 1
va = get_actuals(trans, VALID)


def load(cutoff, tag, build=False):
    """データセットを読む。無ければ（build=True なら）作る。"""
    path = dataset_path(cutoff, TOP_K, tag)
    if not path.exists():
        if not build:
            return None
        a = get_actuals(trans, cutoff)
        build_dataset(trans, customers, articles, cutoff, actuals=a, top_k=TOP_K,
                      customer_filter=pl.Series('customer_id', list(a.keys()), dtype=pl.Int32),
                      sample_fn=lambda d: downsample_negatives(d, 30.0),
                      n_chunks=8, cache_tag=tag)
    df = pl.read_parquet(path)
    df = add_user2item_similarity(df, trans, cutoff, n_items)
    return add_bpr_similarity(df, trans, cutoff, n_items, dim=BPR_DIM, iterations=BPR_ITERS)


valid = load(VALID, 'valid')
cols = feature_columns(valid)
print('valid {:,}行'.format(len(valid)), flush=True)

print("\n{:>6}{:>14}{:>10}{:>10}{:>7}".format('weeks', 'train行', 'R@12', 'MAP@12', 'sec'),
      flush=True)
for n in WEEKS:
    cuts = [VALID - timedelta(days=7 * i) for i in range(1, n + 1)]
    with timer() as t:
        parts = []
        missing = False
        for c in cuts:
            d = load(c, 'train_neg30', build=True)
            if d is None:
                print('  週{} : {} のデータセットが無く作れない。この条件を飛ばす'.format(n, c),
                      flush=True)
                missing = True
                break
            parts.append(d.select(cols + ['label']))
        if missing:
            del parts; gc.collect()
            continue
        train = pl.concat(parts, how='vertical')
        del parts; gc.collect()
        model, _ = train_lgb_binary(train, cols, categorical=[], params=LGB,
                                    num_boost_round=ROUNDS)
        pred = predict(model, valid, cols)
        res = {r['k']: r for r in evaluate_ranking(pred, va, ks=[12]).to_dicts()}
    m = res[12]['map']
    print('{:>6}{:>14,}{:>10.4f}{:>10.5f}{:>7.0f}'.format(
        n, len(train), res[12]['recall'], m, t()), flush=True)
    log_experiment(name='train_weeks_long/{}週'.format(n), stage='rerank',
                   recall_at_k=res[12]['recall'], k=12, map_at_12=m, runtime_sec=t(),
                   n_candidates_per_user=TOP_K,
                   note='最良構成 BPR({},{}) lr={} leaves={} rounds={}'.format(
                       BPR_DIM, BPR_ITERS, LGB['learning_rate'], LGB['num_leaves'], ROUNDS))
    del train, model, pred; gc.collect()
