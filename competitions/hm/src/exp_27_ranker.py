"""実験27: LightGBM lambdarank vs binary。

6位が公開しているモデル比較表では、同じ LightGBM なら lambdarank が上:
  lightgbm binary     220候補 -> cv 0.0396
  lightgbm lambdarank 220候補 -> cv 0.0400  (+0.0004)
  lightgbm lambdarank 1000候補 -> cv 0.0381 (候補が多いと崩れる)
本文の「binary優勢」は CatBoost binary が最良という意味で、
LightGBM 内では lambdarank が勝っている。こちらは binary しか試していない。

チーム間の対立は CatBoost YetiRank に集中していて
（11位「LGBMRankerより明確に良い」/ 6位qyxs「CVがかなり低い」）、
lambdarank には否定的な報告がない。まずここから試す。

構成は public 最良のもの（候補200 / 10戦略・等重み / BPR(128,300) / 学習6週）。
特徴量は実験26で効果ゼロだった追加分を除いた100個に戻す。
基準: binary で MAP@12 = 0.04171

負例ダウンサンプリングについて
  binary は正例の30倍に間引いている。ranker も同じデータで比較するのが
  1変数比較として正しいので、まず同条件で測る。
  （間引きが ranker に不利に働く可能性はあるが、それは別の変数）

使い方: python exp_27_ranker.py [truncation_level ...]
"""
from __future__ import annotations

import gc
import sys
from datetime import timedelta

import polars as pl

from hm.config import VALID_CUTOFF, load_converted
from hm.metrics import get_actuals, evaluate_ranking
from hm.exp_log import log_experiment, timer
from hm.pipeline import build_sources, feature_columns
from hm.combine import combine
from hm.embeddings import add_bpr_similarity, add_user2item_similarity
from hm.rerank import (attach_labels, downsample_negatives, train_lgb_binary,
                       train_lgb_ranker, predict)
import hm.features as F

TOP_K, TOP_N = 200, 100
BPR_DIM, BPR_ITERS, N_WEEKS = 128, 300, 6
LR, LEAVES, ROUNDS = 0.0188, 188, 796
VALID = VALID_CUTOFF
CUTS = [VALID - timedelta(days=7 * i) for i in range(1, N_WEEKS + 1)]
TRUNCS = [int(x) for x in (sys.argv[1:] or ['12', '30'])]

trans, customers, articles = load_converted()
n_items = int(articles['article_id'].max()) + 1
va = get_actuals(trans, VALID)
print('基準（binary・同構成）: MAP@12 = 0.04171', flush=True)


def build(cutoff, actuals, downsample):
    ids = pl.Series('customer_id', list(actuals.keys()), dtype=pl.Int32)
    srcs = build_sources(trans, customers, articles, cutoff, TOP_N,
                         customer_filter=ids, build_missing=False)
    ds = combine(srcs, top_k=TOP_K)
    del srcs
    gc.collect()
    ds = attach_labels(ds, actuals)
    if downsample:
        ds = downsample_negatives(ds, 30.0)
    ds = F.build_all_features(ds, trans, articles, customers, cutoff,
                              n_chunks=16 if not downsample else 8)
    ds = add_user2item_similarity(ds, trans, cutoff, n_items)
    return add_bpr_similarity(ds, trans, cutoff, n_items, dim=BPR_DIM, iterations=BPR_ITERS)


valid = build(VALID, va, downsample=False)
# 実験26で効果ゼロだった追加特徴は使わない（score_* も combine 側で付くので除く）
cols = [c for c in feature_columns(valid) if not c.startswith('score_')]
print('valid {:,}行 特徴量{}'.format(len(valid), len(cols)), flush=True)

parts = []
for c in CUTS:
    a = get_actuals(trans, c)
    parts.append(build(c, a, downsample=True).select(cols + ['label', 'customer_id']))
train = pl.concat(parts, how='vertical')
del parts
gc.collect()
print('train {:,}行 正例{:,}'.format(len(train), train['label'].sum()), flush=True)

runs = [('binary', None)] + [('lambdarank_t{}'.format(t), t) for t in TRUNCS]
print('\n{:<18}{:>10}{:>11}{:>8}'.format('model', 'R@12', 'MAP@12', 'sec'), flush=True)
for label, trunc in runs:
    with timer() as t:
        if trunc is None:
            model, _ = train_lgb_binary(
                train, cols, categorical=[],
                params={'learning_rate': LR, 'num_leaves': LEAVES},
                num_boost_round=ROUNDS)
        else:
            model, _ = train_lgb_ranker(
                train, cols,
                params={'learning_rate': LR, 'num_leaves': LEAVES,
                        'lambdarank_truncation_level': trunc},
                num_boost_round=ROUNDS)
        pred = predict(model, valid, cols)
        res = {r['k']: r for r in evaluate_ranking(pred, va, ks=[12]).to_dicts()}
    m = res[12]['map']
    print('{:<18}{:>10.4f}{:>11.5f}{:>8.0f}'.format(label, res[12]['recall'], m, t()),
          flush=True)
    log_experiment(name='ranker/{}'.format(label), stage='rerank',
                   recall_at_k=res[12]['recall'], k=12, map_at_12=m, runtime_sec=t(),
                   n_candidates_per_user=TOP_K,
                   note='候補200 等重み BPR(128,300) 学習6週 lr={} leaves={} {}本'.format(
                       LR, LEAVES, ROUNDS))
    del model, pred
    gc.collect()
