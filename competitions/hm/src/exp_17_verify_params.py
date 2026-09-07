"""実験17: Optuna の最良 lr/leaves/rounds を既定パラメータと組んで再現できるか確認。

実験16を12試行で打ち切ったため study が失われ、trial 10 の
min_data_in_leaf / feature_fraction / bagging_fraction / lambda_* が復元できない。
（各試行の全パラメータをログに出していなかった設計ミス。exp_16 は修正済み）

ログから分かるのは lr=0.0283 / leaves=155 / rounds=529 のみ。
これを既定パラメータ（min_data_in_leaf=100, ff=0.8, bf=0.8, 正則化なし）と
組んだときに trial 10 の 0.04141 をどこまで再現できるかを測る。

比較:
  trial 0（現行構成の同条件）= 0.04104
  trial 10（Optuna最良）      = 0.04141
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

N_WEEKS, TOP_K = 2, 200
BPR_DIM, BPR_ITERS = 128, 300
VALID = VALID_CUTOFF
CUTS = [VALID - timedelta(days=7 * i) for i in range(1, N_WEEKS + 1)]

trans, customers, articles = load_converted()
n_items = int(articles['article_id'].max()) + 1
va = get_actuals(trans, VALID)

def load(cutoff, tag):
    df = pl.read_parquet(dataset_path(cutoff, TOP_K, tag))
    df = add_user2item_similarity(df, trans, cutoff, n_items)
    return add_bpr_similarity(df, trans, cutoff, n_items, dim=BPR_DIM, iterations=BPR_ITERS)

valid = load(VALID, 'valid')
cols = feature_columns(valid)
train = pl.concat([load(c, 'train_neg30').select(cols + ['label']) for c in CUTS],
                  how='vertical')
del articles, customers; gc.collect()
print('train {:,}行 / valid {:,}行'.format(len(train), len(valid)), flush=True)

CANDIDATES = [
    ('現行(基準)', {'learning_rate': 0.05, 'num_leaves': 63}, 300),
    ('optuna_t10', {'learning_rate': 0.0283, 'num_leaves': 155}, 529),
    ('optuna_t3',  {'learning_rate': 0.0188, 'num_leaves': 188}, 796),
    # leaves をもう少し増やす方向（trial 10 と 5 の間）も1点見る
    ('t10_leaves250', {'learning_rate': 0.0283, 'num_leaves': 250}, 529),
]

print("\n{:<16}{:>9}{:>8}{:>8}{:>10}{:>7}".format(
    'name', 'lr', 'leaves', 'rounds', 'MAP@12', 'sec'), flush=True)
for name, params, rounds in CANDIDATES:
    with timer() as t:
        model, _ = train_lgb_binary(train, cols, categorical=[], params=params,
                                    num_boost_round=rounds, seed=42)
        pred = predict(model, valid, cols)
        m = evaluate_ranking(pred, va, ks=[12])['map'][0]
    print('{:<16}{:>9.4f}{:>8}{:>8}{:>10.5f}{:>7.0f}'.format(
        name, params['learning_rate'], params['num_leaves'], rounds, m, t()), flush=True)
    log_experiment(name='verify/{}'.format(name), stage='rerank', map_at_12=m, k=12,
                   runtime_sec=t(), n_candidates_per_user=TOP_K,
                   note='既定パラメータ+lr/leaves/rounds 学習{}週 BPR({},{})'.format(
                       N_WEEKS, BPR_DIM, BPR_ITERS))
    del model, pred; gc.collect()
