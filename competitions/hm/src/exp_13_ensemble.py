"""実験13: アンサンブル。既存の埋め込みキャッシュを使い回すので追加のBPR学習が不要。

根拠（第3部原文）
 - 9位「my final submission is based on 1 model trained on all data samples and the
   other 3 models trained based on k fold split（4 models in total）」
   one model: public 0.0338 -> ensemble: public 0.0343（+0.0005）
 - 6位「Our best single model is catboost with cv 0.0403 / lb 0.0341.
   We use several models to ensemble(cv: 0.0412, lb: 0.0348)」（+0.0007）
 - 3位「The final version is the average score of 16 models which were trained with
   different hyperparameters (learning rate, max_depth and even random seed)」
 - 11位は optuna で重みを最適化し cv 0.03589 -> 0.03619。ただし本人が
   「I think it is overestimated」と書いているので、まず単純平均で測る。
 - 10位「Final model was an ensemble of four LightGBM models. But the improvement was
   rather small, ~1% compared to the single best model」

3位に倣い learning_rate / num_leaves / seed を変えた複数モデルの平均を取る。
順位の平均（rank averaging）も併せて見る。スコアのスケールが揃わない場合に強い。

使い方: python exp_13_ensemble.py [週数]
"""
from __future__ import annotations

import gc
import sys
from datetime import timedelta

import numpy as np
import polars as pl

from hm.config import VALID_CUTOFF, load_converted
from hm.metrics import get_actuals, evaluate_ranking
from hm.exp_log import log_experiment, timer
from hm.pipeline import dataset_path, feature_columns
from hm.embeddings import add_bpr_similarity, add_user2item_similarity
from hm.rerank import train_lgb_binary, predict

N_WEEKS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TOP_K = 100
BPR_DIM = 64
VALID = VALID_CUTOFF
CUTS = [VALID - timedelta(days=7 * i) for i in range(1, N_WEEKS + 1)]

# 3位「different hyperparameters (learning rate, max_depth and even random seed)」
VARIANTS = [
    {'learning_rate': 0.05, 'num_leaves': 63,  'seed': 42},
    {'learning_rate': 0.03, 'num_leaves': 127, 'seed': 7},
    {'learning_rate': 0.08, 'num_leaves': 31,  'seed': 2024},
    {'learning_rate': 0.05, 'num_leaves': 95,  'seed': 555, 'feature_fraction': 0.6},
]

trans, customers, articles = load_converted()
n_items = int(articles['article_id'].max()) + 1
va = get_actuals(trans, VALID)

def load(cutoff, tag):
    df = pl.read_parquet(dataset_path(cutoff, TOP_K, tag))
    df = add_user2item_similarity(df, trans, cutoff, n_items)
    return add_bpr_similarity(df, trans, cutoff, n_items, dim=BPR_DIM)

valid = load(VALID, 'valid')
cols = feature_columns(valid)
train = pl.concat([load(c, 'train_neg30').select(cols + ['label']) for c in CUTS],
                  how='vertical')
print('train {:,}行 / valid {:,}行'.format(len(train), len(valid)), flush=True)

print("\n{:<28}{:>9}{:>10}{:>7}".format('model', 'R@12', 'MAP@12', 'sec'), flush=True)
scores = []
ranks = []
for i, params in enumerate(VARIANTS):
    n_rounds = int(300 * 0.05 / params['learning_rate'])   # lr に応じて本数を調整
    with timer() as t:
        model, _ = train_lgb_binary(train, cols, categorical=[], params=params,
                                    num_boost_round=n_rounds, seed=params['seed'])
        pred = predict(model, valid, cols)
        res = {r['k']: r for r in evaluate_ranking(pred, va, ks=[12]).to_dicts()}
    s = pred['pred_score'].to_numpy()
    scores.append(s)
    ranks.append(pred.with_columns(pl.Series('s', s))
                 .with_columns(pl.col('s').rank(descending=True).over('customer_id')
                               .alias('r'))['r'].to_numpy())
    label = 'lr={lr} leaves={lv} seed={sd}'.format(
        lr=params['learning_rate'], lv=params['num_leaves'], sd=params['seed'])
    print('{:<28}{:>9.4f}{:>10.5f}{:>7.0f}'.format(
        label, res[12]['recall'], res[12]['map'], t()), flush=True)
    log_experiment(name='ensemble/single_{}'.format(i), stage='rerank',
                   recall_at_k=res[12]['recall'], k=12, map_at_12=res[12]['map'],
                   runtime_sec=t(), n_candidates_per_user=TOP_K,
                   note='LGB binary {} 学習{}週'.format(label, N_WEEKS))
    del model, pred
    gc.collect()

print('', flush=True)
for name, arr, desc in [
    ('score平均', np.mean(scores, axis=0), '予測スコアの単純平均'),
    ('rank平均', -np.mean(ranks, axis=0), '顧客内順位の平均（スケール差に強い）'),
]:
    blended = valid.with_columns(pl.Series('pred_score', arr))
    res = {r['k']: r for r in evaluate_ranking(blended, va, ks=[12]).to_dicts()}
    print('{:<28}{:>9.4f}{:>10.5f}   {}'.format(
        name, res[12]['recall'], res[12]['map'], desc), flush=True)
    log_experiment(name='ensemble/{}'.format(name), stage='rerank',
                   recall_at_k=res[12]['recall'], k=12, map_at_12=res[12]['map'],
                   n_candidates_per_user=TOP_K,
                   note='{}モデル {} 学習{}週'.format(len(VARIANTS), desc, N_WEEKS))
