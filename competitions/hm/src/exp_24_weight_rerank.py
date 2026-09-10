"""実験24: 候補結合の重みが第2段にどう転移するかを実測する。

実験23で recall@200 を4週平均で最適化したが、候補段階の指標が第2段の性能を
どこまで予測するかは未検証。特に重要なのは、第2段の98特徴量のうち
**重み変更で変わるのは candidate_score と rank の2つだけ**という点。
rank_<戦略> 10個は各戦略の内部順位なので重みに依存しない。

  -> 第2段は各戦略の順位を直接見ているので、結合スコアが崩れても並べ直せるはず。
  -> だとすると重みの役割は「どの候補が上位200件に残るか」に限られ、
     候補段階の MAP@12（= candidate_score 順の精度）は第2段の性能を予測しない。

この仮説を、性格の違う4構成を同じ条件で回して確かめる。

使い方: python exp_24_weight_rerank.py
"""
from __future__ import annotations

import gc
import json
from datetime import timedelta
from pathlib import Path

import polars as pl

from hm.config import VALID_CUTOFF, EXPERIMENT_DIR, load_converted
from hm.metrics import get_actuals, evaluate_candidates, evaluate_ranking
from hm.exp_log import log_experiment, timer
from hm.pipeline import build_sources, feature_columns
from hm.combine import combine
from hm.embeddings import add_bpr_similarity, add_user2item_similarity
from hm.rerank import downsample_negatives, train_lgb_binary, predict
import hm.features as F

TOP_K, TOP_N = 200, 100
BPR_DIM, BPR_ITERS, N_WEEKS = 128, 300, 6
VALID = VALID_CUTOFF
CUTS = [VALID - timedelta(days=7 * i) for i in range(1, N_WEEKS + 1)]

cfg = json.loads((EXPERIMENT_DIR / 'night_best_params.json').read_text(encoding='utf-8'))
LGB_PARAMS, ROUNDS = cfg['params'], cfg['rounds']

trans, customers, articles = load_converted()
n_items = int(articles['article_id'].max()) + 1
va = get_actuals(trans, VALID)

# --- 比較する重み構成
opt_path = EXPERIMENT_DIR / 'best_weights_opt.json'
NAMES = ['repurchase', 'timedecay', 'same_product_code', 'purchase_interval',
         'user_cf', 'also_bought', 'item2item_cf', 'popular',
         'popular_by_age', 'popular_by_channel']
configs = {'equal': {n: 1.0 for n in NAMES},
           'legacy': {**{n: 1.0 for n in NAMES},
                      'timedecay': 3.5, 'repurchase': 1.5, 'popular_by_age': 1.5}}
if opt_path.exists():
    configs['optuna_r200'] = json.loads(opt_path.read_text(encoding='utf-8'))['weights']
# 実験23のログから「両立型」(trial 5) を拾えたら足す
bal = EXPERIMENT_DIR / 'weights_balanced.json'
if bal.exists():
    configs['balanced'] = json.loads(bal.read_text(encoding='utf-8'))['weights']

print('比較する構成: {}'.format(list(configs)), flush=True)
print('基準（equal の実測）: 再rank MAP@12 = 0.04153', flush=True)


def build(cutoff, weights, actuals, downsample):
    """指定の重みで候補を結合し、特徴量を付けてデータセットを作る。"""
    ids = pl.Series('customer_id', list(actuals.keys()), dtype=pl.Int32)
    srcs = build_sources(trans, customers, articles, cutoff, TOP_N,
                         customer_filter=ids, build_missing=False)
    ds = combine(srcs, weights=weights, top_k=TOP_K)
    del srcs
    gc.collect()
    from hm.rerank import attach_labels
    ds = attach_labels(ds, actuals)
    if downsample:
        ds = downsample_negatives(ds, 30.0)
    ds = F.build_all_features(ds, trans, articles, customers, cutoff,
                              n_chunks=16 if not downsample else 8)
    ds = add_user2item_similarity(ds, trans, cutoff, n_items)
    return add_bpr_similarity(ds, trans, cutoff, n_items, dim=BPR_DIM, iterations=BPR_ITERS)


print('\n{:<14}{:>10}{:>11}{:>12}{:>11}{:>8}'.format(
    'config', 'cand R@200', 'cand MAP12', 'rerank R@12', 'rerank MAP', 'sec'), flush=True)
for label, w in configs.items():
    with timer() as t:
        valid = build(VALID, w, va, downsample=False)
        cols = feature_columns(valid)
        cand = {r['k']: r for r in
                evaluate_candidates(valid, va, ks=[12, 200]).to_dicts()}
        parts = []
        for c in CUTS:
            a = get_actuals(trans, c)
            parts.append(build(c, w, a, downsample=True).select(cols + ['label']))
        train = pl.concat(parts, how='vertical')
        del parts
        gc.collect()
        model, _ = train_lgb_binary(train, cols, categorical=[], params=LGB_PARAMS,
                                    num_boost_round=ROUNDS)
        pred = predict(model, valid, cols)
        res = {r['k']: r for r in evaluate_ranking(pred, va, ks=[12]).to_dicts()}
    print('{:<14}{:>10.4f}{:>11.5f}{:>12.4f}{:>11.5f}{:>8.0f}'.format(
        label, cand[200]['recall'], cand[12]['map'],
        res[12]['recall'], res[12]['map'], t()), flush=True)
    log_experiment(name='weight_rerank/{}'.format(label), stage='rerank',
                   recall_at_k=res[12]['recall'], k=12, map_at_12=res[12]['map'],
                   runtime_sec=t(), n_candidates_per_user=TOP_K,
                   note='候補R@200={:.4f} 候補MAP12={:.5f} 重み={}'.format(
                       cand[200]['recall'], cand[12]['map'],
                       {k: round(v, 2) for k, v in w.items()}))
    del valid, train, model, pred
    gc.collect()
