"""実験25: u2tag を含む13戦略を、最適化した重みで第2段まで検証する。

実験21では u2tag を**等重み**で足して再rank MAP@12 が -0.00025 と悪化した。
実験24で「候補MAP@12 は第2段を予測しない／recall@200 は予測する」と分かったので、
u2tag の不採用判断は等重みという条件付きのものだったことになる。

重みで押し出しを制御できれば、u2tag が持つプール上限 +0.0358 を活かせるはず。
実験23（with_u2tag=true）が出した重みで、10戦略の最良と比べる。

比較:
  10戦略 + 最適化重み : 再rank MAP@12 = 0.04181（実験24の実測）
  13戦略 + 最適化重み : ?
"""
from __future__ import annotations

import gc
import json
from datetime import timedelta

import polars as pl

from hm.config import VALID_CUTOFF, EXPERIMENT_DIR, load_converted
from hm.metrics import get_actuals, evaluate_candidates, evaluate_ranking
from hm.exp_log import log_experiment, timer
from hm.pipeline import build_sources, feature_columns, U2TAG_TAGS
from hm.combine import combine
from hm.embeddings import add_bpr_similarity, add_user2item_similarity
from hm.rerank import attach_labels, downsample_negatives, train_lgb_binary, predict
import hm.features as F
import hm.candidates as C

TOP_K, TOP_N = 200, 100
BPR_DIM, BPR_ITERS, N_WEEKS = 128, 300, 6
VALID = VALID_CUTOFF
CUTS = [VALID - timedelta(days=7 * i) for i in range(1, N_WEEKS + 1)]

cfg = json.loads((EXPERIMENT_DIR / 'night_best_params.json').read_text(encoding='utf-8'))
LGB_PARAMS, ROUNDS = cfg['params'], cfg['rounds']
wpath = EXPERIMENT_DIR / 'best_weights_u2tag.json'
if not wpath.exists():
    raise SystemExit('13戦略の重みがない: {}'.format(wpath))
W = json.loads(wpath.read_text(encoding='utf-8'))['weights']
print('13戦略の重み: {}'.format({k: round(v, 3) for k, v in W.items()}), flush=True)
print('比較基準（10戦略+最適化重み）: 再rank MAP@12 = 0.04181', flush=True)

trans, customers, articles = load_converted()
n_items = int(articles['article_id'].max()) + 1
va = get_actuals(trans, VALID)


def build(cutoff, actuals, downsample):
    ids = pl.Series('customer_id', list(actuals.keys()), dtype=pl.Int32)
    srcs = build_sources(trans, customers, articles, cutoff, TOP_N,
                         customer_filter=ids, build_missing=False)
    for t in U2TAG_TAGS:
        nm = f'u2tag_{t}'
        df = C.load_cached(nm, cutoff, TOP_N, customer_filter=ids)
        if df is not None:
            srcs[nm] = df
    ds = combine(srcs, weights=W, top_k=TOP_K)
    del srcs
    gc.collect()
    ds = attach_labels(ds, actuals)
    if downsample:
        ds = downsample_negatives(ds, 30.0)
    ds = F.build_all_features(ds, trans, articles, customers, cutoff,
                              n_chunks=16 if not downsample else 8)
    ds = add_user2item_similarity(ds, trans, cutoff, n_items)
    return add_bpr_similarity(ds, trans, cutoff, n_items, dim=BPR_DIM, iterations=BPR_ITERS)


with timer() as t:
    valid = build(VALID, va, downsample=False)
    cols = feature_columns(valid)
    cand = {r['k']: r for r in evaluate_candidates(valid, va, ks=[12, 200]).to_dicts()}
    print('候補: R@200={:.4f} MAP@12={:.5f} 特徴量{}'.format(
        cand[200]['recall'], cand[12]['map'], len(cols)), flush=True)
    parts = []
    for c in CUTS:
        a = get_actuals(trans, c)
        parts.append(build(c, a, downsample=True).select(cols + ['label']))
    train = pl.concat(parts, how='vertical')
    del parts
    gc.collect()
    print('train {:,}行 正例{:,}'.format(len(train), train['label'].sum()), flush=True)
    model, _ = train_lgb_binary(train, cols, categorical=[], params=LGB_PARAMS,
                                num_boost_round=ROUNDS)
    pred = predict(model, valid, cols)
    res = {r['k']: r for r in evaluate_ranking(pred, va, ks=[12]).to_dicts()}

print('\n13戦略+最適化重み: 再rank MAP@12 = {:.5f}  (10戦略最良 0.04181 から {:+.5f})'.format(
    res[12]['map'], res[12]['map'] - 0.04181), flush=True)
log_experiment(name='u2tag_weighted/13strategies', stage='rerank',
               recall_at_k=res[12]['recall'], k=12, map_at_12=res[12]['map'],
               runtime_sec=t(), n_candidates_per_user=TOP_K,
               note='13戦略 最適化重み 候補R@200={:.4f}'.format(cand[200]['recall']))
