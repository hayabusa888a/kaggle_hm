"""実験26: 未移植だった特徴量をまとめて足したときの効果。

旧ノートブック実装にあって src/hm に移植し損ねていた約23特徴を実装した
（色の嗜好 / 同一product_code / 購買周期 / 未購買カテゴリ / その他派生）。
加えて combine が item2item_cf・also_bought・user_cf の**類似度スコアの生値**を
渡すようにした（これまで順位しか渡していなかった）。

ベースは public 最良の構成。重み最適化は CV では +0.00024 だったが
public は -0.00087 と悪化したため、**等重み**に戻している。

  候補200 / 10戦略・等重み / BPR(128,300)+SVD / 学習6週
  LightGBM lr=0.0188 leaves=188 796本  -> 再rank MAP@12 = 0.04160（実測）

比較:
  base  = 既存98特徴（＋スコア3本を除く）
  extra = 追加後（124特徴前後）
同じ候補・同じハイパラで特徴量だけを変える1変数比較。
"""
from __future__ import annotations

import gc
from datetime import timedelta

import polars as pl

from hm.config import VALID_CUTOFF, load_converted
from hm.metrics import get_actuals, evaluate_ranking
from hm.exp_log import log_experiment, timer
from hm.pipeline import build_sources, feature_columns
from hm.combine import combine
from hm.embeddings import add_bpr_similarity, add_user2item_similarity
from hm.features_extra import add_all_extra
from hm.rerank import attach_labels, downsample_negatives, train_lgb_binary, predict
import hm.features as F

TOP_K, TOP_N = 200, 100
BPR_DIM, BPR_ITERS, N_WEEKS = 128, 300, 6
LGB = {'learning_rate': 0.0188, 'num_leaves': 188}
ROUNDS = 796
VALID = VALID_CUTOFF
CUTS = [VALID - timedelta(days=7 * i) for i in range(1, N_WEEKS + 1)]

trans, customers, articles = load_converted()
n_items = int(articles['article_id'].max()) + 1
va = get_actuals(trans, VALID)
print('基準（既存98特徴・同構成）: 再rank MAP@12 = 0.04160', flush=True)


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
    ds = add_all_extra(ds, trans, articles, cutoff)
    ds = add_user2item_similarity(ds, trans, cutoff, n_items)
    return add_bpr_similarity(ds, trans, cutoff, n_items, dim=BPR_DIM, iterations=BPR_ITERS)


with timer() as t:
    valid = build(VALID, va, downsample=False)
all_cols = feature_columns(valid)
# 追加したぶんを特定して、それを抜いた集合を base とする
EXTRA_PREFIX = ('u_favorite_colour', 'u_colour_', 'u_fav_colour', 'ua_is_favorite_colour',
                'a_colour_group_code', 'ua_same_pc', 'ua_has_same_pc', 'ua_days_since_same_pc',
                'ua_is_unseen', 'ua_unseen_cat', 'u_expected_next', 'u_is_overdue',
                'ua_expected_next', 'ua_is_overdue', 'u_total_spend', 'u_price_exploration',
                'ua_channel_match', 'ua_price_percentile', 'a_is_new', 'a_is_recent',
                'score_')
extra_cols = [c for c in all_cols if c.startswith(EXTRA_PREFIX)]
base_cols = [c for c in all_cols if c not in extra_cols]
print('valid {:,}行 ({:.0f}s)  base={} extra=+{} 合計={}'.format(
    len(valid), t(), len(base_cols), len(extra_cols), len(all_cols)), flush=True)

parts = []
for c in CUTS:
    a = get_actuals(trans, c)
    parts.append(build(c, a, downsample=True).select(all_cols + ['label']))
train = pl.concat(parts, how='vertical')
del parts
gc.collect()
print('train {:,}行 正例{:,}'.format(len(train), train['label'].sum()), flush=True)

print('\n{:<10}{:>9}{:>10}{:>11}{:>8}'.format('variant', 'n_feat', 'R@12', 'MAP@12', 'sec'),
      flush=True)
for label, cols in [('base', base_cols), ('extra', all_cols)]:
    with timer() as t:
        model, _ = train_lgb_binary(train, cols, categorical=[], params=LGB,
                                    num_boost_round=ROUNDS)
        pred = predict(model, valid, cols)
        res = {r['k']: r for r in evaluate_ranking(pred, va, ks=[12]).to_dicts()}
    m = res[12]['map']
    print('{:<10}{:>9}{:>10.4f}{:>11.5f}{:>8.0f}'.format(
        label, len(cols), res[12]['recall'], m, t()), flush=True)
    log_experiment(name='extra_features/{}'.format(label), stage='rerank',
                   recall_at_k=res[12]['recall'], k=12, map_at_12=m, runtime_sec=t(),
                   n_candidates_per_user=TOP_K,
                   note='特徴量{}個 等重み BPR(128,300) 学習6週 lr=0.0188'.format(len(cols)))
    if label == 'extra':
        imp = sorted(zip(cols, model.feature_importance('gain')), key=lambda x: -x[1])
        print('\n重要度 上位20（gain）:', flush=True)
        for i, (n, g) in enumerate(imp[:20], 1):
            mark = ' <- 新規' if n in extra_cols else ''
            print('  {:>2}. {:<34}{:>12,.0f}{}'.format(i, n, g, mark), flush=True)
        print('\n新規特徴の重要度順位:', flush=True)
        rank = {n: i for i, (n, _) in enumerate(imp, 1)}
        for n in sorted(extra_cols, key=lambda x: rank[x]):
            print('  {:>4}位  {:<34}{:>12,.0f}'.format(
                rank[n], n, dict(imp)[n]), flush=True)
    del model, pred
    gc.collect()
