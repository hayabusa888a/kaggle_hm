"""実験05: リランキング。recall戦略メタ特徴の効果を1変数で切り分ける。

3位の原文は2つのことを同時に言っている。
  (1) 「whether this article is recalled by strategy_name」= 二値フラグ
  (2) 「the rank of this article under the strategy_name」= 戦略内の順位
そしてコメント欄で、質問者が挙げた2種類の rank のうち
  「Both the ranking features you described was used in my final model. But If only
    talking about the boosting from 0.02855 to 0.0326, it's the latter one（ranking num）
    contributing most.」
と答えている。既存実装は (1) しか入っていないので、(2) の寄与をここで実測する。

variant:
  none    メタ特徴なし（候補スコアと通常特徴のみ）
  flag    in_<strategy> のみ = 既存実装相当
  rank    in_<strategy> + rank_<strategy> = 3位の構成
それぞれ categorical_features の明示指定あり/なしも見る（6位: +0.0005〜0.0008）。
"""
from __future__ import annotations

import gc
import sys
from datetime import date

import polars as pl

from hm.config import VALID_CUTOFF, load_converted
from hm.metrics import get_actuals, evaluate_ranking
from hm.exp_log import log_experiment, timer
from hm.features import CATEGORICAL_FEATURES
from hm.pipeline import build_dataset, feature_columns, ALL_STRATEGIES
from hm.rerank import downsample_negatives, train_lgb_binary, predict

TRAIN_CUTOFFS = [date(2020, 9, 8)]
VALID = VALID_CUTOFF
TOP_N = 100
TOP_K = int(sys.argv[1]) if len(sys.argv) > 1 else 100
NEG_RATIO = 30.0        # 3位「negative samples with amount of 30*len(pos_samples)」

trans, customers, articles = load_converted()

print(f'学習fold: {TRAIN_CUTOFFS}  valid: {VALID}  top_k={TOP_K}')

train_parts = []
for cutoff in TRAIN_CUTOFFS:
    with timer() as t:
        acts = get_actuals(trans, cutoff)
        ids = pl.Series('customer_id', list(acts.keys()), dtype=pl.Int32)
        ds = build_dataset(trans, customers, articles, cutoff, actuals=acts,
                           top_n=TOP_N, top_k=TOP_K, customer_filter=ids,
                           sample_fn=lambda d: downsample_negatives(d, NEG_RATIO),
                           n_chunks=4, cache_tag=f'train_neg{NEG_RATIO:.0f}')
    print(f'  fold {cutoff}: {len(ds):,}行 正例{ds["label"].sum():,} ({t():.0f}s)')
    train_parts.append(ds)
train = pl.concat(train_parts, how='vertical')
del train_parts; gc.collect()

valid_actuals = get_actuals(trans, VALID)
valid_ids = pl.Series('customer_id', list(valid_actuals.keys()), dtype=pl.Int32)
with timer() as t:
    valid = build_dataset(trans, customers, articles, VALID, actuals=valid_actuals,
                          top_n=TOP_N, top_k=TOP_K, customer_filter=valid_ids, n_chunks=8, cache_tag='valid')
print(f'  valid: {len(valid):,}行 正例{valid["label"].sum():,} ({t():.0f}s)')

all_cols = feature_columns(train)
META_FLAG = [f'in_{s}' for s in ALL_STRATEGIES]
META_RANK = [f'rank_{s}' for s in ALL_STRATEGIES]

VARIANTS = {
    'none':  [c for c in all_cols if c not in META_FLAG + META_RANK],
    'flag':  [c for c in all_cols if c not in META_RANK],
    'rank':  all_cols,
}

print(f'\n特徴量数: none={len(VARIANTS["none"])} flag={len(VARIANTS["flag"])} rank={len(VARIANTS["rank"])}')
print(f"\n{'variant':<10}{'categorical':>13}{'n_feat':>8}{'R@12':>9}{'MAP@12':>10}{'sec':>8}")
for vname, cols in VARIANTS.items():
    for use_cat in (False, True):
        with timer() as t:
            model, cats = train_lgb_binary(
                train, cols, categorical=CATEGORICAL_FEATURES if use_cat else [])
            pred = predict(model, valid, cols, cats)
            res = {r['k']: r for r in evaluate_ranking(pred, valid_actuals, ks=[12, 100]).to_dicts()}
        print(f'{vname:<10}{str(use_cat):>13}{len(cols):>8}'
              f'{res[12]["recall"]:>9.4f}{res[12]["map"]:>10.5f}{t():>8.0f}')
        log_experiment(name=f'rerank/meta={vname},cat={use_cat}', stage='rerank',
                       recall_at_k=res[12]['recall'], k=12, map_at_12=res[12]['map'],
                       runtime_sec=t(), n_candidates_per_user=TOP_K,
                       note=f'LGB binary neg{NEG_RATIO:.0f}x train={TRAIN_CUTOFFS} '
                            f'valid={VALID} n_feat={len(cols)}')
        del model, pred; gc.collect()
