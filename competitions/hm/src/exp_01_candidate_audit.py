"""実験01: 現状の候補生成を評価ハーネスで測り直す（valid週 cutoff=2020-09-15）。

やること
 1. 9位のデータ分析（valid週の商品の何%が直近30日にあるか / 顧客の何%に履歴があるか）を再現
 2. 各候補戦略の単体 recall（8位の表と同じ形）
 3. 全戦略を重み付き結合したときの recall@K（現行 best_weights と等重みを比較）
すべてホストCPU + polars lazy で走る。GPUもコンテナも要らない。
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

import polars as pl

from hm.config import CANDIDATE_DIR, CONVERTED_DIR, VALID_CUTOFF
from hm.metrics import get_actuals, evaluate_candidates
from hm.exp_log import log_experiment, timer

CUTOFF = VALID_CUTOFF
STRATEGIES = ['timedecay', 'repurchase', 'variant', 'also_bought',
              'item2item_cf', 'bin_popular', 'purchase_interval']
KS = [12, 50, 100, 200, 300, 500]

print(f'cutoff={CUTOFF}  正解週={CUTOFF + timedelta(days=1)} .. {CUTOFF + timedelta(days=7)}')

trans = pl.read_parquet(CONVERTED_DIR / 'transactions.parquet',
                        columns=['t_dat', 'customer_id', 'article_id'])
actuals = get_actuals(trans, CUTOFF)
valid_customers = pl.Series('customer_id', list(actuals.keys()), dtype=pl.Int32)
print(f'valid顧客数: {len(actuals):,}  正解ペア数: {sum(len(v) for v in actuals.values()):,}')

# ---------------------------------------------------------------- 9位のデータ分析
recent = trans.filter((pl.col('t_dat') > pl.lit(CUTOFF - timedelta(days=30)))
                      & (pl.col('t_dat') <= pl.lit(CUTOFF)))
valid_rows = trans.filter(pl.col('t_dat') > pl.lit(CUTOFF))
items_valid = set(valid_rows['article_id'].unique().to_list())
items_recent = set(recent['article_id'].unique().to_list())
users_recent = set(recent['customer_id'].unique().to_list())
print(f'[9位再現] valid週の商品が直近30日にも出現: '
      f'{len(items_valid & items_recent) / len(items_valid):.4f}  (原文 0.92)')
print(f'[9位再現] valid週の顧客が直近30日に取引あり: '
      f'{len(set(actuals) & users_recent) / len(actuals):.4f}  (原文 0.47)')
del recent, valid_rows, trans

# ------------------------------------------------------- 各戦略の単体 recall / MAP
def load(name: str) -> pl.DataFrame | None:
    path = CANDIDATE_DIR / f'{name}_{CUTOFF}.parquet'
    if not path.exists():
        return None
    return (pl.scan_parquet(path)
            .filter(pl.col('customer_id').is_in(valid_customers))
            .select(['customer_id', 'article_id', 'score', 'rank'])
            .collect())

singles: dict[str, pl.DataFrame] = {}
print('\n=== 各戦略の単体性能（valid週, 分母=valid顧客全員） ===')
print(f"{'strategy':<20}{'n/user':>9}{'cov%':>7}{'R@12':>9}{'R@100':>9}{'MAP@12':>9}{'sec':>7}")
for name in STRATEGIES:
    with timer() as t:
        df = load(name)
        if df is None:
            print(f'{name:<20}  (ファイルなし)')
            continue
        singles[name] = df
        res = evaluate_candidates(df, actuals, ks=KS, score_col='score')
    n_user = df['customer_id'].n_unique()
    per_user = len(df) / max(n_user, 1)
    cov = n_user / len(actuals) * 100
    r = {row['k']: row for row in res.to_dicts()}
    print(f'{name:<20}{per_user:>9.1f}{cov:>7.1f}{r[12]["recall"]:>9.4f}'
          f'{r[100]["recall"]:>9.4f}{r[12]["map"]:>9.5f}{t():>7.1f}')
    log_experiment(name=f'single/{name}', stage='candidate',
                   recall_at_k=r[100]['recall'], k=100, map_at_12=r[12]['map'],
                   runtime_sec=t(), n_candidates_per_user=per_user,
                   note=f'cutoff={CUTOFF} 単体 coverage={cov:.1f}%')


# ------------------------------------------------- 結合後の recall と候補プールの上限
import json

def combine(weights: dict[str, float]) -> pl.DataFrame:
    """create_candidates_scored と同じ min-max 正規化 + 重み付き加算。"""
    parts = []
    for name, w in weights.items():
        if w <= 0 or name not in singles:
            continue
        df = singles[name].select(['customer_id', 'article_id', 'score'])
        stats = df.group_by('customer_id').agg([
            pl.col('score').max().alias('mx'), pl.col('score').min().alias('mn')])
        parts.append(
            df.join(stats, on='customer_id', how='left')
            .with_columns(
                pl.when(pl.col('mx') == pl.col('mn')).then(1.0)
                .otherwise((pl.col('score') - pl.col('mn')) / (pl.col('mx') - pl.col('mn')))
                .alias('norm'))
            .select(['customer_id', 'article_id', (pl.col('norm') * w).alias('ws')])
        )
    return (pl.concat(parts)
            .group_by(['customer_id', 'article_id'])
            .agg(pl.col('ws').sum().alias('candidate_score')))

with open(CANDIDATE_DIR / 'best_weights_2020-09-15.json') as f:
    best_weights = json.load(f)['weights']

print('\n=== 結合後（分母=valid顧客全員） ===')
for label, weights in [
    ('best_weights(09-15でチューニング済)', best_weights),
    ('equal_weights', {k: 1.0 for k in STRATEGIES}),
]:
    with timer() as t:
        comb = combine(weights)
        res = evaluate_candidates(comb, actuals, ks=KS)
    per_user = len(comb) / comb['customer_id'].n_unique()
    print(f'\n[{label}]  候補/人={per_user:.1f}  全体={len(comb):,}行  ({t():.1f}s)')
    print(f"{'K':>6}{'recall':>10}{'MAP':>10}")
    for row in res.to_dicts():
        print(f"{row['k']:>6}{row['recall']:>10.4f}{row['map']:>10.5f}")
    r100 = [r for r in res.to_dicts() if r['k'] == 100][0]
    log_experiment(name=f'combined/{label}', stage='candidate',
                   recall_at_k=r100['recall'], k=100, map_at_12=r100['map'],
                   runtime_sec=t(), n_candidates_per_user=per_user,
                   note=f'cutoff={CUTOFF} 7戦略結合')

# 候補プールそのものの上限（順位を無視して、候補に正解が含まれているか）
pool = pl.concat([d.select(['customer_id', 'article_id']) for d in singles.values()]).unique()
hit = (pool.join(
    pl.DataFrame({'customer_id': [c for c, a in actuals.items() for _ in a],
                  'article_id': [x for a in actuals.values() for x in a]},
                 schema={'customer_id': pl.Int32, 'article_id': pl.Int32}),
    on=['customer_id', 'article_id'], how='inner').height)
total_pos = sum(len(v) for v in actuals.values())
print(f'\n=== 候補プールの上限（K無制限） ===')
print(f'  プール総数: {len(pool):,}  ({len(pool)/len(actuals):.1f}件/人)')
print(f'  正解の被覆率: {hit:,}/{total_pos:,} = {hit/total_pos:.4f}')
log_experiment(name='pool_ceiling/7strategies', stage='candidate',
               recall_at_k=hit/total_pos, k=-1, map_at_12=None,
               n_candidates_per_user=len(pool)/len(actuals),
               note='K無制限。順位を無視した候補プールの被覆率上限')
