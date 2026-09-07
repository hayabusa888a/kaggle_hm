"""来週の売上予測モデル。10位の施策。

原文:
  "In addition to what many people used (rebuys, I2I, CF, Top-Sellers etc.), I created a
   simple sales prediction model. Based on recent sales of an article it predicted next
   weeks sales. The main use of this model was helping to detect products that are
   phasing in or out (since we don't have that data).
   Example: an article that is only available for a short time (special collaboration
   etc.) might have daily sales like: 0 0 28 431 389 105 32 11. Using recent popularity
   will massively overestimate next weeks sales while the prediction model learned the
   article is phasing out (out of stock).
   It was in the end one of the most used features, probably since it covers popularity
   and (future) availability in one."

リーク規律
  cutoff C 用の予測器は、**ラベルが C 時点で観測済みの週ペアだけ**で学習する。
  つまり学習に使う週 w は w + 7 <= C を満たすものに限る。
  特徴量は t_dat <= w、ラベルは (w, w+7] の売上。
  こうすると C より後の情報は一切入らない。10位が
  「I tagged the models with the last date used to create it. That way it was
    impossible to leak any information」と書いているのと同じ扱い。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from .config import FEATURE_DIR

# 特徴量に使う窓（日）
WINDOWS = (7, 14, 21, 28, 56)


def _article_state(transactions: pl.DataFrame, w: date) -> pl.DataFrame:
    """週 w 時点での商品ごとの状態量。t_dat <= w しか見ない。"""
    hist = transactions.filter(pl.col('t_dat') <= pl.lit(w))
    base = (hist.group_by('article_id').agg([
        pl.len().alias('f_total'),
        pl.col('t_dat').min().alias('_first'),
        pl.col('t_dat').max().alias('_last'),
        pl.col('price').mean().alias('f_price_mean'),
    ]).with_columns([
        (pl.lit(w) - pl.col('_first')).dt.total_days().cast(pl.Int32).alias('f_age_days'),
        (pl.lit(w) - pl.col('_last')).dt.total_days().cast(pl.Int32).alias('f_days_since_last'),
    ]).drop(['_first', '_last']))

    for d in WINDOWS:
        win = hist.filter(pl.col('t_dat') > pl.lit(w - timedelta(days=d)))
        base = base.join(
            win.group_by('article_id').agg([
                pl.len().alias(f'f_sales_{d}d'),
                pl.col('customer_id').n_unique().alias(f'f_buyers_{d}d'),
            ]), on='article_id', how='left'
        ).with_columns([pl.col(f'f_sales_{d}d').fill_null(0),
                        pl.col(f'f_buyers_{d}d').fill_null(0)])

    # 「立ち上がり / 失速」を表す比。10位が検出したかったのはこの形。
    return base.with_columns([
        (pl.col('f_sales_7d') / pl.col('f_sales_14d').clip(lower_bound=1)).alias('f_r_7_14'),
        (pl.col('f_sales_14d') / pl.col('f_sales_28d').clip(lower_bound=1)).alias('f_r_14_28'),
        (pl.col('f_sales_28d') / pl.col('f_sales_56d').clip(lower_bound=1)).alias('f_r_28_56'),
        (pl.col('f_sales_7d') / pl.col('f_total').clip(lower_bound=1)).alias('f_r_7_total'),
    ])


def _next_week_sales(transactions: pl.DataFrame, w: date) -> pl.DataFrame:
    """(w, w+7] の売上。学習ラベル。"""
    return (transactions
            .filter((pl.col('t_dat') > pl.lit(w))
                    & (pl.col('t_dat') <= pl.lit(w + timedelta(days=7))))
            .group_by('article_id').agg(pl.len().alias('y')))


def predict_next_week_sales(transactions: pl.DataFrame, cutoff: date,
                            n_train_weeks: int = 10, min_sales: int = 1,
                            cache: bool = True) -> pl.DataFrame:
    """cutoff 時点の各商品について、翌週 (cutoff, cutoff+7] の売上予測を返す。

    返り値: (article_id, a_pred_next_sales, a_pred_vs_recent)
      a_pred_vs_recent = 予測 / 直近7日実績。1未満なら失速、1超なら立ち上がり。
    """
    path = FEATURE_DIR / f'salesfc_{cutoff}_w{n_train_weeks}.parquet'
    if cache and path.exists():
        return pl.read_parquet(path)

    import lightgbm as lgb

    # ラベルが cutoff 時点で観測済みの週だけを学習に使う
    train_weeks = [cutoff - timedelta(days=7 * i) for i in range(1, n_train_weeks + 1)]

    parts = []
    for w in train_weeks:
        x = _article_state(transactions, w).filter(pl.col('f_total') >= min_sales)
        y = _next_week_sales(transactions, w)
        parts.append(x.join(y, on='article_id', how='left')
                     .with_columns(pl.col('y').fill_null(0)))
    train = pl.concat(parts, how='vertical')
    del parts

    feat_cols = [c for c in train.columns if c.startswith('f_')]
    xt = train.select([pl.col(c).cast(pl.Float32) for c in feat_cols]).to_numpy()
    # 売上はロングテールなので log1p 空間で回帰する
    yt = np.log1p(train['y'].to_numpy().astype(np.float64))

    model = lgb.train(
        {'objective': 'regression', 'metric': 'l2', 'learning_rate': 0.05,
         'num_leaves': 63, 'min_data_in_leaf': 50, 'feature_fraction': 0.8,
         'bagging_fraction': 0.8, 'bagging_freq': 1, 'verbosity': -1, 'seed': 42},
        lgb.Dataset(xt, label=yt, feature_name=feat_cols), num_boost_round=200)

    now = _article_state(transactions, cutoff).filter(pl.col('f_total') >= min_sales)
    xn = now.select([pl.col(c).cast(pl.Float32) for c in feat_cols]).to_numpy()
    pred = np.expm1(model.predict(xn)).clip(min=0)

    out = now.select(['article_id', 'f_sales_7d']).with_columns(
        pl.Series('a_pred_next_sales', pred.astype(np.float32))
    ).with_columns(
        (pl.col('a_pred_next_sales') / pl.col('f_sales_7d').cast(pl.Float32).clip(lower_bound=1))
        .alias('a_pred_vs_recent')
    ).drop('f_sales_7d')

    out.write_parquet(path)
    return out


def add_sales_forecast(candidates: pl.DataFrame, transactions: pl.DataFrame,
                       cutoff: date, cache: bool = True) -> pl.DataFrame:
    """候補に来週売上予測を足す。商品単位の特徴なので article_id で join するだけ。"""
    fc = predict_next_week_sales(transactions, cutoff, cache=cache)
    return (candidates.join(fc, on='article_id', how='left')
            .with_columns([pl.col('a_pred_next_sales').fill_null(0),
                           pl.col('a_pred_vs_recent').fill_null(0)]))
