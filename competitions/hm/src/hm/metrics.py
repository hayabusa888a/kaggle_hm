"""評価ハーネス。第1段は recall@K、第2段は MAP@12。

以降すべての実験でこのファイルの関数だけを使う。
実験ごとに指標の定義がブレると、比較した数字に意味がなくなるため。
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Mapping, Sequence

import polars as pl

from .config import LABEL_DAYS, MAP_K


# ---------------------------------------------------------------- 正解の作成

def get_actuals(transactions: pl.DataFrame, cutoff: date,
                label_days: int = LABEL_DAYS) -> dict[int, set[int]]:
    """cutoff の翌日から label_days 日間に買われた (customer -> {article}) を返す。"""
    label_end = cutoff + timedelta(days=label_days)
    grouped = (
        transactions
        .filter((pl.col('t_dat') > pl.lit(cutoff)) & (pl.col('t_dat') <= pl.lit(label_end)))
        .group_by('customer_id')
        .agg(pl.col('article_id').unique().alias('actual'))
    )
    return {row['customer_id']: set(row['actual']) for row in grouped.iter_rows(named=True)}


def actuals_to_frame(actuals: Mapping[int, Iterable[int]]) -> pl.DataFrame:
    """dict 形式の正解を (customer_id, article_id, label=1) の長形式にする。"""
    cids, aids = [], []
    for cid, arts in actuals.items():
        for aid in arts:
            cids.append(int(cid))
            aids.append(int(aid))
    return pl.DataFrame(
        {'customer_id': cids, 'article_id': aids},
        schema={'customer_id': pl.Int32, 'article_id': pl.Int32},
    ).with_columns(pl.lit(1, dtype=pl.Int8).alias('label'))


# ---------------------------------------------------- 1顧客ぶんのスカラー指標

def apk(actual_set: set[int], predicted: Sequence[int], k: int) -> float:
    """1顧客の average precision@k。重複予測は無視する（Kaggle公式と同じ扱い）。"""
    if not actual_set:
        return 0.0
    score = 0.0
    hits = 0
    seen: set[int] = set()
    for rank, article_id in enumerate(predicted, start=1):
        if rank > k:
            break
        if article_id in seen:
            continue
        seen.add(article_id)
        if article_id in actual_set:
            hits += 1
            score += hits / rank
    return score / min(len(actual_set), k)


average_precision_at_k = apk


# ------------------------------------------------------- データフレーム版の中核

def _ranked(df: pl.DataFrame, score_col: str, k_max: int) -> pl.DataFrame:
    """(customer_id, article_id, label) を score 降順に並べ 1始まりの rank を振る。

    同点は article_id 昇順で決定的に割る。乱数で順位が変わると実験が再現しなくなる。
    """
    out = (
        df.unique(subset=['customer_id', 'article_id'])
        .sort(['customer_id', score_col, 'article_id'], descending=[False, True, False])
        .with_columns(
            (pl.int_range(pl.len(), dtype=pl.Int32).over('customer_id') + 1).alias('_rank')
        )
    )
    return out.filter(pl.col('_rank') <= k_max)


def _metrics_frame(predictions: pl.DataFrame,
                   actuals: Mapping[int, set[int]],
                   ks: Sequence[int],
                   score_col: str,
                   denominator: str) -> pl.DataFrame:
    """recall@k と MAP@k をまとめて計算する。

    denominator='all_valid' … 正解を持つ全顧客を分母にする（候補が1件も出ていない顧客は0点）。
    denominator='covered'   … 候補が1件以上ある顧客だけを分母にする。
    候補生成を絞ると 'covered' は勝手に上がるので、意思決定には 'all_valid' を使うこと。
    """
    if denominator not in ('all_valid', 'covered'):
        raise ValueError(denominator)

    truth = actuals_to_frame(actuals)
    n_actual = (
        truth.group_by('customer_id').agg(pl.len().alias('n_actual'))
    )
    k_max = max(ks)

    preds = predictions.select(
        pl.col('customer_id').cast(pl.Int32),
        pl.col('article_id').cast(pl.Int32),
        pl.col(score_col).cast(pl.Float64),
    ).join(n_actual, on='customer_id', how='inner')  # 正解のない顧客は評価対象外

    ranked = (
        _ranked(preds, score_col, k_max)
        .join(truth, on=['customer_id', 'article_id'], how='left')
        .with_columns(pl.col('label').fill_null(0).cast(pl.Int32))
    )

    n_denom = len(n_actual) if denominator == 'all_valid' else ranked['customer_id'].n_unique()

    rows = []
    for k in ks:
        at_k = ranked.filter(pl.col('_rank') <= k).with_columns(
            pl.col('label').cum_sum().over('customer_id').alias('_cum_hits')
        )
        per_customer = at_k.group_by('customer_id').agg([
            pl.col('label').sum().alias('hits'),
            (pl.col('label') * pl.col('_cum_hits') / pl.col('_rank')).sum().alias('ap_num'),
            pl.col('n_actual').first(),
        ]).with_columns([
            (pl.col('hits') / pl.col('n_actual')).alias('recall'),
            (pl.col('ap_num') / pl.min_horizontal(pl.col('n_actual'), pl.lit(k))).alias('ap'),
        ])
        rows.append({
            'k': k,
            'recall': per_customer['recall'].sum() / n_denom,
            'map': per_customer['ap'].sum() / n_denom,
        })
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ 公開API

def recall_at_k(candidates: pl.DataFrame,
                actuals: Mapping[int, set[int]],
                k: int,
                score_col: str = 'candidate_score',
                denominator: str = 'all_valid') -> float:
    """候補生成の評価。指示書の第1段の指標。"""
    return _metrics_frame(candidates, actuals, [k], score_col, denominator)['recall'][0]


def map_at_12(predictions: pl.DataFrame,
              actuals: Mapping[int, set[int]],
              score_col: str = 'pred_score',
              denominator: str = 'all_valid') -> float:
    """最終評価。指示書の第2段の指標。"""
    return _metrics_frame(predictions, actuals, [MAP_K], score_col, denominator)['map'][0]


map_at_k = map_at_12


def evaluate_candidates(candidates: pl.DataFrame,
                        actuals: Mapping[int, set[int]],
                        ks: Sequence[int] = (12, 100, 200, 300),
                        score_col: str = 'candidate_score',
                        denominator: str = 'all_valid') -> pl.DataFrame:
    """候補集合を複数のKで一度に評価する。recall の上限を見るのに使う。"""
    return _metrics_frame(candidates, actuals, ks, score_col, denominator)


def evaluate_ranking(predictions: pl.DataFrame,
                     actuals: Mapping[int, set[int]],
                     ks: Sequence[int] = (12, 24, 48, 100),
                     score_col: str = 'pred_score',
                     denominator: str = 'all_valid') -> pl.DataFrame:
    """リランキング後の予測を評価する。k=12 の map が最終指標。"""
    return _metrics_frame(predictions, actuals, ks, score_col, denominator)


def hits_at_k(candidates: pl.DataFrame, actuals: Mapping[int, set[int]], k: int = 100,
              score_col: str = 'candidate_score') -> int:
    """上位k件に入った正解ペアの総数。

    1位が週ごとの HitNum@100 を公開しているので、そのまま突き合わせられる。
      2020-09-16週: 39142 / 2020-09-09週: 38427 / 2020-09-02週: 41019
    recall と違い「正解ペアを何本拾えたか」の絶対数なので、実装バグの検知に使いやすい。
    """
    truth = actuals_to_frame(actuals)
    preds = candidates.select(
        pl.col('customer_id').cast(pl.Int32), pl.col('article_id').cast(pl.Int32),
        pl.col(score_col).cast(pl.Float64))
    ranked = _ranked(preds, score_col, k)
    return ranked.join(truth, on=['customer_id', 'article_id'], how='inner').height
