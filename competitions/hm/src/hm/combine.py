"""複数の候補戦略を1つの候補集合にまとめ、同時に第2段用のメタ特徴を作る。

3位の原文（+0.0041、銀圏->入賞圏）:
  "Features about recall strategy: whether this article is recalled by `strategy_name`
   and the rank of this article under the `strategy_name`"
  "These features, plus an expansion of candidates num for each user from dozens to
   hundred, boost my LB score from 0.02855 to 0.03262"
  "if I only increase the recall num and don't adding the recall features, the CV score
   is very very poor"
コメント欄で本人が、2種類の rank（戦略固有の関連度スコア / 単純な順位）のうち
**順位（ranking num）の寄与が最も大きかった**と明言している。
したがって in_<strategy> の二値フラグだけでなく rank_<strategy> を必ず残す。
"""
from __future__ import annotations

import polars as pl

# その戦略で拾われなかったことを表す番兵。欠損のままだと木が「無い」を学習できない。
MISSING_RANK = 999

# 順位だけでなくスコアの生値も第2段に渡す戦略
SCORE_PASSTHROUGH = ('item2item_cf', 'also_bought', 'user_cf')


def _normalize(df: pl.DataFrame) -> pl.DataFrame:
    """顧客ごとに score を min-max 正規化する。戦略ごとにスケールが違うため。"""
    stats = df.group_by('customer_id').agg([
        pl.col('score').max().alias('_mx'), pl.col('score').min().alias('_mn')])
    return (df.join(stats, on='customer_id', how='left')
            .with_columns(
                pl.when(pl.col('_mx') == pl.col('_mn')).then(1.0)
                .otherwise((pl.col('score') - pl.col('_mn')) / (pl.col('_mx') - pl.col('_mn')))
                .alias('norm_score'))
            .drop(['_mx', '_mn']))


def combine(sources: dict[str, pl.DataFrame],
            weights: dict[str, float] | None = None,
            top_k: int | None = None,
            with_meta: bool = True) -> pl.DataFrame:
    """戦略ごとの候補を重複除去しつつ結合する（8位「重複は除去する」）。

    sources: 戦略名 -> (customer_id, article_id, score, rank)
    返り値 : customer_id, article_id, candidate_score, rank
             with_meta=True なら in_<s> と rank_<s> を戦略ぶん追加

    top_k を指定すると candidate_score 上位 top_k で切る。
    """
    weights = weights or {name: 1.0 for name in sources}

    scored = []
    for name, df in sources.items():
        w = weights.get(name, 0.0)
        if w <= 0:
            continue
        scored.append(_normalize(df.select(['customer_id', 'article_id', 'score']))
                      .select(['customer_id', 'article_id',
                               (pl.col('norm_score') * w).alias('ws')]))
    combined = (pl.concat(scored)
                .group_by(['customer_id', 'article_id'])
                .agg(pl.col('ws').sum().alias('candidate_score')))

    if with_meta:
        for name, df in sources.items():
            meta = (df.select(['customer_id', 'article_id',
                               pl.col('rank').cast(pl.Int16).alias(f'rank_{name}')])
                    .unique(subset=['customer_id', 'article_id']))
            combined = (combined.join(meta, on=['customer_id', 'article_id'], how='left')
                        .with_columns([
                            pl.col(f'rank_{name}').is_not_null().cast(pl.Int8).alias(f'in_{name}'),
                            pl.col(f'rank_{name}').fill_null(MISSING_RANK),
                        ]))
        # 類似度そのものを渡す戦略。順位だけだと「どれくらい似ているか」が落ちる。
        # i2i は類似度、also_bought は共起の強さで、値の大小に意味がある。
        for name in SCORE_PASSTHROUGH:
            if name not in sources:
                continue
            s = (sources[name].select(['customer_id', 'article_id',
                                       pl.col('score').cast(pl.Float32).alias(f'score_{name}')])
                 .unique(subset=['customer_id', 'article_id']))
            combined = (combined.join(s, on=['customer_id', 'article_id'], how='left')
                        .with_columns(pl.col(f'score_{name}').fill_null(0.0)))

    combined = combined.sort(
        ['customer_id', 'candidate_score', 'article_id'], descending=[False, True, False]
    ).with_columns((pl.int_range(pl.len(), dtype=pl.Int32).over('customer_id') + 1).alias('rank'))

    if top_k is not None:
        combined = combined.filter(pl.col('rank') <= top_k)
    return combined


def meta_feature_names(sources) -> list[str]:
    """第2段に渡すメタ特徴の列名。"""
    names = list(sources)
    return [f'in_{n}' for n in names] + [f'rank_{n}' for n in names]
