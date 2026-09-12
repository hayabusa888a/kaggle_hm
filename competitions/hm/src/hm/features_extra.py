"""追加特徴量。

旧ノートブック実装にあって src/hm へ移植し損ねていたものをまとめて実装する。
9位は約300特徴量と報告しているのに対し、移植直後は98個しかなく、
色の嗜好・購買周期・同一product_code・未購買カテゴリ系がまとめて欠けていた。

リーク規律は features.py と同じ。どの関数も cutoff 以前しか見ない。
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl


def _hist(transactions: pl.DataFrame, cutoff: date, days: int | None = None) -> pl.DataFrame:
    df = transactions.filter(pl.col('t_dat') <= pl.lit(cutoff))
    if days is not None:
        df = df.filter(pl.col('t_dat') > pl.lit(cutoff - timedelta(days=days)))
    return df


def _days_since(col: str, cutoff: date) -> pl.Expr:
    return (pl.lit(cutoff) - pl.col(col)).dt.total_days().cast(pl.Int32)


# ------------------------------------------------------------------ 色の嗜好

def add_colour_features(candidates: pl.DataFrame, transactions: pl.DataFrame,
                        articles: pl.DataFrame, cutoff: date) -> pl.DataFrame:
    """色の嗜好。H&M はファッションなので色は購買判断の主要因のはず。

    u_favorite_colour      最も買っている色
    u_colour_diversity     買った色の種類数
    u_colour_loyalty       好きな色が購買に占める割合（一貫性）
    u_fav_colour_cat_div   好きな色を何カテゴリで買うか
    a_colour_group_code    候補商品の色
    ua_is_favorite_colour  候補商品が好きな色か
    """
    df = _hist(transactions, cutoff)
    art = articles.select(['article_id', 'colour_group_code', 'product_type_no'])
    j = df.join(art, on='article_id', how='inner').drop_nulls('colour_group_code')

    totals = j.group_by('customer_id').agg(pl.len().alias('_t'))
    fav = (j.group_by(['customer_id', 'colour_group_code']).agg(pl.len().alias('_n'))
           .sort(['customer_id', '_n', 'colour_group_code'], descending=[False, True, False])
           .group_by('customer_id').first()
           .join(totals, on='customer_id', how='left')
           .select(['customer_id',
                    pl.col('colour_group_code').alias('u_favorite_colour'),
                    (pl.col('_n') / pl.col('_t')).alias('u_colour_loyalty')]))
    div = j.group_by('customer_id').agg(
        pl.col('colour_group_code').n_unique().alias('u_colour_diversity'))
    # 好きな色だけを買うのか、その色を幅広いカテゴリで使うのか
    fav_cat = (j.join(fav.select(['customer_id', 'u_favorite_colour']),
                      on='customer_id', how='inner')
               .filter(pl.col('colour_group_code') == pl.col('u_favorite_colour'))
               .group_by('customer_id')
               .agg(pl.col('product_type_no').n_unique().alias('u_fav_colour_cat_div')))

    return (candidates
            .join(art.select(['article_id', 'colour_group_code']), on='article_id', how='left')
            .join(fav, on='customer_id', how='left')
            .join(div, on='customer_id', how='left')
            .join(fav_cat, on='customer_id', how='left')
            .with_columns([
                (pl.col('colour_group_code') == pl.col('u_favorite_colour'))
                .cast(pl.Int8).fill_null(0).alias('ua_is_favorite_colour'),
                pl.col('u_colour_loyalty').fill_null(0.0),
                pl.col('u_colour_diversity').fill_null(0),
                pl.col('u_fav_colour_cat_div').fill_null(0),
                pl.col('u_favorite_colour').fill_null(-1),
            ])
            .rename({'colour_group_code': 'a_colour_group_code'})
            .with_columns(pl.col('a_colour_group_code').fill_null(-1)))


# ------------------------------------------------------- 同一 product_code 系

def add_same_product_code_features(candidates: pl.DataFrame, transactions: pl.DataFrame,
                                   articles: pl.DataFrame, cutoff: date) -> pl.DataFrame:
    """同一 product_code（色・サイズ違い）の購買履歴。

    article_id は色/サイズごとに別IDなので、「同じ服の別の色を買っていた」ことは
    article_id 単位の特徴では拾えない。8位が product code statistics として
    挙げていた領域。
    """
    df = _hist(transactions, cutoff)
    pc = articles.select(['article_id', 'product_code'])
    j = df.join(pc, on='article_id', how='inner')

    agg = (j.group_by(['customer_id', 'product_code']).agg([
        pl.len().alias('ua_same_pc_count'),
        pl.col('t_dat').max().alias('_last'),
        pl.col('t_dat').min().alias('_first'),
    ]).with_columns([
        _days_since('_last', cutoff).alias('ua_days_since_same_pc'),
        pl.when(pl.col('ua_same_pc_count') > 1)
        .then((pl.col('_last') - pl.col('_first')).dt.total_days().cast(pl.Float64)
              / (pl.col('ua_same_pc_count') - 1))
        .otherwise(-1.0).alias('ua_same_pc_avg_interval'),
    ]).drop(['_last', '_first']))

    return (candidates.join(pc, on='article_id', how='left')
            .join(agg, on=['customer_id', 'product_code'], how='left')
            .with_columns([
                pl.col('ua_same_pc_count').fill_null(0),
                pl.col('ua_days_since_same_pc').fill_null(-1),
                pl.col('ua_same_pc_avg_interval').fill_null(-1.0),
            ])
            .with_columns((pl.col('ua_same_pc_count') > 0).cast(pl.Int8).alias('ua_has_same_pc'))
            .drop('product_code'))


# ------------------------------------------------------------------ 購買周期

def add_cycle_features(candidates: pl.DataFrame) -> pl.DataFrame:
    """購買周期。9位が「重要度上位10はすべて再購入系」と報告している領域。

    「そろそろ買う頃か」を、ユーザー全体と（ユーザー×product_code）の両方で表す。
    build_user_features と add_same_product_code_features を通したあとに呼ぶこと。
    """
    return candidates.with_columns([
        (pl.col('u_avg_purchase_interval') - pl.col('u_days_since_last_purchase'))
        .alias('u_expected_next_purchase_days'),
        ((pl.col('u_days_since_last_purchase') > pl.col('u_avg_purchase_interval'))
         & (pl.col('u_avg_purchase_interval') > 0)).cast(pl.Int8).alias('u_is_overdue'),
        (pl.col('ua_same_pc_avg_interval') - pl.col('ua_days_since_same_pc'))
        .alias('ua_expected_next_days'),
        ((pl.col('ua_days_since_same_pc') > pl.col('ua_same_pc_avg_interval'))
         & (pl.col('ua_same_pc_avg_interval') > 0)).cast(pl.Int8).alias('ua_is_overdue'),
    ])


# ------------------------------------------------------------- 未購買カテゴリ

def add_unseen_category_features(candidates: pl.DataFrame, transactions: pl.DataFrame,
                                 articles: pl.DataFrame, cutoff: date,
                                 recent_days: int = 14) -> pl.DataFrame:
    """まだ買っていないカテゴリの中での人気度・順位。

    「この顧客が手を出していない領域で、いま売れているもの」を表す。
    既に買っているカテゴリは uc_* が担っているので、その裏側を埋める。
    """
    art = articles.select(['article_id', 'product_type_no'])
    pop = (_hist(transactions, cutoff, recent_days)
           .join(art, on='article_id', how='inner')
           .group_by(['product_type_no', 'article_id'])
           .agg(pl.len().cast(pl.Float32).alias('_sales')))
    pop = (pop.sort(['product_type_no', '_sales', 'article_id'],
                    descending=[False, True, False])
           .with_columns((pl.int_range(pl.len()).over('product_type_no') + 1)
                         .cast(pl.Int16).alias('_rank')))

    seen = (_hist(transactions, cutoff).join(art, on='article_id', how='inner')
            .select(['customer_id', 'product_type_no']).unique()
            .with_columns(pl.lit(1, dtype=pl.Int8).alias('_seen')))

    has_pt = 'product_type_no' in candidates.columns
    out = candidates if has_pt else candidates.join(art, on='article_id', how='left')

    out = (out
           .join(pop.select(['article_id',
                             pl.col('_sales').alias('_pop'),
                             pl.col('_rank').alias('_rk')]),
                 on='article_id', how='left')
           .join(seen, on=['customer_id', 'product_type_no'], how='left')
           .with_columns([pl.col('_seen').fill_null(0),
                          pl.col('_pop').fill_null(0.0),
                          pl.col('_rk').fill_null(999)])
           .with_columns([
               (1 - pl.col('_seen')).cast(pl.Int8).alias('ua_is_unseen_category'),
               # 既に買っているカテゴリでは 0 にして「未購買領域の指標」に絞る
               pl.when(pl.col('_seen') == 0).then(pl.col('_pop'))
               .otherwise(0.0).alias('ua_unseen_cat_popularity'),
               pl.when(pl.col('_seen') == 0).then(pl.col('_rk'))
               .otherwise(999).alias('ua_unseen_cat_rank'),
           ])
           .drop(['_seen', '_pop', '_rk']))
    return out if has_pt else out.drop('product_type_no')


# ------------------------------------------------------------------ その他派生

def add_misc_features(candidates: pl.DataFrame, transactions: pl.DataFrame,
                      cutoff: date) -> pl.DataFrame:
    """残りの派生特徴。user/article 特徴を join したあとに呼ぶこと。

    u_total_spend         累計購買金額（回数とは別の軸）
    u_price_exploration   価格帯をどれだけ探索するか
    ua_channel_match      顧客の主チャネルと商品の主チャネルの一致度
    ua_price_percentile   顧客の価格分布の中でこの商品がどの位置か
    a_is_new / a_is_recent 新商品フラグ
    """
    spend = (_hist(transactions, cutoff).group_by('customer_id')
             .agg(pl.col('price').sum().alias('u_total_spend')))
    return (candidates.join(spend, on='customer_id', how='left')
            .with_columns(pl.col('u_total_spend').fill_null(0.0))
            .with_columns([
                (pl.col('u_price_std') / pl.col('u_price_mean').clip(lower_bound=1e-9))
                .alias('u_price_exploration'),
                # チャネルは 1/2 の平均なので、差が小さいほど一致している
                (1.0 - (pl.col('u_channel_mean') - pl.col('a_channel_mean')).abs())
                .alias('ua_channel_match'),
                ((pl.col('a_price_mean') - pl.col('u_price_mean'))
                 / pl.col('u_price_std').clip(lower_bound=1e-6))
                .alias('ua_price_percentile'),
                (pl.col('a_days_since_first_sale') <= 30).cast(pl.Int8).alias('a_is_new'),
                (pl.col('a_days_since_first_sale') <= 90).cast(pl.Int8).alias('a_is_recent'),
            ]))


def add_all_extra(candidates: pl.DataFrame, transactions: pl.DataFrame,
                  articles: pl.DataFrame, cutoff: date) -> pl.DataFrame:
    """追加特徴量をまとめて付ける。

    build_all_features（ユーザー/商品/交互作用）を通したあとに呼ぶ。
    cycle と misc は先行する特徴の列に依存するので順序を変えないこと。
    """
    out = add_colour_features(candidates, transactions, articles, cutoff)
    out = add_same_product_code_features(out, transactions, articles, cutoff)
    out = add_unseen_category_features(out, transactions, articles, cutoff)
    out = add_cycle_features(out)
    out = add_misc_features(out, transactions, cutoff)
    return out.with_columns([pl.col(c).cast(pl.Float32)
                             for c, t in zip(out.columns, out.dtypes) if t == pl.Float64])
