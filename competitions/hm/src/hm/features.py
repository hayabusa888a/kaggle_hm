"""第2段（リランキング）の特徴量。すべて polars / CPU。

リーク規律: どの関数も cutoff より後の取引を一切見ない。
キャッシュはマウント配下の outputs/features に置く（元実装の /home/rapids は
コンテナを作り直すと消えるため、キャッシュとして信用できなかった）。

出典を明記した特徴:
 - 6位「items' discount: use max/mean to represent the common price of the item.
        use the price for each row to calculate the discount of the item」
 - 6位「For item's first purchased time, it may represent the "release time" of this item.
        So we can use "user" as key to calculate the statistics of "release time"」
 - 9位「item feature: average age of customers who have bought this item in the last 7/30/90 days」
 - 9位「重要度上位10はすべて再購入系（何日前にこの商品を買ったか / 何回買ったか）」
 - 11位「mean and std of price, sales_channel_id」
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from .config import FEATURE_DIR

WINDOWS = (7, 14, 30, 60)


def _hist(transactions: pl.DataFrame, cutoff: date, days: int | None = None) -> pl.DataFrame:
    df = transactions.filter(pl.col('t_dat') <= pl.lit(cutoff))
    if days is not None:
        df = df.filter(pl.col('t_dat') > pl.lit(cutoff - timedelta(days=days)))
    return df


def _days_since(col: str, cutoff: date) -> pl.Expr:
    return (pl.lit(cutoff) - pl.col(col)).dt.total_days().cast(pl.Int32)


# ------------------------------------------------------------------ ユーザー特徴

def build_user_features(transactions: pl.DataFrame, customers: pl.DataFrame,
                        cutoff: date, cache: bool = True) -> pl.DataFrame:
    path = FEATURE_DIR / f'user_{cutoff}.parquet'
    if cache and path.exists():
        return pl.read_parquet(path)

    df = _hist(transactions, cutoff)
    f = (df.group_by('customer_id').agg([
        pl.len().alias('u_total_purchases'),
        pl.col('article_id').n_unique().alias('u_unique_articles'),
        pl.col('t_dat').max().alias('_last'),
        pl.col('t_dat').min().alias('_first'),
        pl.col('price').mean().alias('u_price_mean'),
        pl.col('price').std().alias('u_price_std'),
        pl.col('price').max().alias('u_price_max'),
        pl.col('sales_channel_id').mean().alias('u_channel_mean'),
        pl.col('t_dat').n_unique().alias('_n_baskets'),
    ]).with_columns([
        _days_since('_last', cutoff).alias('u_days_since_last_purchase'),
        _days_since('_first', cutoff).alias('u_days_since_first_purchase'),
    ]).drop(['_last', '_first']))

    for days in WINDOWS:
        f = f.join(
            _hist(transactions, cutoff, days).group_by('customer_id')
            .agg(pl.len().alias(f'u_purchases_last_{days}d')),
            on='customer_id', how='left'
        ).with_columns(pl.col(f'u_purchases_last_{days}d').fill_null(0))

    f = f.join(
        _hist(transactions, cutoff, 30).group_by('customer_id').agg([
            pl.col('article_id').n_unique().alias('u_unique_articles_last_30d'),
            pl.col('price').mean().alias('u_price_mean_30d')]),
        on='customer_id', how='left'
    ).with_columns([
        pl.col('u_unique_articles_last_30d').fill_null(0),
        (pl.col('u_total_purchases')
         / (pl.col('u_days_since_first_purchase') / 7.0).clip(lower_bound=1))
        .alias('u_purchase_freq_per_week'),
        (pl.col('u_total_purchases') / pl.col('_n_baskets')).alias('u_avg_basket_size'),
    ]).drop('_n_baskets')

    # 再購入率と、購入間隔の平均
    ua = df.group_by(['customer_id', 'article_id']).agg(pl.len().alias('n'))
    f = f.join(ua.group_by('customer_id').agg((pl.col('n') > 1).mean().alias('u_repeat_rate')),
               on='customer_id', how='left')
    baskets = df.select(['customer_id', 't_dat']).unique().sort(['customer_id', 't_dat'])
    f = f.join(
        baskets.with_columns(
            (pl.col('t_dat') - pl.col('t_dat').shift(1).over('customer_id'))
            .dt.total_days().alias('gap'))
        .drop_nulls('gap').group_by('customer_id')
        .agg(pl.col('gap').mean().alias('u_avg_purchase_interval')),
        on='customer_id', how='left'
    ).with_columns(pl.col('u_avg_purchase_interval').fill_null(-1))

    # 直近30日に買ったもののうち、それ以前に買ったことがない商品の割合（新しい物好きか）
    old = (_hist(transactions, cutoff).filter(
        pl.col('t_dat') <= pl.lit(cutoff - timedelta(days=30)))
        .select(['customer_id', 'article_id']).unique()
        .with_columns(pl.lit(1, dtype=pl.Int8).alias('_old')))
    f = f.join(
        _hist(transactions, cutoff, 30).select(['customer_id', 'article_id']).unique()
        .join(old, on=['customer_id', 'article_id'], how='left')
        .group_by('customer_id').agg(pl.col('_old').is_null().mean().alias('u_new_article_rate_30d')),
        on='customer_id', how='left'
    ).with_columns(pl.col('u_new_article_rate_30d').fill_null(0))

    f = f.join(customers.select(['customer_id', 'age', 'age_bin', 'FN', 'Active',
                                 'club_member_status_enc', 'fashion_news_frequency_enc']),
               on='customer_id', how='left')
    f.write_parquet(path)
    return f


def add_user_category_features(user_features: pl.DataFrame, transactions: pl.DataFrame,
                               articles: pl.DataFrame, cutoff: date) -> pl.DataFrame:
    """カテゴリ多様性と、6位の「発売日の代理変数を顧客ごとに集計」。"""
    df = _hist(transactions, cutoff)
    release = (df.group_by('article_id').agg(pl.col('t_dat').min().alias('_release')))
    j = df.join(articles.select(['article_id', 'product_type_no']), on='article_id', how='left') \
          .join(release, on='article_id', how='left')
    agg = j.group_by('customer_id').agg([
        pl.col('product_type_no').n_unique().alias('u_category_diversity'),
        # 買った商品の「発売からの経過日数」の平均/最小 = 新しい物好き / 古い物好き（6位）
        (pl.col('t_dat') - pl.col('_release')).dt.total_days().mean().alias('u_age_of_bought_mean'),
        (pl.col('t_dat') - pl.col('_release')).dt.total_days().min().alias('u_age_of_bought_min'),
    ])
    return user_features.join(agg, on='customer_id', how='left')


# -------------------------------------------------------------------- 商品特徴

def build_article_features(transactions: pl.DataFrame, articles: pl.DataFrame,
                           customers: pl.DataFrame, cutoff: date,
                           cache: bool = True) -> pl.DataFrame:
    path = FEATURE_DIR / f'article_{cutoff}.parquet'
    if cache and path.exists():
        return pl.read_parquet(path)

    df = _hist(transactions, cutoff)
    f = (df.group_by('article_id').agg([
        pl.len().alias('a_total_sales'),
        pl.col('customer_id').n_unique().alias('a_unique_buyers'),
        pl.col('t_dat').min().alias('_first_sale'),
        pl.col('t_dat').max().alias('_last_sale'),
        pl.col('price').mean().alias('a_price_mean'),
        pl.col('price').std().alias('a_price_std'),
        pl.col('price').max().alias('_price_max'),
        pl.col('sales_channel_id').mean().alias('a_channel_mean'),
    ]).with_columns([
        _days_since('_first_sale', cutoff).alias('a_days_since_first_sale'),
        _days_since('_last_sale', cutoff).alias('a_days_since_last_sale'),
    ]).drop(['_first_sale', '_last_sale']))

    for days in WINDOWS:
        f = f.join(
            _hist(transactions, cutoff, days).group_by('article_id')
            .agg(pl.len().alias(f'a_sales_last_{days}d')),
            on='article_id', how='left'
        ).with_columns(pl.col(f'a_sales_last_{days}d').fill_null(0))

    f = f.with_columns([
        (pl.col('a_sales_last_7d') / pl.col('a_sales_last_14d').clip(lower_bound=1)).alias('a_trend_7_14'),
        (pl.col('a_sales_last_14d') / pl.col('a_sales_last_30d').clip(lower_bound=1)).alias('a_trend_14_30'),
        (pl.col('a_sales_last_30d') / pl.col('a_sales_last_60d').clip(lower_bound=1)).alias('a_trend_30_60'),
        # 6位: 平常価格(max/mean)に対する直近価格の比 = 値引き率
        (pl.col('a_price_mean') / pl.col('_price_max')).alias('a_discount_mean_vs_max'),
    ]).drop('_price_max')

    # 6位の値引き率の本体: 直近7日の実売価格 / 平常価格
    f = f.join(
        _hist(transactions, cutoff, 7).group_by('article_id')
        .agg(pl.col('price').mean().alias('_price_7d')),
        on='article_id', how='left'
    ).with_columns(
        (pl.col('_price_7d') / pl.col('a_price_mean')).alias('a_discount_7d')
    ).drop('_price_7d')

    ua = df.group_by(['customer_id', 'article_id']).agg(pl.len().alias('n'))
    f = f.join(ua.group_by('article_id').agg([
        pl.col('n').mean().alias('a_avg_purchase_count_per_buyer'),
        (pl.col('n') > 1).mean().alias('a_repeat_rate')]), on='article_id', how='left')

    # 9位: この商品を買った顧客の平均年齢（直近7/30日）
    ages = customers.select(['customer_id', 'age'])
    for days in (7, 30):
        f = f.join(
            _hist(transactions, cutoff, days).join(ages, on='customer_id', how='left')
            .group_by('article_id').agg([
                pl.col('age').mean().alias(f'a_buyer_age_mean_{days}d'),
                pl.col('age').std().alias(f'a_buyer_age_std_{days}d')]),
            on='article_id', how='left')

    attrs = articles.select(['article_id', 'product_type_no', 'garment_group_no',
                             'index_group_no', 'section_no', 'department_no', 'product_code'])
    variant = articles.group_by('product_code').agg(pl.len().alias('a_variant_count'))
    f = (f.join(attrs, on='article_id', how='left')
         .join(variant, on='product_code', how='left')
         .with_columns(pl.col('a_variant_count').fill_null(1))
         .drop('product_code'))
    f.write_parquet(path)
    return f


# ヒント: LightGBM/CatBoost に categorical_features として明示指定する列（6位: +0.0005〜0.0008）
CATEGORICAL_FEATURES = ['product_type_no', 'garment_group_no', 'index_group_no',
                        'section_no', 'department_no', 'age_bin',
                        'club_member_status_enc', 'fashion_news_frequency_enc']


# --------------------------------------------------------------- ユーザー×商品

def add_user_article_features(candidates: pl.DataFrame, transactions: pl.DataFrame,
                              articles: pl.DataFrame, customers: pl.DataFrame,
                              cutoff: date) -> pl.DataFrame:
    """候補 (customer_id, article_id, ...) に交互作用特徴を足す。

    9位が「重要度上位10はすべて再購入系」と報告している通り、この塊が一番効く。
    """
    df = _hist(transactions, cutoff)
    out = candidates

    # (user, item) の購入回数と最終購入からの経過日数（全期間 + 窓）
    ua = (df.group_by(['customer_id', 'article_id'])
          .agg([pl.len().alias('ua_count'), pl.col('t_dat').max().alias('_last')])
          .with_columns(_days_since('_last', cutoff).alias('ua_days_since_last')).drop('_last'))
    out = (out.join(ua, on=['customer_id', 'article_id'], how='left')
           .with_columns([pl.col('ua_count').fill_null(0),
                          pl.col('ua_days_since_last').fill_null(-1)]))
    for days in (7, 30):
        out = out.join(
            _hist(transactions, cutoff, days).group_by(['customer_id', 'article_id'])
            .agg(pl.len().alias(f'ua_count_{days}d'),),
            on=['customer_id', 'article_id'], how='left'
        ).with_columns(pl.col(f'ua_count_{days}d').fill_null(0))

    # 年齢ビン別の、その商品の購入率
    age = customers.select(['customer_id', pl.col('age_bin').fill_null(-1)])
    dfa = df.join(age, on='customer_id', how='left')
    rate = (dfa.group_by(['age_bin', 'article_id']).agg(pl.len().alias('c'))
            .join(dfa.group_by('age_bin').agg(pl.len().alias('t')), on='age_bin', how='left')
            .select(['age_bin', 'article_id', (pl.col('c') / pl.col('t')).alias('ua_age_bin_rate')]))
    out = (out.join(age, on='customer_id', how='left')
           .join(rate, on=['age_bin', 'article_id'], how='left')
           .with_columns(pl.col('ua_age_bin_rate').fill_null(0)))

    # 候補商品と同じ属性をどれだけ買っているか / 最後にその属性を買ってから何日か
    cats = ['product_type_no', 'garment_group_no', 'index_group_no', 'department_no']
    art_cats = articles.select(['article_id'] + cats)
    dfc = df.join(art_cats, on='article_id', how='left')
    totals = df.group_by('customer_id').agg(pl.len().alias('_u_total'))
    out = out.join(art_cats, on='article_id', how='left')
    for c in cats:
        uc = (dfc.group_by(['customer_id', c]).agg([
            pl.len().alias(f'uc_{c}_count'),
            pl.col('t_dat').max().alias('_l')])
            .join(totals, on='customer_id', how='left')
            .with_columns([
                (pl.col(f'uc_{c}_count') / pl.col('_u_total')).alias(f'uc_{c}_ratio'),
                _days_since('_l', cutoff).alias(f'uc_{c}_days_since'),
            ]).drop(['_u_total', '_l']))
        out = (out.join(uc, on=['customer_id', c], how='left')
               .with_columns([pl.col(f'uc_{c}_count').fill_null(0),
                              pl.col(f'uc_{c}_ratio').fill_null(0),
                              pl.col(f'uc_{c}_days_since').fill_null(-1)]))
    return out


def add_price_gap(candidates: pl.DataFrame) -> pl.DataFrame:
    """顧客の平均価格と商品の平均価格の差（指示書の基本集計特徴）。

    user/article 特徴を join したあとに呼ぶこと。
    """
    return candidates.with_columns([
        (pl.col('u_price_mean') - pl.col('a_price_mean')).alias('ua_price_gap'),
        (pl.col('a_price_mean') / pl.col('u_price_mean')).alias('ua_price_ratio'),
    ])


def _shrink(df: pl.DataFrame) -> pl.DataFrame:
    """Float64 を Float32 に落とす。行数が数百万〜数千万なので効きが大きい。"""
    return df.with_columns([pl.col(c).cast(pl.Float32)
                            for c, t in zip(df.columns, df.dtypes) if t == pl.Float64])


def build_all_features(candidates: pl.DataFrame, transactions: pl.DataFrame,
                       articles: pl.DataFrame, customers: pl.DataFrame,
                       cutoff: date, cache: bool = True,
                       n_chunks: int = 1) -> pl.DataFrame:
    """候補にユーザー/商品/交互作用の特徴を全部足す。

    n_chunks > 1 で候補を customer_id の剰余で分割して処理する。
    ユーザー/商品特徴はチャンク間で使い回すので、増えるのは join の回数だけ。
    """
    uf = add_user_category_features(
        build_user_features(transactions, customers, cutoff, cache),
        transactions, articles, cutoff)
    af = build_article_features(transactions, articles, customers, cutoff, cache)
    af = af.drop([c for c in ('product_type_no', 'garment_group_no', 'index_group_no',
                              'department_no') if c in af.columns])

    def one(cand: pl.DataFrame) -> pl.DataFrame:
        out = add_user_article_features(cand, transactions, articles, customers, cutoff)
        out = out.join(uf, on='customer_id', how='left').join(af, on='article_id', how='left')
        return _shrink(add_price_gap(out))

    if n_chunks <= 1:
        return one(candidates)
    return pl.concat([one(candidates.filter(pl.col('customer_id') % n_chunks == i))
                      for i in range(n_chunks)], how='vertical')
