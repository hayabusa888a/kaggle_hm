"""cutoff を渡すと候補一式（と第2段の学習データ）を組み立てる。

週ごとに独立して呼べるようにしてある。10位の
「I trained a full set of models for every week. I tagged the models with the last date
 used to create it. That way it was impossible to leak any information」
に合わせ、cutoff より後を見る処理をここに一切置かない。
"""
from __future__ import annotations

from datetime import date

import polars as pl

from . import candidates as C
from . import features as F
from .combine import combine

PER_CUSTOMER = ['repurchase', 'timedecay', 'same_product_code', 'purchase_interval',
                'user_cf', 'also_bought', 'item2item_cf']
KEYED = {'popular': 'dummy', 'popular_by_age': 'age_bin', 'popular_by_channel': 'sales_channel_id'}
ALL_STRATEGIES = PER_CUSTOMER + list(KEYED)


def build_sources(transactions: pl.DataFrame, customers: pl.DataFrame, articles: pl.DataFrame,
                  cutoff: date, top_n: int = 100,
                  strategies: list[str] | None = None,
                  customer_filter: pl.Series | None = None,
                  build_missing: bool = True) -> dict[str, pl.DataFrame]:
    """戦略名 -> (customer_id, article_id, score, rank) を返す。キー付き人気系は展開済み。"""
    strategies = strategies or ALL_STRATEGIES
    builders = {
        'repurchase': lambda: C.build_repurchase(transactions, cutoff, top_n),
        'timedecay': lambda: C.build_timedecay(transactions, cutoff, top_n),
        'same_product_code': lambda: C.build_same_product_code(transactions, articles, cutoff, top_n),
        'purchase_interval': lambda: C.build_purchase_interval(transactions, cutoff, top_n),
        'user_cf': lambda: C.build_user_cf(transactions, cutoff, top_n),
        'also_bought': lambda: C.build_also_bought(transactions, cutoff, top_n),
        'item2item_cf': lambda: C.build_item2item_cf(transactions, cutoff, top_n),
        'popular': lambda: C.build_popular(transactions, cutoff, top_n),
        'popular_by_age': lambda: C.build_popular_by_age(transactions, customers, cutoff, top_n),
        'popular_by_channel': lambda: C.build_popular_by_channel(transactions, cutoff, top_n),
    }

    keymaps = {
        'dummy': customers.select(['customer_id', pl.lit(0, dtype=pl.Int32).alias('dummy')]),
        'age_bin': customers.select(['customer_id', pl.col('age_bin').fill_null(-1)]),
        'sales_channel_id': None,   # 遅延
    }

    sources: dict[str, pl.DataFrame] = {}
    for name in strategies:
        # キー付き人気系はファイルが数KBなので絞らずに読む
        cf = None if name in KEYED else customer_filter
        df = C.load_cached(name, cutoff, top_n, customer_filter=cf)
        if df is None:
            if not build_missing:
                continue
            builders[name]()   # 生成してディスクに置く
            df = C.load_cached(name, cutoff, top_n, customer_filter=cf)
        if name in KEYED:
            key = KEYED[name]
            if keymaps[key] is None:
                keymaps[key] = C.customer_channel_key(transactions, cutoff)
            km = keymaps[key]
            if customer_filter is not None:
                km = km.filter(pl.col('customer_id').is_in(customer_filter))
            df = C.expand_keyed(df, km, key)
        sources[name] = df
    return sources


def dataset_path(cutoff: date, top_k: int | None, tag: str):
    from .config import FEATURE_DIR
    return FEATURE_DIR / f'dataset_{cutoff}_k{top_k}_{tag}.parquet'


def build_dataset(transactions: pl.DataFrame, customers: pl.DataFrame, articles: pl.DataFrame,
                  cutoff: date, actuals: dict[int, set[int]] | None = None,
                  top_n: int = 100, top_k: int | None = None,
                  strategies: list[str] | None = None,
                  with_meta: bool = True,
                  customer_filter: pl.Series | None = None,
                  sample_fn=None, n_chunks: int = 1,
                  add_u2i: bool = False, n_items: int | None = None,
                  add_salesfc: bool = False,
                  cache_tag: str | None = None) -> pl.DataFrame:
    """候補生成 -> 結合(+メタ特徴) -> ラベル付与 -> (間引き) -> 特徴量付与。

    sample_fn は負例ダウンサンプリングなど。**特徴量を作る前**に適用する。
    22M行ぶんの特徴量を作ってから捨てるとホストのメモリに乗らないため、順序が逆だと落ちる。
    """
    from .rerank import attach_labels

    # 構築に数分かかるので、同じ (cutoff, top_k, tag) は作り直さない
    # （1位の feature store / 10位の feature storing に相当）
    path = dataset_path(cutoff, top_k, cache_tag) if cache_tag else None
    if path is not None and path.exists():
        return pl.read_parquet(path)

    sources = build_sources(transactions, customers, articles, cutoff, top_n,
                            strategies, customer_filter)
    ds = combine(sources, top_k=top_k, with_meta=with_meta)
    if actuals is not None:
        ds = attach_labels(ds, actuals)
    if sample_fn is not None:
        ds = sample_fn(ds)
    ds = F.build_all_features(ds, transactions, articles, customers, cutoff,
                              n_chunks=n_chunks)
    if add_salesfc:
        # 10位の来週売上予測。cutoff 時点で観測済みの週ペアだけで学習した予測器を使う。
        from .sales_forecast import add_sales_forecast
        ds = add_sales_forecast(ds, transactions, cutoff)
    if add_u2i:
        # 3位の user2item 類似度。cutoff ごとに学習した埋め込みしか使わない。
        from .embeddings import add_user2item_similarity
        n = n_items if n_items is not None else int(articles['article_id'].max()) + 1
        ds = add_user2item_similarity(ds, transactions, cutoff, n)
    if path is not None:
        ds.write_parquet(path)
    return ds


NON_FEATURES = {'customer_id', 'article_id', 'label', 't_dat'}


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES and df[c].dtype != pl.Date]
