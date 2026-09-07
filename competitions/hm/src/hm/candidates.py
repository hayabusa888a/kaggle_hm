"""候補生成（第1段）。すべて polars / CPU、cutoff より後のデータには触れない。

リーク規律: どの関数も transactions を `t_dat <= cutoff` で切ってから使う。
週 W の候補を作るのに W 以降の情報を使わない（10位が徹底し CV-LB 乖離ゼロを達成した点）。

候補は 2 つの形で保存する。
 - 顧客ごと : (customer_id, article_id, score, rank)
 - キー付き : (<key>, article_id, score, rank) … 人気系。全顧客ぶん展開すると
              1.37M x 100 行になって無駄なので、キー単位で持ち評価/結合時に展開する。
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from .config import CANDIDATE_DIR

KEYED_STRATEGIES = {'popular', 'popular_by_age', 'popular_by_channel'}


# ------------------------------------------------------------------ 入出力

def cand_path(name: str, cutoff: date, top_n: int) -> 'object':
    return CANDIDATE_DIR / f'{name}_{cutoff}_top{top_n}.parquet'


def load_cached(name: str, cutoff: date, top_n: int,
                customer_filter: 'pl.Series | None' = None) -> pl.DataFrame | None:
    """保存済み候補を読む。

    customer_filter を渡すと scan_parquet で絞ってから materialize する。
    全顧客ぶん（repurchase は 4000万行超）を一度メモリに載せてから filter すると
    数GBの一時領域を食って落ちるため、読む側で絞りきる。
    """
    path = cand_path(name, cutoff, top_n)
    if not path.exists():
        return None
    lf = pl.scan_parquet(path)
    if customer_filter is not None:
        lf = lf.filter(pl.col('customer_id').is_in(customer_filter))
    return lf.collect()


def _finish(df: pl.DataFrame, name: str, cutoff: date, top_n: int,
            key: str = 'customer_id') -> pl.DataFrame:
    """score 降順に rank を振り、top_n で切って保存する。同点は article_id 昇順。"""
    out = (
        df.sort([key, 'score', 'article_id'], descending=[False, True, False])
        .with_columns((pl.int_range(pl.len(), dtype=pl.Int32).over(key) + 1).alias('rank'))
        .filter(pl.col('rank') <= top_n)
        .select([pl.col(key).cast(pl.Int32), pl.col('article_id').cast(pl.Int32),
                 pl.col('score').cast(pl.Float32), pl.col('rank')])
    )
    out.write_parquet(cand_path(name, cutoff, top_n))
    return out


def _hist(transactions: pl.DataFrame, cutoff: date, days: int | None = None) -> pl.DataFrame:
    """cutoff までの履歴。days を指定すると直近 days 日に限定する。"""
    df = transactions.filter(pl.col('t_dat') <= pl.lit(cutoff))
    if days is not None:
        df = df.filter(pl.col('t_dat') > pl.lit(cutoff - timedelta(days=days)))
    return df


def expand_keyed(keyed: pl.DataFrame, customer_keys: pl.DataFrame, key: str) -> pl.DataFrame:
    """キー付き候補を (customer_id, article_id, score, rank) に展開する。

    customer_keys は (customer_id, <key>) の2列。
    """
    return (customer_keys.join(keyed, on=key, how='inner')
            .select(['customer_id', 'article_id', 'score', 'rank']))


def _score_in_chunks(hist: pl.DataFrame, pairs: pl.DataFrame, n_chunks: int = 8) -> pl.DataFrame:
    """hist(customer_id, article_id) x pairs(article_id, target_id, w) を顧客ごとに集約する。

    一度に join すると中間結果が数億行になりホストのメモリで落ちるため、
    customer_id の剰余でチャンクに割って部分集約する。結果は同じ。
    """
    out = []
    for i in range(n_chunks):
        part = (hist.filter(pl.col('customer_id') % n_chunks == i)
                .join(pairs, on='article_id', how='inner')
                .group_by(['customer_id', 'target_id'])
                .agg(pl.col('w').sum().alias('score')))
        out.append(part)
    return pl.concat(out).rename({'target_id': 'article_id'})

# ------------------------------------------------------- 既存戦略（top_n を可変化）

TD_A, TD_B, TD_C, TD_D = 2.5e4, 1.5e5, 2e-1, 1e3


def build_timedecay(transactions: pl.DataFrame, cutoff: date, top_n: int = 100,
                    cache: bool = True) -> pl.DataFrame:
    """Trending（byfone/hervind カーネル系）。9位が Trending として使ったもの。"""
    if cache and (c := load_cached('timedecay', cutoff, top_n)) is not None:
        return c
    df = _hist(transactions, cutoff).with_columns(
        (pl.col('t_dat') + ((pl.lit(cutoff) - pl.col('t_dat')).dt.total_days() % 7)
         .cast(pl.Int32) * pl.duration(days=1)).alias('ldbw'))
    weekly = df.group_by(['ldbw', 'article_id']).agg(pl.len().alias('count'))
    last_week = (weekly.filter(pl.col('ldbw') == cutoff)
                 .select(['article_id', pl.col('count').alias('count_targ')]))
    df = (df.join(weekly, on=['ldbw', 'article_id'], how='left')
          .join(last_week, on='article_id', how='left')
          .with_columns(pl.col('count_targ').fill_null(0))
          .with_columns((pl.col('count_targ') / pl.col('count')).alias('quotient'))
          .with_columns((pl.lit(cutoff) - pl.col('t_dat')).dt.total_days()
                        .clip(lower_bound=1).cast(pl.Float64).alias('days_ago'))
          .with_columns((pl.col('quotient') * (
              TD_A / pl.col('days_ago').sqrt()
              + TD_B * (-TD_C * pl.col('days_ago')).exp() - TD_D
          ).clip(lower_bound=0)).alias('score')))
    res = (df.group_by(['customer_id', 'article_id']).agg(pl.col('score').sum())
           .filter(pl.col('score') > 0))
    return _finish(res, 'timedecay', cutoff, top_n)


def build_repurchase(transactions: pl.DataFrame, cutoff: date, top_n: int = 100,
                     cache: bool = True) -> pl.DataFrame:
    """過去に買った商品を購入日順。8位の単体CV 0.029 の戦略。"""
    if cache and (c := load_cached('repurchase', cutoff, top_n)) is not None:
        return c
    res = (_hist(transactions, cutoff)
           .group_by(['customer_id', 'article_id'])
           .agg(pl.max('t_dat').alias('last_date'))
           .with_columns((1.0 / (1.0 + (pl.lit(cutoff) - pl.col('last_date'))
                                 .dt.total_days().cast(pl.Float64))).alias('score'))
           .drop('last_date'))
    return _finish(res, 'repurchase', cutoff, top_n)


def build_same_product_code(transactions: pl.DataFrame, articles: pl.DataFrame, cutoff: date,
                            top_n: int = 100, source_top: int = 50,
                            cache: bool = True) -> pl.DataFrame:
    """購入品と同じ product_code の別カラー/サイズ。8位の same product code（0.014）。"""
    if cache and (c := load_cached('same_product_code', cutoff, top_n)) is not None:
        return c
    src = build_timedecay(transactions, cutoff, top_n=max(source_top, 100))
    ap = articles.select(['article_id', 'product_code'])
    res = (src.filter(pl.col('rank') <= source_top)
           .select(['customer_id', 'article_id', 'score'])
           .join(ap, on='article_id', how='left')
           .join(ap.rename({'article_id': 'variant_id'}), on='product_code', how='left')
           .filter(pl.col('article_id') != pl.col('variant_id'))
           .drop_nulls('variant_id')
           .group_by(['customer_id', pl.col('variant_id').alias('article_id')])
           .agg(pl.col('score').sum().alias('score')))
    return _finish(res, 'same_product_code', cutoff, top_n)


def build_purchase_interval(transactions: pl.DataFrame, cutoff: date, top_n: int = 100,
                            cache: bool = True) -> pl.DataFrame:
    """購入間隔から「そろそろ買い直す頃」の商品を拾う。"""
    if cache and (c := load_cached('purchase_interval', cutoff, top_n)) is not None:
        return c
    g = (_hist(transactions, cutoff)
         .group_by(['customer_id', 'article_id'])
         .agg([pl.col('t_dat').min().alias('first_date'),
               pl.col('t_dat').max().alias('last_date'),
               pl.len().alias('n')])
         .filter(pl.col('n') >= 2))
    res = (g.with_columns([
        ((pl.col('last_date') - pl.col('first_date')).dt.total_days()
         / (pl.col('n') - 1)).alias('interval'),
        (pl.lit(cutoff) - pl.col('last_date')).dt.total_days().alias('elapsed')])
        .filter(pl.col('interval') > 0)
        # 経過日数が平均間隔に近いほど高スコア
        .with_columns((pl.col('n').cast(pl.Float64)
                       / (1.0 + (pl.col('elapsed') - pl.col('interval')).abs()))
                      .alias('score'))
        .select(['customer_id', 'article_id', 'score']))
    return _finish(res, 'purchase_interval', cutoff, top_n)


# --------------------------------------------------------- 人気系（キー付きで保存）

def build_popular(transactions: pl.DataFrame, cutoff: date, top_n: int = 100,
                  recent_days: int = 7, cache: bool = True) -> pl.DataFrame:
    """全体人気。8位「most popular / 直近1週の売上」単体CV 0.017。

    返り値は (dummy, article_id, score, rank)。dummy は常に 0 の展開用キー。
    """
    if cache and (c := load_cached('popular', cutoff, top_n)) is not None:
        return c
    res = (_hist(transactions, cutoff, recent_days)
           .group_by('article_id').agg(pl.len().cast(pl.Float64).alias('score'))
           .with_columns(pl.lit(0, dtype=pl.Int32).alias('dummy')))
    return _finish(res, 'popular', cutoff, top_n, key='dummy')


def build_popular_by_age(transactions: pl.DataFrame, customers: pl.DataFrame, cutoff: date,
                         top_n: int = 100, recent_days: int = 7,
                         cache: bool = True) -> pl.DataFrame:
    """年齢ビン別人気。9位のビン [0,18,22,28,35,45,55,65,200]、8位の単体CV 0.018。"""
    if cache and (c := load_cached('popular_by_age', cutoff, top_n)) is not None:
        return c
    res = (_hist(transactions, cutoff, recent_days)
           .join(customers.select(['customer_id', 'age_bin']), on='customer_id', how='left')
           .with_columns(pl.col('age_bin').fill_null(-1))
           .group_by(['age_bin', 'article_id']).agg(pl.len().cast(pl.Float64).alias('score')))
    return _finish(res, 'popular_by_age', cutoff, top_n, key='age_bin')


def build_popular_by_channel(transactions: pl.DataFrame, cutoff: date, top_n: int = 100,
                             recent_days: int = 7, cache: bool = True) -> pl.DataFrame:
    """販売チャネル別人気。8位「most popular categorized by sales channel id」0.016。

    顧客側のキーは「その顧客が普段使うチャネル」（履歴の最頻値）。
    """
    if cache and (c := load_cached('popular_by_channel', cutoff, top_n)) is not None:
        return c
    res = (_hist(transactions, cutoff, recent_days)
           .group_by(['sales_channel_id', 'article_id'])
           .agg(pl.len().cast(pl.Float64).alias('score'))
           .with_columns(pl.col('sales_channel_id').cast(pl.Int32)))
    return _finish(res, 'popular_by_channel', cutoff, top_n, key='sales_channel_id')


def customer_channel_key(transactions: pl.DataFrame, cutoff: date) -> pl.DataFrame:
    """顧客 -> 主に使うチャネル。popular_by_channel の展開キー。"""
    return (_hist(transactions, cutoff)
            .group_by(['customer_id', 'sales_channel_id']).agg(pl.len().alias('n'))
            .sort(['customer_id', 'n', 'sales_channel_id'], descending=[False, True, False])
            .group_by('customer_id').first()
            .select(['customer_id', pl.col('sales_channel_id').cast(pl.Int32)]))


# --------------------------------------------------------------- user based CF（8位）

def build_user_cf(transactions: pl.DataFrame, cutoff: date, top_n: int = 100,
                  recent_days: int = 7, history_days: int = 90, group_top: int = 50,
                  cache: bool = True) -> pl.DataFrame:
    """8位の user based CF。コメント欄の手順そのまま。

    顧客Xが A,B,C を購入 -> A,B,C それぞれの購入者集団を抽出 -> 各集団の直近1週の売上を
    計算 -> **単純加算**して上位 top_n。集団ごとに top X を取ると履歴の多い顧客ほど候補が
    増え、大きい集団に重みがつかないので加算にする、と本人が明言している。

    実装は 2 段の join に落とす。
      M[i][j] = 「商品 i を買ったことのある顧客集団」が直近 recent_days 日に j を買った回数
              = 全期間履歴 (c,i) と直近 (c,j) を c で join して (i,j) で数える
      score(X,j) = sum_{i in Xの履歴} M[i][j]

    history_days: score 計算時に使う X の履歴の長さ。全期間だと (X,i)x(i,j) が爆発する。
                  実測(実験03): 30日 -> R@100 0.0859 / 被覆46.5%、90日 -> 0.1210 / 73.2%。
                  180日以上はホストの空きメモリでは落ちたため 90 を既定にした。
    group_top:    M を i ごとに上位 group_top 件に切る。無制限だと同じく爆発する。
    """
    if cache and (c := load_cached('user_cf', cutoff, top_n)) is not None:
        return c

    owners = _hist(transactions, cutoff).select(['customer_id', 'article_id']).unique()
    recent = (_hist(transactions, cutoff, recent_days)
              .select(['customer_id', pl.col('article_id').alias('target_id')]))

    # M[i][j]: 商品 i の購入者集団における j の直近売上
    m = (owners.join(recent, on='customer_id', how='inner')
         .group_by(['article_id', 'target_id']).agg(pl.len().cast(pl.Float64).alias('w')))
    m = (m.sort(['article_id', 'w', 'target_id'], descending=[False, True, False])
         .with_columns((pl.int_range(pl.len()).over('article_id') + 1).alias('r'))
         .filter(pl.col('r') <= group_top).drop('r'))

    hist = (_hist(transactions, cutoff, history_days)
            .select(['customer_id', 'article_id']).unique())
    res = _score_in_chunks(hist, m)
    return _finish(res, 'user_cf', cutoff, top_n)


# ------------------------------------------------------------------- i2i（商品間）

def _recent_universe(transactions: pl.DataFrame, cutoff: date, days: int = 30) -> pl.Series:
    """直近 days 日に売れた商品。9位「valid週の商品の92%は直近30日にも出現」を根拠に
    候補の宛先をここに絞る。"""
    return _hist(transactions, cutoff, days)['article_id'].unique()


def _apply_i2i(pairs: pl.DataFrame, transactions: pl.DataFrame, cutoff: date,
               history_days: int, top_n: int, name: str,
               exclude_purchased: bool = True) -> pl.DataFrame:
    """商品間スコア表 pairs(article_id, target_id, w) を顧客の履歴に適用する。"""
    hist = (_hist(transactions, cutoff, history_days)
            .select(['customer_id', 'article_id']).unique())
    res = _score_in_chunks(hist, pairs)
    if exclude_purchased:
        owned = (_hist(transactions, cutoff).select(['customer_id', 'article_id'])
                 .unique().with_columns(pl.lit(True).alias('_owned')))
        res = (res.join(owned, on=['customer_id', 'article_id'], how='left')
               .filter(pl.col('_owned').is_null()).drop('_owned'))
    return _finish(res, name, cutoff, top_n)


def build_also_bought(transactions: pl.DataFrame, cutoff: date, top_n: int = 100,
                      pair_days: int = 180, history_days: int = 30,
                      min_count: int = 10, pair_top: int = 50,
                      cache: bool = True) -> pl.DataFrame:
    """同時購入（同じ顧客・同じ日のバスケット）による i2i。"""
    if cache and (c := load_cached('also_bought', cutoff, top_n)) is not None:
        return c
    basket = _hist(transactions, cutoff, pair_days).select(
        ['customer_id', 't_dat', 'article_id']).unique()
    universe = _recent_universe(transactions, cutoff)
    pairs = (basket.join(basket.rename({'article_id': 'target_id'}),
                         on=['customer_id', 't_dat'], how='inner')
             .filter(pl.col('article_id') != pl.col('target_id'))
             .filter(pl.col('target_id').is_in(universe))
             .group_by(['article_id', 'target_id']).agg(pl.len().cast(pl.Float64).alias('w'))
             .filter(pl.col('w') >= min_count))
    pairs = (pairs.sort(['article_id', 'w', 'target_id'], descending=[False, True, False])
             .with_columns((pl.int_range(pl.len()).over('article_id') + 1).alias('r'))
             .filter(pl.col('r') <= pair_top).drop('r'))
    return _apply_i2i(pairs, transactions, cutoff, history_days, top_n, 'also_bought')


def build_item2item_cf(transactions: pl.DataFrame, cutoff: date, top_n: int = 100,
                       pair_days: int = 90, history_days: int = 30,
                       min_count: int = 5, pair_top: int = 50,
                       cache: bool = True) -> pl.DataFrame:
    """アイテム間の cos 類似（同一顧客の共起 / sqrt(n_i * n_j)）による i2i。

    元実装は scipy の疎行列 + Python ループだったが、共起を polars で数えて
    cos 正規化する形に置き換えた。同じ量を計算していて桁違いに速い。
    """
    if cache and (c := load_cached('item2item_cf', cutoff, top_n)) is not None:
        return c
    ui = _hist(transactions, cutoff, pair_days).select(
        ['customer_id', 'article_id']).unique()
    n_i = ui.group_by('article_id').agg(pl.len().cast(pl.Float64).alias('n'))
    universe = _recent_universe(transactions, cutoff)
    co = (ui.join(ui.rename({'article_id': 'target_id'}), on='customer_id', how='inner')
          .filter(pl.col('article_id') != pl.col('target_id'))
          .filter(pl.col('target_id').is_in(universe))
          .group_by(['article_id', 'target_id']).agg(pl.len().cast(pl.Float64).alias('c'))
          .filter(pl.col('c') >= min_count))
    pairs = (co.join(n_i, on='article_id', how='left')
             .join(n_i.rename({'article_id': 'target_id', 'n': 'n_t'}),
                   on='target_id', how='left')
             .with_columns((pl.col('c') / (pl.col('n') * pl.col('n_t')).sqrt()).alias('w'))
             .select(['article_id', 'target_id', 'w']))
    pairs = (pairs.sort(['article_id', 'w', 'target_id'], descending=[False, True, False])
             .with_columns((pl.int_range(pl.len()).over('article_id') + 1).alias('r'))
             .filter(pl.col('r') <= pair_top).drop('r'))
    return _apply_i2i(pairs, transactions, cutoff, history_days, top_n, 'item2item_cf')
