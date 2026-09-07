"""実験10: 候補プールは十分リッチか。取りこぼしの内訳を分解する。

比較の物差し（第3部原文）
 - 1位が週ごとの HitNum@100 を公開: 2020-09-16週=39142 / 09-09週=38427 / 09-02週=41019
 - 3位「recall rates increased from ~10% to ~18%」で入賞圏。候補は1顧客あたり数百
 - 9位「valid週の商品の92%は直近30日にも出現 / 顧客の47%のみ直近30日に取引あり」

取りこぼしを3つに分ける。
 A. その顧客に候補が1件も出ていない
 B. 正解商品が候補プール全体（全顧客のunion）に一度も現れない = 商品自体を拾えていない
 C. 商品はプールにあるが、その顧客には出ていない = 紐付けの失敗
Bなら候補戦略の宛先（商品側）が足りない。Cなら顧客と商品を結ぶ戦略が足りない。
打ち手が変わるので分けて測る。
"""
from __future__ import annotations

import polars as pl

from hm.config import VALID_CUTOFF, load_converted
from hm.metrics import get_actuals, actuals_to_frame, evaluate_candidates, hits_at_k
from hm.pipeline import build_sources, PER_CUSTOMER, KEYED
from hm.combine import combine

CUTOFF = VALID_CUTOFF
TOP_N = 100

trans, customers, articles = load_converted()
actuals = get_actuals(trans, CUTOFF)
valid_ids = pl.Series('customer_id', list(actuals.keys()), dtype=pl.Int32)
truth = actuals_to_frame(actuals)
n_pos = len(truth)
n_cust = len(actuals)
print('valid顧客 {:,} / 正解ペア {:,}'.format(n_cust, n_pos))

sources = build_sources(trans, customers, articles, CUTOFF, TOP_N,
                        customer_filter=valid_ids, build_missing=False)
pool = combine(sources, with_meta=False)

# ---------------------------------------------------------- 1位のHitNumと突き合わせ
print('\n=== 1位の HitNum@K と突き合わせ ===')
for k in (12, 100, 200, 300):
    print('  HitNum@{:<4}: {:>7,}'.format(k, hits_at_k(pool, actuals, k)))
print('  1位の同週(2020-09-16週) HitNum@100: 39,142')

res = evaluate_candidates(pool, actuals, ks=[12, 100, 200, 300, 1000])
print('\n=== recall（分母=valid顧客全員） ===')
for r in res.to_dicts():
    print('  recall@{:<5}: {:.4f}'.format(r['k'], r['recall']))
print('  3位が入賞圏に到達したときの recall: ~0.18')

# --------------------------------------------------------------- 取りこぼしの分解
print('\n=== 取りこぼしの内訳 ===')
covered_cust = set(pool['customer_id'].unique().to_list())
pool_items = set(pool['article_id'].unique().to_list())

hit_pairs = pool.join(truth, on=['customer_id', 'article_id'], how='inner').height
miss = truth.filter(
    ~pl.struct(['customer_id', 'article_id']).is_in(
        pool.select(['customer_id', 'article_id'])
        .with_columns(pl.struct(['customer_id', 'article_id']).alias('s'))['s']))

a = miss.filter(~pl.col('customer_id').is_in(pl.Series(list(covered_cust), dtype=pl.Int32))).height
b = miss.filter(pl.col('customer_id').is_in(pl.Series(list(covered_cust), dtype=pl.Int32))
                & ~pl.col('article_id').is_in(pl.Series(list(pool_items), dtype=pl.Int32))).height
c = len(miss) - a - b
print('  拾えた正解ペア            : {:>7,} ({:.1%})'.format(hit_pairs, hit_pairs / n_pos))
print('  A 顧客に候補が1件もない   : {:>7,} ({:.1%})'.format(a, a / n_pos))
print('  B 商品がプールに存在しない: {:>7,} ({:.1%})'.format(b, b / n_pos))
print('  C 商品はあるが紐付かない  : {:>7,} ({:.1%})'.format(c, c / n_pos))

# ------------------------------------------------- 取りこぼした商品はどんな商品か
print('\n=== 取りこぼした商品の素性 ===')
from datetime import timedelta
recent30 = set(trans.filter((pl.col('t_dat') > pl.lit(CUTOFF - timedelta(days=30)))
                            & (pl.col('t_dat') <= pl.lit(CUTOFF)))['article_id'].unique().to_list())
ever = set(trans.filter(pl.col('t_dat') <= pl.lit(CUTOFF))['article_id'].unique().to_list())
missB = set(miss.filter(~pl.col('article_id').is_in(pl.Series(list(pool_items), dtype=pl.Int32)))
            ['article_id'].unique().to_list())
print('  Bで落ちた商品の種類数            : {:,}'.format(len(missB)))
print('    うち直近30日に売れていた       : {:,} ({:.1%})'.format(
    len(missB & recent30), len(missB & recent30) / max(len(missB), 1)))
print('    うち過去に一度も売れていない(新商品): {:,} ({:.1%})'.format(
    len(missB - ever), len(missB - ever) / max(len(missB), 1)))
print('  プールが持つ商品の種類数         : {:,}'.format(len(pool_items)))
print('  直近30日に売れた商品の種類数     : {:,}'.format(len(recent30)))

# ------------------------------------------------------- 顧客セグメント別の recall
print('\n=== 顧客セグメント別 recall@100 ===')
has_recent = set(trans.filter((pl.col('t_dat') > pl.lit(CUTOFF - timedelta(days=30)))
                              & (pl.col('t_dat') <= pl.lit(CUTOFF)))['customer_id'].unique().to_list())
for label, ids in [('直近30日に取引あり', set(actuals) & has_recent),
                   ('直近30日に取引なし', set(actuals) - has_recent)]:
    sub_actuals = {k: v for k, v in actuals.items() if k in ids}
    sub_pool = pool.filter(pl.col('customer_id').is_in(pl.Series(list(ids), dtype=pl.Int32)))
    r = evaluate_candidates(sub_pool, sub_actuals, ks=[100, 1000])
    d = {x['k']: x['recall'] for x in r.to_dicts()}
    print('  {:<20} 顧客{:>7,} ({:>5.1%})  recall@100={:.4f}  上限={:.4f}'.format(
        label, len(ids), len(ids) / n_cust, d[100], d[1000]))

# ------------------------------------------------------------- 戦略ごとの独自貢献
print('\n=== 各戦略が「その戦略だけが拾えた」正解ペア数 ===')
per_strategy = {}
for name, df in sources.items():
    per_strategy[name] = set(
        df.select(['customer_id', 'article_id']).unique()
        .join(truth, on=['customer_id', 'article_id'], how='inner')
        .select(pl.concat_str(['customer_id', 'article_id'], separator='_'))
        .to_series().to_list())
all_names = list(per_strategy)
print('  {:<22}{:>10}{:>10}'.format('戦略', 'ヒット数', '独自'))
for name in all_names:
    others = set().union(*[per_strategy[o] for o in all_names if o != name])
    print('  {:<22}{:>10,}{:>10,}'.format(name, len(per_strategy[name]),
                                          len(per_strategy[name] - others)))
