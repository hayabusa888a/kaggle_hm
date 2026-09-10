"""実験21: u2tag2i を加えたときのプールの変化。

単体性能は弱かったが、判断材料は「プールに新しい正解を持ち込めるか」。
実験10の診断では取りこぼしの72%が「商品はプールにあるがその顧客に紐付かない」
ケースだったので、顧客->商品の経路を増やす u2tag2i はそこを狙っている。

比較の基準（valid週・10戦略・候補200相当）:
  recall@100 0.1627 / recall@200 0.2226 / recall@300 0.2541 / K無制限 0.2689
  HitNum@100 30,133 / @200 42,591
"""
from __future__ import annotations

import gc

import polars as pl

from hm.config import VALID_CUTOFF, load_converted
from hm.metrics import (get_actuals, actuals_to_frame, evaluate_candidates, hits_at_k)
from hm.exp_log import log_experiment, timer
from hm.pipeline import build_sources, KEYED
from hm.combine import combine
import hm.candidates as C

CUTOFF = VALID_CUTOFF
TOP_N = 100
KS = [12, 100, 200, 300, 1000]
NEW = ['u2tag_section_no', 'u2tag_department_no', 'u2tag_product_type_no']

trans, customers, articles = load_converted()
actuals = get_actuals(trans, CUTOFF)
valid_ids = pl.Series('customer_id', list(actuals.keys()), dtype=pl.Int32)
truth = actuals_to_frame(actuals)
n_pos = len(truth)
print('valid顧客 {:,} / 正解ペア {:,}'.format(len(actuals), n_pos), flush=True)

base = build_sources(trans, customers, articles, CUTOFF, TOP_N,
                     customer_filter=valid_ids, build_missing=False)
print('既存戦略: {}'.format(len(base)), flush=True)

extra = {}
for name in NEW:
    df = C.load_cached(name, CUTOFF, TOP_N, customer_filter=valid_ids)
    if df is not None:
        extra[name] = df
print('追加戦略: {}'.format(list(extra)), flush=True)


def report(label, srcs):
    with timer() as t:
        pool = combine(srcs, with_meta=False)
        res = {r['k']: r for r in evaluate_candidates(pool, actuals, ks=KS).to_dicts()}
    per_user = len(pool) / pool['customer_id'].n_unique()
    h100, h200 = hits_at_k(pool, actuals, 100), hits_at_k(pool, actuals, 200)
    print('\n[{}]  戦略{}  候補/人={:.1f}  ({:.0f}s)'.format(label, len(srcs), per_user, t()), flush=True)
    print('  {:>6}{:>10}'.format('K', 'recall'), flush=True)
    for k in KS:
        print('  {:>6}{:>10.4f}'.format(k, res[k]['recall']), flush=True)
    print('  HitNum@100={:,}  @200={:,}'.format(h100, h200), flush=True)
    log_experiment(name='pool/{}'.format(label), stage='candidate',
                   recall_at_k=res[200]['recall'], k=200, map_at_12=res[12]['map'],
                   runtime_sec=t(), n_candidates_per_user=per_user,
                   note='上限={:.4f} HitNum@200={:,}'.format(res[1000]['recall'], h200))
    del pool
    gc.collect()
    return res


r_base = report('10strategies', base)
r_all = report('13strategies_u2tag', {**base, **extra})

# どの属性が独自に効いているか（1つずつ足す）
print('\n=== u2tag を1つずつ足したとき ===', flush=True)
print('{:<26}{:>10}{:>10}{:>10}'.format('追加した戦略', 'R@200', '上限', '差(上限)'), flush=True)
print('{:<26}{:>10.4f}{:>10.4f}{:>10}'.format(
    '(なし)', r_base[200]['recall'], r_base[1000]['recall'], '—'), flush=True)
for name, df in extra.items():
    res = {r['k']: r for r in
           evaluate_candidates(combine({**base, name: df}, with_meta=False),
                               actuals, ks=[200, 1000]).to_dicts()}
    print('{:<26}{:>10.4f}{:>10.4f}{:>+10.4f}'.format(
        name, res[200]['recall'], res[1000]['recall'],
        res[1000]['recall'] - r_base[1000]['recall']), flush=True)
    gc.collect()

# 独自貢献: その戦略だけが拾えた正解ペア
print('\n=== 各戦略の独自貢献（13戦略中） ===', flush=True)
allsrc = {**base, **extra}
hit = {}
for name, df in allsrc.items():
    hit[name] = set(
        df.select(['customer_id', 'article_id']).unique()
        .join(truth, on=['customer_id', 'article_id'], how='inner')
        .select(pl.concat_str(['customer_id', 'article_id'], separator='_'))
        .to_series().to_list())
print('  {:<26}{:>10}{:>10}'.format('戦略', 'ヒット', '独自'), flush=True)
for name in allsrc:
    others = set().union(*[hit[o] for o in allsrc if o != name])
    mark = ' <- 追加' if name in extra else ''
    print('  {:<26}{:>10,}{:>10,}{}'.format(
        name, len(hit[name]), len(hit[name] - others), mark), flush=True)
