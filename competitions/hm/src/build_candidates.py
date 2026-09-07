"""指定 cutoff の候補を1戦略ずつ生成してディスクに置く。

使い方: python build_candidates.py 2020-09-08 [strategy ...]

戦略ごとにプロセスを分けるのは、ホストの空きメモリが数GB規模しかなく、
1プロセスで全戦略を回すと中間結果が積み上がって落ちるため。
プロセス終了でOSにメモリが返るので、これが一番確実。
"""
import sys
from datetime import date

import polars as pl

from hm.config import load_converted
from hm.exp_log import timer
import hm.candidates as C
from hm.pipeline import ALL_STRATEGIES

cutoff = date.fromisoformat(sys.argv[1])
wanted = sys.argv[2:] or ALL_STRATEGIES
TOP_N = 100

trans, customers, articles = load_converted()
builders = {
    'repurchase': lambda: C.build_repurchase(trans, cutoff, TOP_N),
    'timedecay': lambda: C.build_timedecay(trans, cutoff, TOP_N),
    'same_product_code': lambda: C.build_same_product_code(trans, articles, cutoff, TOP_N),
    'purchase_interval': lambda: C.build_purchase_interval(trans, cutoff, TOP_N),
    'user_cf': lambda: C.build_user_cf(trans, cutoff, TOP_N),
    'also_bought': lambda: C.build_also_bought(trans, cutoff, TOP_N),
    'item2item_cf': lambda: C.build_item2item_cf(trans, cutoff, TOP_N),
    'popular': lambda: C.build_popular(trans, cutoff, TOP_N),
    'popular_by_age': lambda: C.build_popular_by_age(trans, customers, cutoff, TOP_N),
    'popular_by_channel': lambda: C.build_popular_by_channel(trans, cutoff, TOP_N),
}

for name in wanted:
    if C.load_cached(name, cutoff, TOP_N) is not None:
        print(f'{name:<20} cached', flush=True)
        continue
    with timer() as t:
        df = builders[name]()
    print(f'{name:<20} {len(df):>12,}行  {t():>6.1f}s', flush=True)
