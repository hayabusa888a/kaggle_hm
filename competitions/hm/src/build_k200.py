"""提出用: 候補200のデータセットを1 cutoff ずつ構築する。

使い方: python build_k200.py <cutoff>
1プロセス1 cutoff にするのは、終了時にOSへメモリが返るのを利用してピークを抑えるため。
"""
import sys
from datetime import date

import polars as pl

from hm.config import load_converted
from hm.metrics import get_actuals
from hm.exp_log import timer
from hm.pipeline import build_dataset, dataset_path
from hm.rerank import downsample_negatives

cutoff = date.fromisoformat(sys.argv[1])
TOP_K = 200
NEG_RATIO = 30.0

if dataset_path(cutoff, TOP_K, 'train_neg30').exists():
    print('{}: キャッシュ済み'.format(cutoff), flush=True)
    raise SystemExit(0)

trans, customers, articles = load_converted()
a = get_actuals(trans, cutoff)
with timer() as t:
    ds = build_dataset(
        trans, customers, articles, cutoff, actuals=a, top_k=TOP_K,
        customer_filter=pl.Series('customer_id', list(a.keys()), dtype=pl.Int32),
        sample_fn=lambda d: downsample_negatives(d, NEG_RATIO),
        n_chunks=8, cache_tag='train_neg30')
print('{}: {:,}行 正例{:,} ({:.0f}s)'.format(cutoff, len(ds), ds['label'].sum(), t()), flush=True)
