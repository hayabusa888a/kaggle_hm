"""候補200のデータセットを1 cutoff ずつ構築する。

使い方: python build_k200.py <cutoff> [valid|train_neg30]
  既定は train_neg30（負例を正例の30倍に間引いた学習用）。
  valid を指定すると間引かずに全候補を作る（評価用）。

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
tag = sys.argv[2] if len(sys.argv) > 2 else 'train_neg30'
TOP_K = 200
NEG_RATIO = 30.0

if dataset_path(cutoff, TOP_K, tag).exists():
    print('{} [{}]: キャッシュ済み'.format(cutoff, tag), flush=True)
    raise SystemExit(0)

trans, customers, articles = load_converted()
a = get_actuals(trans, cutoff)
sample_fn = None if tag == 'valid' else (lambda d: downsample_negatives(d, NEG_RATIO))
with timer() as t:
    ds = build_dataset(
        trans, customers, articles, cutoff, actuals=a, top_k=TOP_K,
        customer_filter=pl.Series('customer_id', list(a.keys()), dtype=pl.Int32),
        sample_fn=sample_fn, n_chunks=16 if tag == 'valid' else 8, cache_tag=tag)
print('{} [{}]: {:,}行 正例{:,} ({:.0f}s)'.format(
    cutoff, tag, len(ds), ds['label'].sum(), t()), flush=True)
