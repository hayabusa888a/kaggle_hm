"""データセット構築の各段階でメモリと行数を出す。どこで落ちるかを推測せず特定する。"""
import ctypes, gc, sys
from datetime import date
import polars as pl

class _M(ctypes.Structure):
    _fields_=[('l',ctypes.c_ulong),('ml',ctypes.c_ulong),('tp',ctypes.c_ulonglong),
              ('ap',ctypes.c_ulonglong),('tpf',ctypes.c_ulonglong),('apf',ctypes.c_ulonglong),
              ('tv',ctypes.c_ulonglong),('av',ctypes.c_ulonglong),('ae',ctypes.c_ulonglong)]

def avail():
    m=_M(); m.l=ctypes.sizeof(_M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ap/1e9

def note(msg):
    print(f'[avail {avail():5.1f}GB] {msg}', flush=True)

from hm.config import load_converted
from hm.metrics import get_actuals
from hm.pipeline import build_sources
from hm.combine import combine
from hm.rerank import attach_labels, downsample_negatives
import hm.features as F

CUT = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2020, 9, 8)
TOP_K = 100

note('start')
trans, customers, articles = load_converted()
note(f'loaded transactions {len(trans):,}')

acts = get_actuals(trans, CUT)
ids = pl.Series('customer_id', list(acts.keys()), dtype=pl.Int32)
note(f'actuals {len(acts):,} customers')

srcs = build_sources(trans, customers, articles, CUT, 100, customer_filter=ids)
note(f'sources: ' + ', '.join(f'{k}={len(v)/1e6:.1f}M' for k, v in srcs.items()))

cand = combine(srcs, top_k=TOP_K)
del srcs; gc.collect()
note(f'combined {len(cand):,} rows, {len(cand.columns)} cols')

cand = attach_labels(cand, acts)
note(f'labeled, pos={cand["label"].sum():,}')

cand = downsample_negatives(cand, 30.0)
gc.collect()
note(f'downsampled {len(cand):,} rows')

ds = F.build_all_features(cand, trans, articles, customers, CUT, n_chunks=4)
note(f'features {ds.shape}  推定 {ds.estimated_size()/1e9:.2f}GB')
