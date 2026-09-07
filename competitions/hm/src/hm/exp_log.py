"""実験ログ。施策名 / recall@K / MAP@12 / 実行時間 を1行ずつ追記する。"""
from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime

import polars as pl

from .config import EXPERIMENT_DIR

LOG_PATH = EXPERIMENT_DIR / 'experiment_log.csv'

COLUMNS = [
    'timestamp', 'stage', 'name', 'recall_at_k', 'k',
    'map_at_12', 'runtime_sec', 'n_candidates_per_user', 'note',
]


@contextmanager
def timer():
    """with timer() as t: ... のあと t() で経過秒。"""
    start = time.time()
    elapsed = {}
    yield lambda: elapsed.get('sec', time.time() - start)
    elapsed['sec'] = time.time() - start


def log_experiment(name: str,
                   stage: str,
                   recall_at_k: float | None = None,
                   k: int | None = None,
                   map_at_12: float | None = None,
                   runtime_sec: float | None = None,
                   n_candidates_per_user: float | None = None,
                   note: str = '') -> pl.DataFrame:
    """1実験ぶんを追記する。stage は 'candidate' か 'rerank'。"""
    if stage not in ('candidate', 'rerank'):
        raise ValueError("stage must be 'candidate' or 'rerank'")
    row = pl.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stage': stage,
        'name': name,
        'recall_at_k': recall_at_k,
        'k': k,
        'map_at_12': map_at_12,
        'runtime_sec': runtime_sec,
        'n_candidates_per_user': n_candidates_per_user,
        'note': note,
    }], schema={
        'timestamp': pl.Utf8, 'stage': pl.Utf8, 'name': pl.Utf8,
        'recall_at_k': pl.Float64, 'k': pl.Int32, 'map_at_12': pl.Float64,
        'runtime_sec': pl.Float64, 'n_candidates_per_user': pl.Float64, 'note': pl.Utf8,
    })
    if LOG_PATH.exists():
        row = pl.concat([read_log(), row], how='vertical')
    row.write_csv(LOG_PATH)
    return row.tail(1)


def read_log() -> pl.DataFrame:
    if not LOG_PATH.exists():
        return pl.DataFrame(schema={c: pl.Utf8 for c in COLUMNS})
    return pl.read_csv(LOG_PATH, schema_overrides={
        'recall_at_k': pl.Float64, 'k': pl.Int32, 'map_at_12': pl.Float64,
        'runtime_sec': pl.Float64, 'n_candidates_per_user': pl.Float64,
    })


def print_log(stage: str | None = None) -> None:
    df = read_log()
    if stage is not None:
        df = df.filter(pl.col('stage') == stage)
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=60):
        print(df)
