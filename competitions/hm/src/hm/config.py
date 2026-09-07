"""パスと定数。コンテナ(/workspace)とホストのどちらから読んでも同じ場所を指す。"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path


def _find_root() -> Path:
    # コンテナでは docker-compose が ./competitions/hm を /workspace にマウントしている
    if Path('/workspace/data/raw').exists():
        return Path('/workspace')
    # ホストから: このファイルは <root>/src/hm/config.py
    return Path(__file__).resolve().parents[2]


ROOT = _find_root()
DATA_DIR = ROOT / 'data' / 'raw'
CONVERTED_DIR = ROOT / 'outputs' / 'converted'
CANDIDATE_DIR = ROOT / 'outputs' / 'candidates'
# 特徴量はコンテナ内 /home/rapids ではなくマウント配下に置く。
# /home/rapids はコンテナを作り直すと消えるので、キャッシュとして信用できない。
FEATURE_DIR = ROOT / 'outputs' / 'features'
MODEL_DIR = ROOT / 'outputs' / 'models'
SUBMISSION_DIR = ROOT / 'outputs' / 'submissions'
EXPERIMENT_DIR = ROOT / 'outputs' / 'experiments'

for _d in (CANDIDATE_DIR, FEATURE_DIR, MODEL_DIR, SUBMISSION_DIR, EXPERIMENT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 9位の validation 設計: 最終週 2020-09-16 以降を holdout。
# cutoff は「学習に使ってよい最後の日」。cutoff+1〜cutoff+7 が正解週。
VALID_CUTOFF = date(2020, 9, 15)   # 正解週 = 2020-09-16 .. 2020-09-22
SUB_CUTOFF = date(2020, 9, 22)    # 提出用（正解週 = テスト週）
LABEL_DAYS = 7

# 候補生成の評価は recall@K、最終評価は MAP@12
EVAL_CUTOFFS = (12, 24, 48, 96, 100, 200, 300)
MAP_K = 12

# 9位の年齢ビン
AGE_BINS = [0, 18, 22, 28, 35, 45, 55, 65, 200]


def load_maps() -> dict:
    """整数ID <-> 元IDの対応表を読む。"""
    with open(CONVERTED_DIR / 'id_maps.json') as f:
        maps = json.load(f)
    return {
        'customer_id_map': maps['customer_id_map'],
        'article_id_map': maps['article_id_map'],
        'customer_id_reverse': {int(v): k for k, v in maps['customer_id_map'].items()},
        'article_id_reverse': {int(v): k for k, v in maps['article_id_map'].items()},
    }


def load_converted():
    """整数変換済みの transactions / customers / articles を polars で読む。"""
    import polars as pl
    return (
        pl.read_parquet(CONVERTED_DIR / 'transactions.parquet'),
        pl.read_parquet(CONVERTED_DIR / 'customers.parquet'),
        pl.read_parquet(CONVERTED_DIR / 'articles.parquet'),
    )
