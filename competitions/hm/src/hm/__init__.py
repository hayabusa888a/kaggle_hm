"""H&M late submission 共通ライブラリ。

candidate generation → reranking の二段構成で使う評価ハーネスと実験ログ。
すべての実験でここの関数だけを使うこと（指標の定義を実験ごとにブレさせない）。
"""
from .config import (
    ROOT, DATA_DIR, CONVERTED_DIR, CANDIDATE_DIR, FEATURE_DIR, MODEL_DIR,
    SUBMISSION_DIR, EXPERIMENT_DIR, EVAL_CUTOFFS, VALID_CUTOFF, SUB_CUTOFF,
    AGE_BINS, load_maps, load_converted,
)
from .metrics import (
    get_actuals, apk, average_precision_at_k, map_at_k, recall_at_k,
    evaluate_candidates, evaluate_ranking, hits_at_k,
)
from .exp_log import log_experiment, read_log, print_log, timer

__all__ = [
    'ROOT', 'DATA_DIR', 'CONVERTED_DIR', 'CANDIDATE_DIR', 'FEATURE_DIR', 'MODEL_DIR',
    'SUBMISSION_DIR', 'EXPERIMENT_DIR', 'EVAL_CUTOFFS', 'VALID_CUTOFF', 'SUB_CUTOFF',
    'AGE_BINS', 'load_maps', 'load_converted',
    'get_actuals', 'apk', 'average_precision_at_k', 'map_at_k', 'recall_at_k',
    'evaluate_candidates', 'evaluate_ranking', 'hits_at_k',
    'log_experiment', 'read_log', 'print_log', 'timer',
]
