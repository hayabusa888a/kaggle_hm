"""実験16: Optuna による LightGBM のハイパーパラメータ探索。

きっかけ（実験13のアンサンブル）
  同じデータ・同じ特徴で、パラメータを変えるだけで単体 CV が動いた。
    lr=0.03 leaves=127  -> 0.04093
    lr=0.05 leaves=95   -> 0.04091
    lr=0.05 leaves=63   -> 0.04053  （現行の提出構成）
    lr=0.08 leaves=31   -> 0.04028
  「学習率を下げ木を深くするほど良い」という単調な傾向が出たので、
  境界を広げて探索する。BPRチューニング(1.5時間)と同じ +0.0004 が
  パラメータだけで得られており、費用対効果が高い。

**AUC ではなく MAP@12 を直接最適化する。**
実験12で BPR単体AUC と MAP@12 が逆相関したため、代理指標は信用しない。

過学習の注意
  探索も評価も同じ valid週(2020-09-15)で行うので、CV は楽観側に出る。
  11位が optuna で blend 重みを最適化したとき
  「I think it is overestimated」と書いているのと同じ状況。
  最終判断は必ず LB 実測で行うこと。CV->LB 写像は 0.831（3点で実測済み）。

使い方: python exp_16_optuna.py [試行数] [週数] [top_k]
"""
from __future__ import annotations

import gc
import sys
from datetime import timedelta

import numpy as np
import optuna
import polars as pl

from hm.config import VALID_CUTOFF, load_converted
from hm.metrics import get_actuals, evaluate_ranking
from hm.exp_log import log_experiment, timer
from hm.pipeline import dataset_path, feature_columns
from hm.embeddings import add_bpr_similarity, add_user2item_similarity
from hm.rerank import train_lgb_binary, predict

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
N_WEEKS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
TOP_K = int(sys.argv[3]) if len(sys.argv) > 3 else 200
BPR_DIM, BPR_ITERS = 128, 300
VALID = VALID_CUTOFF
CUTS = [VALID - timedelta(days=7 * i) for i in range(1, N_WEEKS + 1)]

trans, customers, articles = load_converted()
n_items = int(articles['article_id'].max()) + 1
va = get_actuals(trans, VALID)


def load(cutoff, tag):
    df = pl.read_parquet(dataset_path(cutoff, TOP_K, tag))
    df = add_user2item_similarity(df, trans, cutoff, n_items)
    return add_bpr_similarity(df, trans, cutoff, n_items,
                              dim=BPR_DIM, iterations=BPR_ITERS)


valid = load(VALID, 'valid')
cols = feature_columns(valid)
train = pl.concat([load(c, 'train_neg30').select(cols + ['label']) for c in CUTS],
                  how='vertical')
# 巨大な生データはもう要らない。試行ごとに残しておくとメモリを圧迫する。
del articles, customers
gc.collect()
print('設定: 学習{}週 top_k={} BPR({},{})  train {:,}行 / valid {:,}行'.format(
    N_WEEKS, TOP_K, BPR_DIM, BPR_ITERS, len(train), len(valid)), flush=True)
print('基準（実験13の最良単体 lr=0.03 leaves=127）: 0.04093\n', flush=True)


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float('learning_rate', 0.01, 0.08, log=True)
    params = {
        'learning_rate': lr,
        # 実験13で「深いほど良い」だったので上限を大きく取る
        'num_leaves': trial.suggest_int('num_leaves', 63, 511, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 500, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': 1,
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
        'verbosity': -1,
        'num_threads': 0,
        'seed': 42,
    }
    # 学習率に反比例させて本数を決める（lr だけ下げて本数据え置きだと学習不足になる）
    n_rounds = int(np.clip(300 * 0.05 / lr, 150, 1200))
    trial.set_user_attr('num_boost_round', n_rounds)

    with timer() as t:
        model, _ = train_lgb_binary(train, cols, categorical=[], params=params,
                                    num_boost_round=n_rounds, seed=42)
        pred = predict(model, valid, cols)
        m = evaluate_ranking(pred, va, ks=[12])['map'][0]
    del model, pred
    gc.collect()
    # 打ち切っても復元できるよう全パラメータを出す（実験16を途中で止めて
    # min_data_in_leaf 等が失われた反省）
    print('  trial {:>3}  MAP@12={:.5f}  lr={:.4f} leaves={} rounds={} ({:.0f}s)'.format(
        trial.number, m, lr, params['num_leaves'], n_rounds, t()), flush=True)
    print('       params={}'.format(
        {k: v for k, v in params.items() if k not in ('verbosity', 'num_threads')}),
        flush=True)
    return m


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=8))
# 実験13の最良を初期値として渡し、そこから探索させる
study.enqueue_trial({'learning_rate': 0.03, 'num_leaves': 127, 'min_data_in_leaf': 100,
                     'feature_fraction': 0.8, 'bagging_fraction': 0.8,
                     'lambda_l1': 1e-8, 'lambda_l2': 1e-8, 'min_gain_to_split': 0.0})
study.optimize(objective, n_trials=N_TRIALS)

print('\n=== 最良 ===')
print('MAP@12: {:.5f}  (基準 0.04093 から {:+.5f})'.format(
    study.best_value, study.best_value - 0.04093))
print('num_boost_round: {}'.format(study.best_trial.user_attrs['num_boost_round']))
for k, v in study.best_params.items():
    print('  {:<20}{}'.format(k, v))

log_experiment(name='optuna/best_of_{}'.format(N_TRIALS), stage='rerank',
               map_at_12=study.best_value, k=12, n_candidates_per_user=TOP_K,
               note='Optuna {}試行 学習{}週 BPR({},{}) params={}'.format(
                   N_TRIALS, N_WEEKS, BPR_DIM, BPR_ITERS, study.best_params))

print('\n=== 上位8試行 ===')
rows = sorted([t for t in study.trials if t.value is not None],
              key=lambda t: -t.value)[:8]
for t in rows:
    print('  {:.5f}  lr={:.4f} leaves={:>3} minleaf={:>3} ff={:.2f} bf={:.2f}'.format(
        t.value, t.params['learning_rate'], t.params['num_leaves'],
        t.params['min_data_in_leaf'], t.params['feature_fraction'],
        t.params['bagging_fraction']))
