"""実験19: 最終構成の K 別指標を、候補生成段階とリランキング後で並べる。

提出した最良構成（学習6週 / 候補200 / BPR(128,300) / Optuna最良の全9パラメータ）
そのままで、valid週(2020-09-15)に対する recall@K と MAP@K を K=12..120 で出す。

- 候補生成段階 : candidate_score（10戦略の重み付き結合スコア）で並べたとき
- リランキング後: LightGBM の pred_score で並べたとき

分母は正解を持つ valid顧客 68,984人すべて（候補が出ていない顧客も0点として数える）。
"""
from __future__ import annotations

import gc
import json
from datetime import timedelta
from pathlib import Path

import polars as pl

from hm.config import VALID_CUTOFF, EXPERIMENT_DIR, load_converted
from hm.metrics import get_actuals, evaluate_candidates, evaluate_ranking
from hm.exp_log import timer
from hm.pipeline import dataset_path, feature_columns
from hm.embeddings import add_bpr_similarity, add_user2item_similarity
from hm.rerank import train_lgb_binary, predict

TOP_K, BPR_DIM, BPR_ITERS, N_WEEKS = 200, 128, 300, 6
KS = [12, 24, 48, 96, 120]
VALID = VALID_CUTOFF
CUTS = [VALID - timedelta(days=7 * i) for i in range(1, N_WEEKS + 1)]

cfg = json.loads((EXPERIMENT_DIR / 'night_best_params.json').read_text(encoding='utf-8'))
LGB_PARAMS, ROUNDS = cfg['params'], cfg['rounds']
print('構成: 学習{}週 / 候補{} / BPR({},{}) / rounds={}'.format(
    N_WEEKS, TOP_K, BPR_DIM, BPR_ITERS, ROUNDS), flush=True)
print('LGB: {}'.format(LGB_PARAMS), flush=True)

trans, customers, articles = load_converted()
n_items = int(articles['article_id'].max()) + 1
va = get_actuals(trans, VALID)
print('valid顧客 {:,} / 正解ペア {:,}'.format(
    len(va), sum(len(v) for v in va.values())), flush=True)


def load(cutoff, tag):
    df = pl.read_parquet(dataset_path(cutoff, TOP_K, tag))
    df = add_user2item_similarity(df, trans, cutoff, n_items)
    return add_bpr_similarity(df, trans, cutoff, n_items, dim=BPR_DIM, iterations=BPR_ITERS)


valid = load(VALID, 'valid')
cols = feature_columns(valid)

# ---------- 第1段: 候補生成のスコアで並べたとき
with timer() as t:
    cand_res = evaluate_candidates(valid, va, ks=KS, score_col='candidate_score')
print('候補生成段階の評価 完了 ({:.0f}s)'.format(t()), flush=True)

# ---------- 第2段: リランキング後
train = pl.concat([load(c, 'train_neg30').select(cols + ['label']) for c in CUTS],
                  how='vertical')
del articles, customers
gc.collect()
print('train {:,}行 正例{:,}'.format(len(train), train['label'].sum()), flush=True)

with timer() as t:
    model, _ = train_lgb_binary(train, cols, categorical=[], params=LGB_PARAMS,
                                num_boost_round=ROUNDS)
    pred = predict(model, valid, cols)
    rank_res = evaluate_ranking(pred, va, ks=KS, score_col='pred_score')
print('リランキング後の評価 完了 ({:.0f}s)'.format(t()), flush=True)

# ---------- 出力
c = {r['k']: r for r in cand_res.to_dicts()}
r = {r['k']: r for r in rank_res.to_dicts()}
print('\n' + '=' * 74)
print('valid週 2020-09-16〜09-22 / 分母=正解を持つ全顧客 {:,}人'.format(len(va)))
print('=' * 74)
print('{:>5} | {:>10} {:>10} | {:>10} {:>10} | {:>9}'.format(
    'K', '候補recall', '候補MAP', '再rank recall', '再rank MAP', 'MAP改善'))
print('-' * 74)
for k in KS:
    print('{:>5} | {:>10.4f} {:>10.5f} | {:>13.4f} {:>10.5f} | {:>+9.5f}'.format(
        k, c[k]['recall'], c[k]['map'], r[k]['recall'], r[k]['map'],
        r[k]['map'] - c[k]['map']))

out = {'valid_cutoff': str(VALID), 'n_customers': len(va),
       'config': {'weeks': N_WEEKS, 'top_k': TOP_K,
                  'bpr': [BPR_DIM, BPR_ITERS], 'lgb': LGB_PARAMS, 'rounds': ROUNDS},
       'candidate_stage': {str(k): {'recall': c[k]['recall'], 'map': c[k]['map']} for k in KS},
       'rerank_stage': {str(k): {'recall': r[k]['recall'], 'map': r[k]['map']} for k in KS}}
(EXPERIMENT_DIR / 'final_metrics.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print('\n保存: {}'.format(EXPERIMENT_DIR / 'final_metrics.json'))
