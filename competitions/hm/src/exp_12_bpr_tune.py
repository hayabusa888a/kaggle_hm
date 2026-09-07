"""実験12: BPR のハイパーパラメータ。単体AUC を 3位の 0.720 に近づけられるか。

実験11の結果:
  bpr_sim 単体AUC = 0.6841（3位のBPR: 0.720）
  モデル全体AUC   = 0.8156（3位: 0.806）-> 全体は既に上回っている
残る差は BPR 特徴そのものの質。次元64・反復100は既定値のまま未探索なので振ってみる。

3位の原文は次元も反復数も書いていない（"using implicit" のみ）。
9位は word2vec について「I have tried sizes 32 and 64 and the performance of size 64
is slightly better than size 32」と書いており、64以上を試す価値はある。

比較基準（実験11、学習2週・候補100）:
  base 0.03673 / +bpr(d64,it100) 0.03977 / bpr単体AUC 0.6841

使い方: python exp_12_bpr_tune.py [週数]
"""
from __future__ import annotations

import gc
import sys
from datetime import timedelta

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from hm.config import VALID_CUTOFF, load_converted
from hm.metrics import get_actuals, evaluate_ranking
from hm.exp_log import log_experiment, timer
from hm.pipeline import dataset_path, feature_columns
from hm.embeddings import add_bpr_similarity
from hm.rerank import train_lgb_binary, predict

N_WEEKS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TOP_K = 100
VALID = VALID_CUTOFF
CUTS = [VALID - timedelta(days=7 * i) for i in range(1, N_WEEKS + 1)]
# 4条件の実測（学習2週・候補100）:
#   (64,100)=0.03977 / (128,100)=0.03951 / (64,300)=0.04004 / (128,300)=0.04017
# -> iters は一貫して効くが dim はほぼ効かない（iters=100では悪化）。
# 残りは「iters をさらに伸ばすと頭打ちか」「dimとitersを両方増やすとどうか」を見る。
GRID = [(256, 100), (128, 600), (256, 300)]

trans, customers, articles = load_converted()
n_items = int(articles['article_id'].max()) + 1
va = get_actuals(trans, VALID)

raw_valid = pl.read_parquet(dataset_path(VALID, TOP_K, 'valid'))
raw_train = {c: pl.read_parquet(dataset_path(c, TOP_K, 'train_neg30')) for c in CUTS}
print('valid {:,}行 / train {} 週'.format(len(raw_valid), len(CUTS)), flush=True)

print("\n{:>6}{:>7}{:>10}{:>9}{:>10}{:>8}".format(
    'dim', 'iters', 'BPR単体AUC', 'R@12', 'MAP@12', 'sec'), flush=True)

y = raw_valid['label'].to_numpy()
for dim, iters in GRID:
    with timer() as t:
        valid = add_bpr_similarity(raw_valid, trans, VALID, n_items,
                                   dim=dim, iterations=iters)
        cols = feature_columns(valid)
        train = pl.concat(
            [add_bpr_similarity(raw_train[c], trans, c, n_items,
                                dim=dim, iterations=iters).select(cols + ['label'])
             for c in CUTS], how='vertical')
        model, _ = train_lgb_binary(train, cols, categorical=[])
        pred = predict(model, valid, cols)
        res = {r['k']: r for r in evaluate_ranking(pred, va, ks=[12]).to_dicts()}
    v = np.nan_to_num(valid['bpr_sim'].to_numpy().astype('float64'), nan=0.0)
    auc = roc_auc_score(y, v)
    auc = max(auc, 1 - auc)
    m = res[12]['map']
    print('{:>6}{:>7}{:>10.4f}{:>9.4f}{:>10.5f}{:>8.0f}'.format(
        dim, iters, auc, res[12]['recall'], m, t()), flush=True)
    log_experiment(name='bpr_tune/dim={},iters={}'.format(dim, iters), stage='rerank',
                   recall_at_k=res[12]['recall'], k=12, map_at_12=m, runtime_sec=t(),
                   n_candidates_per_user=TOP_K,
                   note='BPR単体AUC={:.4f} (3位0.720) 学習{}週 valid={}'.format(
                       auc, N_WEEKS, VALID))
    del train, model, pred, valid
    gc.collect()
