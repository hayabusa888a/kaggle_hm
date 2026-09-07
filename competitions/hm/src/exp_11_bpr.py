"""実験11: BPR による user2item 類似度。3位の本命施策。

3位の原文:
  "what improves my model most is the user2item similarity obtained from BPR matrix
   factorization. This BPR model is trained with all the transactions before the target
   week (I've trained one BPR for each week) using implicit.
   The auc of BPR similarity is ~0.720, while the auc of the whole ranking model is
   ~0.806 and the best auc of other single feature is ~0.680.
   At last, this single similarity feature boost my LB score from 0.03363 to 0.03510"

実験07（SVD代替）は単体AUC 0.6275 に留まり、他の最良単体 0.6595 を下回った。
3位は逆に u2i(0.720) > 他の最良(0.680) だったので、順序が逆転している。
目的関数が違う（SVDは二乗誤差=「買っていない=0」、BPRは順位を直接最適化）ことが
原因と考え、implicit の BPR に差し替えて同じ構図が再現するか確認する。

実験10の診断とも整合する: 取りこぼしの72%は「商品はプールにあるが、その顧客に
紐付かない」ケース。顧客×商品の紐付けを学習するモデルがまさに必要な場所。

使い方: python exp_11_bpr.py [週数] [top_k] [次元]
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
from hm.embeddings import add_bpr_similarity, add_user2item_similarity
from hm.rerank import train_lgb_binary, predict

N_WEEKS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 100
DIM = int(sys.argv[3]) if len(sys.argv) > 3 else 64
VALID = VALID_CUTOFF
CUTS = [VALID - timedelta(days=7 * i) for i in range(1, N_WEEKS + 1)]

trans, customers, articles = load_converted()
n_items = int(articles['article_id'].max()) + 1
va = get_actuals(trans, VALID)

def load(cutoff, tag):
    path = dataset_path(cutoff, TOP_K, tag)
    if not path.exists():
        raise SystemExit('未生成: {}'.format(path))
    df = pl.read_parquet(path)
    # SVD版とBPR版を両方載せて比較する。どちらも cutoff 以前のみで学習。
    df = add_user2item_similarity(df, trans, cutoff, n_items)
    return add_bpr_similarity(df, trans, cutoff, n_items, dim=DIM)

with timer() as t:
    valid = load(VALID, 'valid')
print('valid: {:,}行  (埋め込み付与 {:.0f}s)'.format(len(valid), t()), flush=True)
print('  bpr_sim 欠損率: {:.4f}'.format(valid['bpr_sim'].is_null().mean()), flush=True)

cols = feature_columns(valid)
base = [c for c in cols if c not in ('u2i_sim', 'bpr_sim')]

train = pl.concat([load(c, 'train_neg30').select(cols + ['label']) for c in CUTS],
                  how='vertical')
print('train: {:,}行 正例{:,}'.format(len(train), train['label'].sum()), flush=True)

print("\n{:<16}{:>8}{:>9}{:>10}{:>7}".format('variant', 'n_feat', 'R@12', 'MAP@12', 'sec'),
      flush=True)
preds = {}
for name, cs in [('base', base),
                 ('+svd_u2i', base + ['u2i_sim']),
                 ('+bpr', base + ['bpr_sim']),
                 ('+bpr+svd', cols)]:
    with timer() as t:
        model, _ = train_lgb_binary(train, cs, categorical=[])
        pred = predict(model, valid, cs)
        res = {r['k']: r for r in evaluate_ranking(pred, va, ks=[12]).to_dicts()}
    preds[name] = pred
    m = res[12]['map']
    print('{:<16}{:>8}{:>9.4f}{:>10.5f}{:>7.0f}'.format(
        name, len(cs), res[12]['recall'], m, t()), flush=True)
    log_experiment(name='rerank/{},weeks={},k={},dim={}'.format(name, N_WEEKS, TOP_K, DIM),
                   stage='rerank', recall_at_k=res[12]['recall'], k=12, map_at_12=m,
                   runtime_sec=t(), n_candidates_per_user=TOP_K,
                   note='LGB binary meta=rank cat=none neg30x BPR(implicit) valid={}'.format(VALID))
    del model; gc.collect()

# 3位が報告した AUC の構図と突き合わせる
y = valid['label'].to_numpy()
print('\nモデル全体AUC : {:.4f}   (3位: 0.806)'.format(
    roc_auc_score(y, preds['+bpr+svd']['pred_score'].to_numpy())))
for c in ('bpr_sim', 'u2i_sim'):
    v = np.nan_to_num(valid[c].to_numpy().astype('float64'), nan=0.0)
    a = roc_auc_score(y, v)
    print('{:<10} 単体AUC: {:.4f}'.format(c, max(a, 1 - a)))
best = 0.0
bestc = ''
for c in base:
    v = valid[c].to_numpy().astype('float64')
    v = np.nan_to_num(v, nan=0.0)
    if len(np.unique(v)) < 2:
        continue
    a = roc_auc_score(y, v)
    a = max(a, 1 - a)
    if a > best:
        best, bestc = a, c
print('他の最良単体AUC: {:.4f} ({})   (3位: 0.680)'.format(best, bestc))
print('\n3位の構図: u2i(0.720) > 他の最良(0.680)。同じ順序になっていれば代替成功。')
