"""実験07: user2item 類似度を1特徴だけ足したときの効果。

3位の原文:
  "The similarity of user and article is important for ranking. Features about item2item
   similarity, such as count of bought together and word2vec, are commonly used in most
   competitors' model and those also boost my score a lot, but what improves my model
   most is the user2item similarity obtained from BPR matrix factorization.
   This BPR model is trained with all the transactions before the target week
   (I've trained one BPR for each week) using implicit.
   The auc of BPR similarity is ~0.720, while the auc of the whole ranking model is
   ~0.806 and the best auc of other single feature is ~0.680.
   At last, this single similarity feature boost my LB score from 0.03363 to 0.03510"

ここでは implicit が入っていないため TruncatedSVD で同じ量（顧客ベクトルと候補商品
ベクトルの cos 類似）を作る。1位のProNE / 11位・10位のLightFM / 5位・13位のword2vec も
表現が違うだけで同じ発想、と指示書も整理している。

**週ごとに埋め込みを学習し直す**（3位「I've trained one BPR for each week」、
10位「a full set of models for every week」）。5位は全期間word2vecでCV 0.0441に対し
LB 0.0350と乖離させ、本人が原因不明と書いている。同じ轍は踏まない。

検証項目:
 1. u2i_sim あり/なしで MAP@12 がどう動くか
 2. 3位が報告した AUC の並び（モデル全体 0.806 / u2i単体 0.720 / 他の最良単体 0.680）
    と同じ構図になるか

使い方: python exp_07_u2i.py [週数] [top_k]
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
from hm.embeddings import add_user2item_similarity
from hm.rerank import train_lgb_binary, predict

N_WEEKS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 100
VALID = VALID_CUTOFF
CUTS = [VALID - timedelta(days=7 * i) for i in range(1, N_WEEKS + 1)]

trans, customers, articles = load_converted()
n_items = int(articles['article_id'].max()) + 1
va = get_actuals(trans, VALID)

def load_with_u2i(cutoff, tag):
    """キャッシュ済みデータセットを読み、その週の埋め込みで u2i_sim を足す。"""
    path = dataset_path(cutoff, TOP_K, tag)
    if not path.exists():
        raise SystemExit('未生成: {} （先に exp_08 / exp_09 を回すこと）'.format(path))
    df = pl.read_parquet(path)
    return add_user2item_similarity(df, trans, cutoff, n_items)

with timer() as t:
    valid = load_with_u2i(VALID, 'valid')
print('valid: {:,}行 (u2i付与 {:.0f}s)'.format(len(valid), t()), flush=True)
print('  u2i_sim 欠損率: {:.3f}'.format(valid['u2i_sim'].is_null().mean()), flush=True)

cols = feature_columns(valid)
without = [c for c in cols if c != 'u2i_sim']

train = pl.concat([load_with_u2i(c, 'train_neg30').select(cols + ['label']) for c in CUTS],
                  how='vertical')
print('train: {:,}行 正例{:,}  学習週={}'.format(
    len(train), train['label'].sum(), [str(c) for c in CUTS]), flush=True)

print("\n{:<14}{:>8}{:>9}{:>10}{:>7}".format('variant', 'n_feat', 'R@12', 'MAP@12', 'sec'),
      flush=True)
results = {}
for name, cs in [('without_u2i', without), ('with_u2i', cols)]:
    with timer() as t:
        model, _ = train_lgb_binary(train, cs, categorical=[])
        pred = predict(model, valid, cs)
        res = {r['k']: r for r in evaluate_ranking(pred, va, ks=[12]).to_dicts()}
    results[name] = pred
    m = res[12]['map']
    print('{:<14}{:>8}{:>9.4f}{:>10.5f}{:>7.0f}'.format(name, len(cs), res[12]['recall'], m, t()),
          flush=True)
    log_experiment(name='rerank/{},weeks={},k={}'.format(name, N_WEEKS, TOP_K), stage='rerank',
                   recall_at_k=res[12]['recall'], k=12, map_at_12=m, runtime_sec=t(),
                   n_candidates_per_user=TOP_K,
                   note='LGB binary meta=rank cat=none neg30x SVD-u2i(dim32) valid={}'.format(VALID))
    del model; gc.collect()

# 3位が報告した AUC の並びを再現する
y = valid['label'].to_numpy()
whole = roc_auc_score(y, results['with_u2i']['pred_score'].to_numpy())
print('\nモデル全体AUC : {:.4f}   (3位: 0.806)'.format(whole))

aucs = []
for c in cols:
    v = valid[c].to_numpy().astype('float64')
    ok = ~np.isnan(v)
    if ok.sum() < len(v) * 0.5 or len(np.unique(v[ok])) < 2:
        v = np.nan_to_num(v, nan=0.0)
    else:
        v = np.nan_to_num(v, nan=float(np.median(v[ok])))
    if len(np.unique(v)) < 2:
        continue
    a = roc_auc_score(y, v)
    aucs.append((c, max(a, 1 - a)))
aucs.sort(key=lambda x: -x[1])
d = dict(aucs)
print('u2i_sim 単体AUC: {:.4f}   (3位のBPR: 0.720)'.format(d.get('u2i_sim', float('nan'))))
others = [x for x in aucs if x[0] != 'u2i_sim']
print('他の最良単体AUC: {:.4f} ({})   (3位: 0.680)'.format(others[0][1], others[0][0]))
print('\n単体AUC 上位12:')
for c, a in aucs[:12]:
    mark = ' <-- u2i' if c == 'u2i_sim' else ''
    print('  {:<40}{:.4f}{}'.format(c, a, mark))
