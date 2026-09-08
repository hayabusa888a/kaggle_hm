"""提出ファイルを作る。

使い方: python make_submission.py <出力名> [学習週数] [top_k]
例:     python make_submission.py lgb_rank_w1 1 100

構成は実験05の最良（meta=rank / categorical明示なし / 負例30倍 / LightGBM binary）。
学習は SUB_CUTOFF(2020-09-22) より前の週で行い、推論は cutoff=2020-09-22 で行う。
valid(09-15) を学習に含めるのは、9位が Radek への回答で選んだ方式
（validでハイパラを決めたあと、最終週を含む全データで再学習して予測する）。3位も同回答。

履歴のない顧客には人気商品を埋める。
1位「履歴なし顧客には人気商品のみ」、8位「空欄を人気商品で埋めるとCVが上がる」。
"""
from __future__ import annotations

import gc
import sys
from datetime import timedelta

from pathlib import Path

import polars as pl

from hm.config import SUB_CUTOFF, SUBMISSION_DIR, DATA_DIR, load_converted, load_maps
from hm.metrics import get_actuals
from hm.exp_log import timer
from hm.pipeline import build_dataset, build_sources, feature_columns
from hm.combine import combine
from hm.rerank import downsample_negatives, train_lgb_binary, predict
import hm.features as F
import hm.candidates as C
from hm.embeddings import add_bpr_similarity, add_user2item_similarity

NAME = sys.argv[1] if len(sys.argv) > 1 else 'lgb_rank'
N_WEEKS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
TOP_K = int(sys.argv[3]) if len(sys.argv) > 3 else 100
NEG_RATIO = 30.0
BATCH = 150_000
BPR_DIM = int(sys.argv[4]) if len(sys.argv) > 4 else 64
BPR_ITERS = int(sys.argv[5]) if len(sys.argv) > 5 else 100
# LightGBM のパラメータ。実験17で既定値と組んでも再現できることを確認した値。
#   現行 lr=0.05/leaves=63      -> 0.04053
#   optuna_t3 lr=0.0188/leaves=188 -> 0.04128 (+0.00075)
# 引数6番目に JSON パスを渡すと、そこに書かれた**全パラメータ**を使う。
# lr/leaves/rounds だけを個別に渡す旧方式も残すが、Optuna の結果を反映するときは
# 必ず JSON を使うこと。3つだけ渡して他を既定値にすると、探索で効いていた
# bagging_fraction や正則化が落ちる（夜間実行でこの取りこぼしが実際に起きた）。
LGB_PARAMS: dict = {}
LGB_ROUNDS = 300
if len(sys.argv) > 6 and sys.argv[6].endswith('.json'):
    import json as _json
    _d = _json.loads(Path(sys.argv[6]).read_text(encoding='utf-8'))
    LGB_PARAMS = dict(_d.get('params', {}))
    LGB_ROUNDS = int(_d.get('rounds', 300))
elif len(sys.argv) > 6:
    LGB_PARAMS = {'learning_rate': float(sys.argv[6])}
    if len(sys.argv) > 7:
        LGB_PARAMS['num_leaves'] = int(sys.argv[7])
    LGB_ROUNDS = int(sys.argv[8]) if len(sys.argv) > 8 else 300
else:
    LGB_PARAMS = {'learning_rate': 0.05, 'num_leaves': 63}

trans, customers, articles = load_converted()
maps = load_maps()
article_rev = maps['article_id_reverse']
N_ITEMS = int(articles['article_id'].max()) + 1

cuts = [SUB_CUTOFF - timedelta(days=7 * i) for i in range(1, N_WEEKS + 1)]
print('設定: 学習{}週 top_k={} BPR(dim={}, iters={}) rounds={}'.format(
    N_WEEKS, TOP_K, BPR_DIM, BPR_ITERS, LGB_ROUNDS), flush=True)
print('LGBパラメータ: {}'.format(LGB_PARAMS), flush=True)
print('学習週: ' + ', '.join(str(c) for c in cuts), flush=True)

parts = []
for c in cuts:
    a = get_actuals(trans, c)
    d = build_dataset(
        trans, customers, articles, c, actuals=a, top_k=TOP_K,
        customer_filter=pl.Series('customer_id', list(a.keys()), dtype=pl.Int32),
        sample_fn=lambda d: downsample_negatives(d, NEG_RATIO),
        n_chunks=4, cache_tag='train_neg30')
    # user2item 類似度。実験11で +bpr+svd が最良（base 0.03673 -> 0.03986）。
    # 埋め込みはその週(cutoff)以前のデータだけで学習する。
    d = add_user2item_similarity(d, trans, c, N_ITEMS)
    parts.append(add_bpr_similarity(d, trans, c, N_ITEMS, dim=BPR_DIM,
                                    iterations=BPR_ITERS))
    print('  {}: {:,}行'.format(c, len(parts[-1])), flush=True)
train = pl.concat(parts, how='vertical')
del parts
gc.collect()

cols = feature_columns(train)
with timer() as t:
    model, _ = train_lgb_binary(train, cols, categorical=[], params=LGB_PARAMS,
                                num_boost_round=LGB_ROUNDS)
print('学習完了 {:,}行 正例{:,} ({:.0f}s)'.format(
    len(train), train['label'].sum(), t()), flush=True)
del train
gc.collect()

# 推論。候補はバッチごとに scan_parquet で絞って読む。
# 全顧客ぶん（repurchase だけで2500万行）を一度に載せると落ちるため。
print('推論 cutoff={}'.format(SUB_CUTOFF), flush=True)
all_ids = sorted(int(v) for v in maps['customer_id_map'].values())
print('  対象顧客: {:,}'.format(len(all_ids)), flush=True)

predictions: dict[int, str] = {}
for i in range(0, len(all_ids), BATCH):
    ids = pl.Series('customer_id', all_ids[i:i + BATCH], dtype=pl.Int32)
    srcs = build_sources(trans, customers, articles, SUB_CUTOFF, 100,
                         customer_filter=ids, build_missing=False)
    cand = combine(srcs, top_k=TOP_K)
    del srcs
    gc.collect()
    ds = F.build_all_features(cand, trans, articles, customers, SUB_CUTOFF, n_chunks=2)
    del cand
    gc.collect()
    ds = add_user2item_similarity(ds, trans, SUB_CUTOFF, N_ITEMS)
    ds = add_bpr_similarity(ds, trans, SUB_CUTOFF, N_ITEMS, dim=BPR_DIM,
                            iterations=BPR_ITERS)
    ds = predict(model, ds, cols)
    top = (ds.sort(['customer_id', 'pred_score', 'article_id'],
                   descending=[False, True, False])
           .with_columns((pl.int_range(pl.len()).over('customer_id') + 1).alias('r'))
           .filter(pl.col('r') <= 12)
           .group_by('customer_id').agg(pl.col('article_id')))
    for row in top.iter_rows(named=True):
        predictions[row['customer_id']] = ' '.join(article_rev[a] for a in row['article_id'])
    del ds, top
    gc.collect()
    print('  {:,}/{:,}  予測済み{:,}'.format(
        min(i + BATCH, len(all_ids)), len(all_ids), len(predictions)), flush=True)

pop = C.load_cached('popular', SUB_CUTOFF, 100)
fallback = ' '.join(article_rev[a] for a in
                    pop.sort('rank').head(12)['article_id'].to_list())

sample = pl.read_csv(DATA_DIR / 'sample_submission.csv')
cust_map = maps['customer_id_map']
rows = [{'customer_id': cid,
         'prediction': predictions.get(cust_map.get(cid, -1), fallback)}
        for cid in sample['customer_id'].to_list()]
sub = pl.DataFrame(rows)
out = SUBMISSION_DIR / 'submission_{}.csv'.format(NAME)
sub.write_csv(out)
n_fb = sum(1 for r in rows if r['prediction'] == fallback)
print('保存: {}'.format(out))
print('  行数: {:,}  フォールバック: {:,} ({:.1f}%)'.format(
    len(sub), n_fb, n_fb / len(sub) * 100))
