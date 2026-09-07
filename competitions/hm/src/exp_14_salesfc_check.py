"""実験14の事前確認: 来週売上予測が「失速する商品」を検出できているか。

10位の例: 日次売上が 0 0 28 431 389 105 32 11 のような商品は、直近人気で見ると
過大評価するが、予測モデルは減衰を学習する。これが実際に起きているか目視する。
"""
from datetime import timedelta
import polars as pl

from hm.config import VALID_CUTOFF, load_converted
from hm.exp_log import timer
from hm.sales_forecast import predict_next_week_sales

CUTOFF = VALID_CUTOFF
trans, customers, articles = load_converted()

with timer() as t:
    fc = predict_next_week_sales(trans, CUTOFF)
print('予測対象商品: {:,}  ({:.0f}s)'.format(len(fc), t()))

# 実際の翌週売上と突き合わせる（評価目的のみ。特徴量には使わない）
actual = (trans.filter((pl.col('t_dat') > pl.lit(CUTOFF))
                       & (pl.col('t_dat') <= pl.lit(CUTOFF + timedelta(days=7))))
          .group_by('article_id').agg(pl.len().alias('actual')))
recent = (trans.filter((pl.col('t_dat') > pl.lit(CUTOFF - timedelta(days=7)))
                       & (pl.col('t_dat') <= pl.lit(CUTOFF)))
          .group_by('article_id').agg(pl.len().alias('last7d')))

df = (fc.join(actual, on='article_id', how='left').with_columns(pl.col('actual').fill_null(0))
      .join(recent, on='article_id', how='left').with_columns(pl.col('last7d').fill_null(0)))

import numpy as np
a = df['actual'].to_numpy().astype(float)
p = df['a_pred_next_sales'].to_numpy().astype(float)
r = df['last7d'].to_numpy().astype(float)
print('\n=== 翌週実売との相関（対数空間） ===')
print('  予測モデル      : {:.4f}'.format(np.corrcoef(np.log1p(p), np.log1p(a))[0, 1]))
print('  直近7日実績のみ : {:.4f}'.format(np.corrcoef(np.log1p(r), np.log1p(a))[0, 1]))

print('\n=== 直近7日が多いのに翌週落ちる商品(失速)を検出できているか ===')
hot = df.filter(pl.col('last7d') >= 50)
hot = hot.with_columns([
    (pl.col('actual') / pl.col('last7d')).alias('actual_ratio'),
    (pl.col('a_pred_next_sales') / pl.col('last7d')).alias('pred_ratio')])
print('  対象商品(直近7日50件以上): {:,}'.format(len(hot)))
print('  実績比と予測比の相関: {:.4f}'.format(
    np.corrcoef(hot['actual_ratio'].to_numpy(), hot['pred_ratio'].to_numpy())[0, 1]))
worst = hot.sort('actual_ratio').head(8)
print('\n  実際に最も失速した商品 上位8件:')
print('  {:>10}{:>9}{:>9}{:>10}{:>10}'.format('article', 'last7d', '実翌週', '実績比', '予測比'))
for row in worst.iter_rows(named=True):
    print('  {:>10}{:>9}{:>9}{:>10.2f}{:>10.2f}'.format(
        row['article_id'], int(row['last7d']), int(row['actual']),
        row['actual_ratio'], row['pred_ratio']))
