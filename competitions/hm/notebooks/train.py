import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoost, Pool
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

DATA_DIR = Path('/mnt/c/kaggle_nvidia/kaggle_hm/competitions/hm/data')
OUTPUT_DIR = Path('/mnt/c/kaggle_nvidia/kaggle_hm/competitions/hm/outputs/submissions')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N = 12
a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
feature_cols = ['rank', 'score', 'item_popularity', 'user_category_count', 'days_since_last_purchase']

print("データ読み込み中...")
df = pl.read_csv(
    DATA_DIR / 'raw/transactions_train.csv',
    columns=['t_dat', 'customer_id', 'article_id'],
    schema_overrides={'article_id': pl.String, 'customer_id': pl.String}
).with_columns(pl.col('t_dat').str.to_date())

articles = pl.read_csv(
    DATA_DIR / 'raw/articles.csv',
    columns=['article_id', 'index_group_name'],
    schema_overrides={'article_id': pl.String}
)

customers_full = pl.read_csv(
    DATA_DIR / 'raw/customers.csv',
    schema_overrides={'customer_id': pl.String}
)

last_ts = df['t_dat'].max()
print(f"全データ最終日: {last_ts}, 行数: {len(df):,}")

# 年齢ビン作成
customers_age = customers_full.select(['customer_id', 'age']).with_columns(
    pl.when(pl.col('age').is_null())
    .then(-1)
    .otherwise((pl.col('age') / 10).cast(pl.Int32) * 10)
    .alias('age_bin')
)

def add_ldbw(df, last_date):
    return df.with_columns(
        (pl.col('t_dat') +
         ((pl.lit(last_date) - pl.col('t_dat')).dt.total_days() % 7
         ).cast(pl.Int32) * pl.duration(days=1)
        ).alias('ldbw')
    )

def generate_candidates(df_train, last_ts_fold, fold_valid_y, n_candidates=50):
    df_train = add_ldbw(df_train, last_ts_fold)
    
    weekly_sales = df_train.group_by(['ldbw', 'article_id']).agg(pl.len().alias('count'))
    df_train = df_train.join(weekly_sales, on=['ldbw', 'article_id'], how='left')
    
    last_week_sales = (
        weekly_sales.filter(pl.col('ldbw') == last_ts_fold)
        .select(['article_id', 'count']).rename({'count': 'count_targ'})
    )
    
    df_train = (
        df_train
        .join(last_week_sales, on='article_id', how='left')
        .with_columns(pl.col('count_targ').fill_null(0))
        .with_columns((pl.col('count_targ') / pl.col('count')).alias('quotient'))
        .with_columns(
            (pl.lit(last_ts_fold) - pl.col('t_dat')).dt.total_days().clip(lower_bound=1).alias('days_ago')
        )
        .with_columns(
            (
                pl.col('quotient') * (
                    a / pl.col('days_ago').cast(pl.Float64).sqrt() +
                    b * (-c * pl.col('days_ago').cast(pl.Float64)).exp() - d
                ).clip(lower_bound=0)
            ).alias('score')
        )
    )
    
    purchase_scores = (
        df_train.group_by(['customer_id', 'article_id'])
        .agg(pl.col('score').sum())
        .filter(pl.col('score') > 0)
    )
    
    general_pred = (
        df_train.group_by('article_id')
        .agg(pl.col('quotient').sum().alias('total_quotient'))
        .sort('total_quotient', descending=True)
        .head(n_candidates)
        .select('article_id').to_series().to_list()
    )
    
    candidates = (
        purchase_scores
        .sort(['customer_id', 'score'], descending=[False, True])
        .with_columns(
            pl.col('score').rank(method='ordinal', descending=True).over('customer_id').alias('rank')
        )
        .filter(pl.col('rank') <= n_candidates)
    )
    
    # valid_yユーザーにgeneral_predを追加
    valid_y_users = fold_valid_y['customer_id'].unique().to_list()
    general_candidates = pl.DataFrame({
        'customer_id': np.repeat(valid_y_users, n_candidates),
        'article_id': general_pred * len(valid_y_users),
        'score': [0.0] * (n_candidates * len(valid_y_users)),
        'rank': np.tile(np.arange(1, n_candidates+1, dtype=np.uint32), len(valid_y_users))
    })
    
    candidates = pl.concat([candidates, general_candidates]).unique(subset=['customer_id', 'article_id'])
    
    # bin popular候補を追加
    df_train_with_age = df_train.join(customers_age, on='customer_id', how='left')
    recent = df_train_with_age.filter(pl.col('t_dat') >= last_ts_fold - pl.duration(days=28))
    
    bin_popular_candidates = []
    for age_bin in customers_age['age_bin'].unique().to_list():
        top_n = (
            recent.filter(pl.col('age_bin') == age_bin)
            .group_by('article_id')
            .agg(pl.len().alias('count'))
            .sort('count', descending=True)
            .head(n_candidates)
            .select('article_id').to_series().to_list()
        )
        if len(top_n) == 0:
            continue
        users_in_bin = customers_age.filter(pl.col('age_bin') == age_bin)['customer_id'].to_list()
        n = len(top_n)
        bin_df = pl.DataFrame({
            'customer_id': np.repeat(users_in_bin, n),
            'article_id': top_n * len(users_in_bin),
            'score': [0.0] * (n * len(users_in_bin)),
            'rank': np.tile(np.arange(1, n+1, dtype=np.uint32), len(users_in_bin))
        })
        bin_popular_candidates.append(bin_df)
    
    bin_popular_df = pl.concat(bin_popular_candidates)
    candidates = pl.concat([candidates, bin_popular_df]).unique(subset=['customer_id', 'article_id'])
    
    return candidates, df_train, general_pred

def create_features(candidates, df_train, last_ts_fold):
    recent = df_train.filter(pl.col('t_dat') >= last_ts_fold - pl.duration(days=28))
    
    item_popularity = recent.group_by('article_id').agg(pl.len().alias('item_popularity'))
    
    user_category_count = (
        recent.join(articles, on='article_id', how='left')
        .group_by(['customer_id', 'index_group_name'])
        .agg(pl.len().alias('user_category_count'))
    )
    
    user_last_purchase = (
        df_train.group_by('customer_id')
        .agg(pl.col('t_dat').max().alias('last_purchase_date'))
        .with_columns(
            (pl.lit(last_ts_fold) - pl.col('last_purchase_date')).dt.total_days().alias('days_since_last_purchase')
        )
    )
    
    features = (
        candidates
        .join(item_popularity, on='article_id', how='left')
        .join(articles, on='article_id', how='left')
        .join(user_category_count, on=['customer_id', 'index_group_name'], how='left')
        .join(user_last_purchase.drop('last_purchase_date'), on='customer_id', how='left')
        .with_columns([
            pl.col('item_popularity').fill_null(0),
            pl.col('user_category_count').fill_null(0),
            pl.col('days_since_last_purchase').fill_null(999),
        ])
    )
    return features

# 5fold学習
models_lgb = []
models_catboost = []

for fold in range(5):
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/5")
    print(f"{'='*50}")
    
    fold_valid_end_date = (df.select((pl.lit(last_ts) - pl.duration(days=7*fold)).alias('d')))['d'][0]
    fold_valid_start_date = (df.select((pl.lit(last_ts) - pl.duration(days=7*(fold+1))).alias('d')))['d'][0]
    
    df_fold_train = df.filter(pl.col('t_dat') < fold_valid_start_date)
    df_fold_valid = df.filter(
        (pl.col('t_dat') >= fold_valid_start_date) &
        (pl.col('t_dat') < fold_valid_end_date)
    )
    
    last_ts_fold = df_fold_train['t_dat'].max()
    print(f"train: {df_fold_train['t_dat'].min()} 〜 {last_ts_fold}")
    print(f"valid: {fold_valid_start_date} 〜 {fold_valid_end_date}")
    
    fold_valid_y = (
        df_fold_valid.select(['customer_id', 'article_id']).unique()
        .with_columns(pl.lit(1).alias('label'))
    )
    
    candidates_fold, df_fold_train, _ = generate_candidates(df_fold_train, last_ts_fold, fold_valid_y)
    features_fold = create_features(candidates_fold, df_fold_train, last_ts_fold)
    features_fold = (
        features_fold
        .join(fold_valid_y, on=['customer_id', 'article_id'], how='left')
        .with_columns(pl.col('label').fill_null(0).cast(pl.Int32))
    )
    
    pos = features_fold['label'].sum()
    print(f"正例: {pos:,} / 負例: {len(features_fold)-pos:,} / Recall: {pos/len(fold_valid_y):.4f}")
    
    features_fold_pd = features_fold.select(
        ['customer_id', 'article_id'] + feature_cols + ['label']
    ).to_pandas()
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, valid_idx = next(gss.split(features_fold_pd, groups=features_fold_pd['customer_id']))
    
    train_df_fold = features_fold_pd.iloc[train_idx].sort_values('customer_id')
    valid_df_fold = features_fold_pd.iloc[valid_idx].sort_values('customer_id')
    
    train_data_fold = lgb.Dataset(
        train_df_fold[feature_cols], label=train_df_fold['label'],
        group=train_df_fold.groupby('customer_id').size().values
    )
    valid_data_fold = lgb.Dataset(
        valid_df_fold[feature_cols], label=valid_df_fold['label'],
        group=valid_df_fold.groupby('customer_id').size().values
    )
    
    lgb_model_fold = lgb.train(
        {'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [12],
         'learning_rate': 0.05, 'num_leaves': 31, 'min_data_in_leaf': 20, 'verbose': -1},
        train_data_fold, num_boost_round=200, valid_sets=[valid_data_fold],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(20)]
    )
    models_lgb.append(lgb_model_fold)
    
    train_pool_fold = Pool(data=train_df_fold[feature_cols], label=train_df_fold['label'], group_id=train_df_fold['customer_id'])
    valid_pool_fold = Pool(data=valid_df_fold[feature_cols], label=valid_df_fold['label'], group_id=valid_df_fold['customer_id'])
    
    catboost_model_fold = CatBoost({
        'loss_function': 'YetiRank', 'iterations': 200, 'learning_rate': 0.05,
        'depth': 6, 'verbose': 20, 'early_stopping_rounds': 20,
    })
    catboost_model_fold.fit(train_pool_fold, eval_set=valid_pool_fold)
    models_catboost.append(catboost_model_fold)
    
    print(f"Fold {fold+1} 完了")

print("\n全fold学習完了！提出ファイル作成中...")

# 全データで候補生成
fold_valid_y_dummy = df.select(['customer_id', 'article_id']).head(1).with_columns(pl.lit(1).alias('label'))
candidates_all, df_all, general_pred_all = generate_candidates(df, last_ts, fold_valid_y_dummy)
features_all = create_features(candidates_all, df_all, last_ts)
features_all_pd = features_all.select(['customer_id', 'article_id'] + feature_cols).to_pandas()

# アンサンブル予測
lgb_scores = np.mean([m.predict(features_all_pd[feature_cols]) for m in models_lgb], axis=0)
catboost_scores = np.mean([m.predict(features_all_pd[feature_cols]) for m in models_catboost], axis=0)

scaler = MinMaxScaler()
lgb_scores_norm = scaler.fit_transform(lgb_scores.reshape(-1, 1)).flatten()
catboost_scores_norm = scaler.fit_transform(catboost_scores.reshape(-1, 1)).flatten()
features_all_pd['pred_score'] = (lgb_scores_norm + catboost_scores_norm) / 2

submission_rerank = (
    features_all_pd.sort_values(['customer_id', 'pred_score'], ascending=[True, False])
    .groupby('customer_id').head(12)
)

submission_final = (
    submission_rerank.sort_values(['customer_id', 'pred_score'], ascending=[True, False])
    .groupby('customer_id')['article_id']
    .apply(lambda x: ' '.join(x.astype(str)))
    .reset_index()
    .rename(columns={'article_id': 'prediction'})
)

general_pred_str = ' '.join(general_pred_all)
customers = pl.read_csv(
    DATA_DIR / 'raw/customers.csv',
    columns=['customer_id'],
    schema_overrides={'customer_id': pl.String}
).to_pandas()

submission_full = customers.merge(submission_final, on='customer_id', how='left')
submission_full['prediction'] = submission_full['prediction'].fillna(general_pred_str)
submission_full.to_csv(OUTPUT_DIR / 'submission_binpopular_ensemble.csv', index=False)

print(f"保存完了: {len(submission_full):,}ユーザー")
print(submission_full.head(5))
