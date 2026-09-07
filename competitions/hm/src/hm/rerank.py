"""第2段: リランキング。LightGBM binary classifier から始める。

モデル選択の根拠（第3部原文）
 - 6位「Because the cv and lb score of binary classifier is much better than ranking
   model in our case」best single = catboost binary, cv 0.0403
 - 9位「LightGBM classifier is used as a ranking model. There is no significant
   difference between classifier and lambda ranker... the classifier will give a
   slightly higher performance in general」
 - 一方 2位/8位/13位は ranker。原文で結論が割れているので、まず binary を基準にし、
   lambdarank との比較は同じ候補・同じ特徴で後から行う。

サンプリングの根拠
 - 3位「The pos samples are only the ones which are recalled in the candidates and the
   other items that the user purchased in actual are not included」
   「I just keep the negative samples with amount of 30*len(pos_samples)」
 - 9位「keeping all positive samples and random choosing half of the negative samples」
"""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl


def attach_labels(candidates: pl.DataFrame, actuals: dict[int, set[int]]) -> pl.DataFrame:
    """候補に正解ラベルを付ける。

    3位に従い、正例は「候補に入った実購入品のみ」。候補外の実購入は学習データに入れない
    （候補生成が拾えなかったものをランカーに押し付けても学習できないため）。
    """
    from .metrics import actuals_to_frame
    truth = actuals_to_frame(actuals)
    return (candidates.join(truth, on=['customer_id', 'article_id'], how='left')
            .with_columns(pl.col('label').fill_null(0).cast(pl.Int8)))


def downsample_negatives(labeled: pl.DataFrame, ratio: float = 30.0,
                         seed: int = 42) -> pl.DataFrame:
    """正例:負例 = 1:ratio になるよう負例をランダムに間引く（3位は30倍）。

    正例を1つも持たない顧客の行も、その顧客ぶんの負例として ratio 相当だけ残す。
    """
    rng = np.random.default_rng(seed)
    pos = labeled.filter(pl.col('label') == 1)
    neg = labeled.filter(pl.col('label') == 0)
    n_keep = min(len(neg), int(len(pos) * ratio))
    idx = rng.choice(len(neg), size=n_keep, replace=False)
    return pl.concat([pos, neg[idx]], how='vertical')


def downsample_per_customer(labeled: pl.DataFrame, frac: float = 0.5,
                            seed: int = 42) -> pl.DataFrame:
    """9位方式: 正例は全部残し、負例を顧客ごとに frac だけ残す。"""
    return pl.concat([
        labeled.filter(pl.col('label') == 1),
        (labeled.filter(pl.col('label') == 0)
         .with_columns(pl.lit(1).sample(fraction=1.0, shuffle=True, seed=seed).alias('_'))
         .with_columns((pl.int_range(pl.len()).over('customer_id')
                        / pl.len().over('customer_id')).alias('_q'))
         .filter(pl.col('_q') < frac).drop(['_', '_q'])),
    ], how='vertical')


def train_lgb_binary(train: pl.DataFrame, feature_cols: list[str],
                     categorical: list[str] | None = None,
                     params: dict | None = None, num_boost_round: int = 300,
                     seed: int = 42):
    """LightGBM binary classifier。

    categorical を明示指定するのは6位の
      "set categorical_features(department_no, product_type_no, etc.) for lightgbm
       will improve cv score about 0.0005~0.0008"
    に従ったもの。

    polars -> numpy で直接渡す。to_pandas() は pyarrow を要求するうえ、
    700万行規模だと変換コピーだけで数GB積む。
    """
    import lightgbm as lgb

    categorical = [c for c in (categorical or []) if c in feature_cols]
    p = {
        'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
        'num_leaves': 63, 'min_data_in_leaf': 100, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 1, 'verbosity': -1,
        'num_threads': 0, 'seed': seed,
    }
    p.update(params or {})
    x = to_matrix(train, feature_cols)
    ds = lgb.Dataset(x, label=train['label'].to_numpy(), feature_name=feature_cols,
                     categorical_feature=categorical or [], free_raw_data=True)
    return lgb.train(p, ds, num_boost_round=num_boost_round), categorical


def to_matrix(df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """特徴量を float32 の2次元配列にする。欠損は NaN のまま（LightGBMが扱う）。"""
    return df.select([pl.col(c).cast(pl.Float32) for c in feature_cols]).to_numpy()


def predict(model, df: pl.DataFrame, feature_cols: list[str],
            categorical: list[str] | None = None,
            chunk_rows: int = 1_000_000) -> pl.DataFrame:
    """スコアを付ける。

    valid は 700万行規模なので、一度に配列化すると数GB積む。行スライスで回す。
    """
    import gc
    out = np.empty(len(df), dtype=np.float64)
    for start in range(0, len(df), chunk_rows):
        end = min(start + chunk_rows, len(df))
        x = to_matrix(df[start:end], feature_cols)
        out[start:end] = model.predict(x)
        del x
        gc.collect()
    return df.with_columns(pl.Series('pred_score', out))
