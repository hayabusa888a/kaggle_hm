"""user2item 類似度。cutoff ごとに学習し、その週にしか使わない。

なぜ週ごとに学習し直すか（第3部原文）
 - 5位「We use all transaction data to train the word2vec model, which actually leaked
   the future info a little bit ... my map@12 score for last week is 0.0441, but the LB
   is only 0.0350, the gap is much larger than the ones mentioned by others」
 - 10位「when I train a CF-model on all data up to 09-15 and later use this to make
   predictions for the week 09-01, I will boost my CV by implicitly using purchase
   information from the future」「I avoided this by training a full set of models for
   every week」-> CV-LB の線形フィット R²=98.96%

何を作るか
 3位は BPR（implicit）、1位は ProNE、11位/10位は LightFM、5位/13位は word2vec と、
 チームごとに手段は違うが「顧客ベクトルと候補商品ベクトルの類似度」という点は共通。
 ここでは追加依存なしで動く TruncatedSVD（暗黙的フィードバック行列の低ランク近似）で
 同じ量を作る。implicit / gensim が入る環境なら BPR / word2vec に差し替えられるよう、
 embed_articles() の戻り値の形だけ合わせてある。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from .config import FEATURE_DIR


def _user_item_matrix(transactions: pl.DataFrame, cutoff: date, n_items: int,
                      history_days: int | None = None):
    from scipy.sparse import csr_matrix
    df = transactions.filter(pl.col('t_dat') <= pl.lit(cutoff))
    if history_days is not None:
        df = df.filter(pl.col('t_dat') > pl.lit(cutoff - timedelta(days=history_days)))
    ui = df.group_by(['customer_id', 'article_id']).agg(pl.len().alias('n'))
    rows = ui['customer_id'].to_numpy()
    cols = ui['article_id'].to_numpy()
    data = np.log1p(ui['n'].to_numpy()).astype(np.float32)
    n_users = int(rows.max()) + 1
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


def embed_articles(transactions: pl.DataFrame, cutoff: date, n_items: int,
                   dim: int = 32, history_days: int | None = None,
                   seed: int = 42, cache: bool = True) -> np.ndarray:
    """商品ベクトルを (n_items, dim) で返す。cutoff までの取引だけで学習する。

    9位「I have tried sizes 32 and 64 ... I finally use size 32」に合わせ既定は32。
    """
    path = FEATURE_DIR / f'item_emb_{cutoff}_d{dim}.npy'
    if cache and path.exists():
        return np.load(path)
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    uim = _user_item_matrix(transactions, cutoff, n_items, history_days)
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    svd.fit(uim)
    emb = normalize(svd.components_.T.astype(np.float32))   # (n_items, dim)
    np.save(path, emb)
    return emb


def embed_bpr(transactions: pl.DataFrame, cutoff: date, n_items: int,
              dim: int = 64, history_days: int | None = None,
              iterations: int = 100, seed: int = 42,
              cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """BPR行列分解。3位が user2item 類似度に使ったもの。

      "This BPR model is trained with all the transactions before the target week
       (I've trained one BPR for each week) using implicit."
      単体AUC ~0.720（モデル全体 0.806、他の最良単一特徴 0.680）、LB 0.03363 -> 0.03510。

    SVD版（embed_articles）との違いは目的関数。SVDは二乗誤差を最小化するので
    「買っていない = 0」と解釈してしまうが、BPRは「買ったものは買っていないものより
    順位が上」を直接最適化する。購買データは暗黙的フィードバックなので後者が合う。
    実測でも SVD版の単体AUCは 0.6275 しか出ず、3位の 0.720 に遠く及ばなかった。

    返り値: (user_factors, item_factors)
    """
    suffix = f'd{dim}' if iterations == 100 else f'd{dim}_it{iterations}'
    upath = FEATURE_DIR / f'bpr_user_{cutoff}_{suffix}.npy'
    ipath = FEATURE_DIR / f'bpr_item_{cutoff}_{suffix}.npy'
    if cache and upath.exists() and ipath.exists():
        return np.load(upath), np.load(ipath)

    from implicit.bpr import BayesianPersonalizedRanking

    uim = _user_item_matrix(transactions, cutoff, n_items, history_days)
    model = BayesianPersonalizedRanking(
        factors=dim, iterations=iterations, random_state=seed, verify_negative_samples=True)
    model.fit(uim, show_progress=False)
    uf = np.asarray(model.user_factors, dtype=np.float32)
    itf = np.asarray(model.item_factors, dtype=np.float32)
    np.save(upath, uf)
    np.save(ipath, itf)
    return uf, itf


def add_bpr_similarity(candidates: pl.DataFrame, transactions: pl.DataFrame,
                       cutoff: date, n_items: int, dim: int = 64,
                       iterations: int = 100, cache: bool = True) -> pl.DataFrame:
    """候補に bpr_sim（BPRの user factor と item factor の内積）を足す。

    3位はこれを user2item similarity と呼んでいる。内積そのものが BPR の
    スコア（順位づけの根拠）なので、cos ではなく内積を使う。
    """
    uf, itf = embed_bpr(transactions, cutoff, n_items, dim,
                        iterations=iterations, cache=cache)

    c = candidates['customer_id'].to_numpy()
    a = candidates['article_id'].to_numpy()
    sim = np.full(len(c), np.nan, dtype=np.float32)
    nu, ni = uf.shape[0], itf.shape[0]

    step = 1_000_000
    for start in range(0, len(c), step):
        end = min(start + step, len(c))
        cc, aa = c[start:end], a[start:end]
        ok = (cc < nu) & (aa < ni)
        if not ok.any():
            continue
        sim[start:end][ok] = np.einsum('ij,ij->i', uf[cc[ok]], itf[aa[ok]])
    return candidates.with_columns(pl.Series('bpr_sim', sim))


def user_vectors(transactions: pl.DataFrame, cutoff: date, item_emb: np.ndarray,
                 history_days: int = 90) -> tuple[np.ndarray, np.ndarray]:
    """顧客ベクトル = 購入履歴の商品ベクトルの平均。

    3位がコメント欄で word2vec 類似度について
    「if a user has purchased [a1, a2, a3] in the history, "word2vec item2item similarity"
      is the mean/sum/max of [cosine_sim(a1, candidate), cosine_sim(a2, candidate), ...]」
    と説明しているのと同じ発想を、平均ベクトルで一度に取る形。
    9位も「simply averaging the item embedding of a customer transaction history」。

    返り値: (customer_ids, vectors)
    """
    from sklearn.preprocessing import normalize
    df = (transactions
          .filter((pl.col('t_dat') <= pl.lit(cutoff))
                  & (pl.col('t_dat') > pl.lit(cutoff - timedelta(days=history_days))))
          .select(['customer_id', 'article_id']).unique().sort('customer_id'))
    cids = df['customer_id'].to_numpy()
    aids = df['article_id'].to_numpy()
    uniq, inv = np.unique(cids, return_inverse=True)
    acc = np.zeros((len(uniq), item_emb.shape[1]), dtype=np.float32)
    np.add.at(acc, inv, item_emb[aids])
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float32)[:, None]
    return uniq, normalize(acc / np.maximum(counts, 1))


def add_user2item_similarity(candidates: pl.DataFrame, transactions: pl.DataFrame,
                             cutoff: date, n_items: int, dim: int = 32,
                             history_days: int = 90, cache: bool = True) -> pl.DataFrame:
    """候補に u2i_sim（顧客ベクトルと候補商品ベクトルの cos 類似）を足す。

    3位はこの1特徴だけで単体AUC 0.720（モデル全体0.806、他の最良単一特徴0.680）、
    LB 0.03363 -> 0.03510 と報告している。
    """
    item_emb = embed_articles(transactions, cutoff, n_items, dim, cache=cache)
    cids, uvec = user_vectors(transactions, cutoff, item_emb, history_days)

    # customer_id -> uvec の行番号
    pos = np.full(int(cids.max()) + 1, -1, dtype=np.int32)
    pos[cids] = np.arange(len(cids), dtype=np.int32)

    c = candidates['customer_id'].to_numpy()
    a = candidates['article_id'].to_numpy()
    sim = np.full(len(c), np.nan, dtype=np.float32)

    # 700万行 x 32次元の一時配列を2本作るとメモリが飛ぶ（実測でsegfault）。
    # 行スライスで回す。
    step = 1_000_000
    for start in range(0, len(c), step):
        end = min(start + step, len(c))
        cc = c[start:end]
        idx = np.where(cc < len(pos), pos[np.minimum(cc, len(pos) - 1)], -1)
        ok = idx >= 0
        if not ok.any():
            continue
        sim[start:end][ok] = np.einsum(
            'ij,ij->i', uvec[idx[ok]], item_emb[a[start:end][ok]])
    return candidates.with_columns(pl.Series('u2i_sim', sim))
