# H&M 二段構成レコメンダ

`hm/` は候補生成（第1段）とリランキング（第2段）の共通ライブラリ。
`exp_*.py` は1施策=1スクリプトの実験。数値はすべて
`outputs/experiments/experiment_log.csv` に追記される。

## 評価の定義（全実験で共通）

| 段 | 指標 | 関数 |
|---|---|---|
| 第1段 候補生成 | recall@K | `hm.recall_at_k` / `hm.evaluate_candidates` |
| 第2段 リランキング | MAP@12 | `hm.map_at_12` / `hm.evaluate_ranking` |
| 補助 | HitNum@K | `hm.hits_at_k`（1位が週ごとの実測値を公開しており突き合わせられる） |

**分母は既定で `all_valid`**（正解を持つ全顧客）。`covered`（候補がある顧客だけ）は
候補を絞るほど勝手に上がるので、意思決定に使わないこと。

validation は9位に合わせ最終週 holdout。`VALID_CUTOFF = 2020-09-15`
（正解週 = 09-16〜09-22）、提出用は `SUB_CUTOFF = 2020-09-22`。

## リーク規律

週 W の特徴量を作るモデルは W より前のデータだけで学習する。
`candidates.py` / `features.py` / `embeddings.py` のどの関数も
`t_dat <= cutoff` で切ってから計算しており、cutoff より後を参照する経路はない。

根拠は10位の原文（週ごとに学習し直して CV-LB 線形フィット R²=98.96%）と、
5位の失敗報告（全期間 word2vec で CV 0.0441 に対し LB 0.0350）。

## 実行の仕方

```bash
cd competitions/hm/src

# 候補生成（1戦略ずつ別プロセス。メモリが返るのでこれが確実）
python build_candidates.py 2020-09-15
python build_candidates.py 2020-09-08

# 第1段の評価
python exp_04_pool.py

# 第2段（メタ特徴の有無 x categorical指定の有無）
python exp_05_rerank.py 100     # 引数は1顧客あたりの候補数 top_k

# user2item 類似度の効果
python exp_07_u2i.py 100
```

### 実行時の注意

- **同じスクリプトを二重に起動しないこと。** ホストのメモリは並走に耐えない。
  Git Bash の `ps` は Windows プロセスを取りこぼすので、確認は PowerShell 側で行う:
  `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select ProcessId, CommandLine`
- 出力は必ずファイルにリダイレクトする（`| tail` を挟むとバッファされて何も見えない）

## 候補生成の戦略（cutoff=2020-09-15 の実測）

| 戦略 | 出典 | 候補/人 | 被覆% | R@100 | MAP@12 |
|---|---|---|---|---|---|
| popular_by_age | 9位/8位 | 100.0 | 100.0 | 0.1267 | 0.0094 |
| user_cf | 8位 | 95.3 | 73.2 | 0.1210 | 0.0229 |
| popular | 8位 | 100.0 | 100.0 | 0.1181 | 0.0088 |
| popular_by_channel | 8位 | 100.0 | 91.9 | 0.1120 | 0.0078 |
| repurchase | 8位 | 44.3 | 91.9 | 0.0494 | 0.0234 |
| timedecay | 9位 Trending | 22.0 | 89.6 | 0.0489 | 0.0250 |
| same_product_code | 8位 | 73.6 | 89.2 | 0.0452 | 0.0069 |
| item2item_cf | 6位/11位 | 79.7 | 45.5 | 0.0443 | 0.0073 |
| also_bought | 5位 | 56.0 | 43.9 | 0.0304 | 0.0060 |
| purchase_interval | 独自 | 5.0 | 54.5 | 0.0068 | 0.0051 |

結合後（10戦略・320.9件/人）: recall@100 = 0.1627 / @200 = 0.2226 / @300 = 0.2541、
K無制限の上限 = 0.2689。

## 既知の未処理

- `purchase_interval` は polars 移植でスコア定義がずれ R@100 0.0119 -> 0.0068 に劣化。
  leave-one-out の寄与は -0.0002 なので放置しているが、直すか捨てるかは未決。
- `embeddings.py` は BPR ではなく TruncatedSVD。`implicit` が入る環境なら
  `embed_articles()` の中身を差し替えれば他はそのまま動く。
