# 引き継ぎ：H&M 二段構成レコメンダ「勉強用」セッション

このファイルを新しい Claude Code セッションの冒頭に貼るか、
「competitions/hm/docs/handoff_study.md を読んで」と指示すれば文脈が繋がる。

## このセッションの役割

**手法の理解が目的。スコア改善の作業はしない**（そちらは別セッションで進行中）。
実装済みのコードを教材として、なぜその手法が効くのか／効かないのかを掘る。

コードを書き換えたり、重い実験を回したりする必要はない。
理解のために小さく動かして確かめるのは歓迎。

## 前提となる成果（すでに完了している）

Kaggle「H&M Personalized Fashion Recommendations」に late submission で参加。
候補生成 → リランキングの二段構成を実装し、以下を達成した。

| | public | private |
|---|---|---|
| 起点（別ロジック） | 0.03127 | 0.03121 |
| 最終 | **0.03518** | **0.03511** |

private 0.03511 は上位13解法の 8位 kazuki（0.03476）を上回り、6位（0.03529）に迫る水準。

実装記録（読み物としてまとまっている）:
https://claude.ai/code/artifact/5db562fe-0fe0-4a70-be57-33f7541aa8f1

## 読むべき場所

| 対象 | パス |
|---|---|
| 共通ライブラリ | `competitions/hm/src/hm/` |
| 実験スクリプト（1施策=1本） | `competitions/hm/src/exp_*.py` |
| 使い方・設計方針 | `competitions/hm/src/README.md` |
| 全実験の数値（75行） | `competitions/hm/outputs/experiments/experiment_log.csv` |
| K別の最終指標 | `competitions/hm/outputs/experiments/final_metrics.json` |
| 上位13解法の一次資料 | ユーザーが持っている `hm_top13_solutions.md`（第3部が原文全文） |

`src/hm/` の各モジュールには、**どの解法のどの記述を根拠にしたか**を
docstring に原文の引用つきで書いてある。教材としてはここが中心。

| モジュール | 中身 |
|---|---|
| `metrics.py` | recall@K / MAP@K / hits@K。分母の取り方の議論もコメントにある |
| `candidates.py` | 候補生成10戦略。8位の user based CF はコメント欄の手順そのまま |
| `combine.py` | 3位の `in_<戦略>` / `rank_<戦略>` メタ特徴 |
| `embeddings.py` | 3位の BPR user2item。SVD版との違いもコメントに残してある |
| `features.py` | 102特徴量。6位の値引き率、9位の購入者平均年齢など |
| `rerank.py` | 3位の負例30倍ダウンサンプリング、LightGBM binary |
| `sales_forecast.py` | 10位の来週売上予測（実装したが効かなかった） |
| `pipeline.py` | cutoff を渡すと候補〜学習データまで組み立てる |

## 掘ると面白い論点（実測で答えが出ているもの）

1. **なぜ BPR は効いて SVD は効かなかったのか**
   BPR +0.00304 に対し SVD +0.00045。単体AUC も 0.684 vs 0.628。
   暗黙的フィードバックに対する目的関数の違い（ペア順位 vs 二乗誤差）。

2. **なぜ「候補数を増やすだけ」だと CV が悪化するのか**
   3位が原文で警告している現象。実際に1本目の提出で再現した
   （候補プール拡張直後は起点を下回った）。メタ特徴とセットで初めて効く。

3. **原文の報告が再現しなかった2件**
   - 6位の categorical_features 明示（報告 +0.0005〜0.0008 → 実測 −0.00043）
   - 10位の来週売上予測（「最も使われた特徴の1つ」→ 実測 −0.00009）
   後者は予測性能自体は良い（相関 0.9399）のに効かない。理由は入力の重複。

4. **単体AUC と MAP@12 が逆相関した件**
   BPR の反復数を増やすと AUC は下がるのに MAP は上がった。
   代理指標を目標にすることの危うさ。

5. **リーク規律と CV-LB 写像の関係**
   週ごとに全モデルを学習し直した結果、換算率が 4提出で 0.831±0.002 に収束。
   10位の R²=98.96% と同じ状態。対照的に 5位は全期間 word2vec で大きく乖離。

6. **取りこぼしの構造**
   正解 213,728 ペアのうち、72.0% が「商品はプールにあるのに、その顧客に
   紐付かなかった」ケース。商品の網羅ではなく顧客×商品の紐付けが弱点。

## 環境メモ

- 実験はホストの Python（polars / LightGBM / implicit / optuna）で動く。GPU 不要
- Git Bash の `ps` は Windows プロセスを取りこぼす。確認は PowerShell の
  `Get-CimInstance Win32_Process -Filter "Name='python.exe'"`
- 出力は必ずファイルにリダイレクトする（`| tail` を挟むとバッファされて見えない）
- ホストに pyarrow が無いので `polars.to_pandas()` は落ちる。`to_numpy()` を使う
