#!/bin/bash
# u2tag2i を組み込んだ13戦略で、valid + 学習6週のデータセットを作り直して評価する。
# exp_19 は valid(2020-09-15) から6週遡るので、必要な cutoff は 09-08 〜 08-04。
set -u
cd /c/kaggle_nvidia/kaggle_hm/competitions/hm/src
export PYTHONIOENCODING=utf-8
CUTOFFS="2020-09-15 2020-09-08 2020-09-01 2020-08-25 2020-08-18 2020-08-11 2020-08-04"

echo "===== u2tag 候補の生成 $(date '+%H:%M:%S') ====="
for C in $CUTOFFS; do
  for T in u2tag_section_no u2tag_department_no u2tag_product_type_no; do
    python -u build_candidates.py $C $T
  done
done

echo ""
echo "===== k200 データセットの再構築（メタ特徴が6列増える） $(date '+%H:%M:%S') ====="
# 既存の k200 キャッシュは10戦略ぶんなので退避してから作り直す
mkdir -p ../outputs/features/_pre_u2tag
mv ../outputs/features/dataset_*_k200_*.parquet ../outputs/features/_pre_u2tag/ 2>/dev/null
python -u build_k200.py 2020-09-15 valid
for C in 2020-09-08 2020-09-01 2020-08-25 2020-08-18 2020-08-11 2020-08-04; do
  python -u build_k200.py $C train_neg30
done

echo ""
echo "===== 評価（最良構成・学習6週・13戦略） $(date '+%H:%M:%S') ====="
python -u exp_19_final_metrics.py

echo "U2TAG PIPELINE DONE $(date '+%H:%M:%S')"
