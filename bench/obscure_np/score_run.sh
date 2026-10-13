#!/usr/bin/env bash
# Score every candidate on one finished eval run (Kaggle/Colab output dir) in one go:
# LB variants (base2, c2fix), gate variants (gentle = jc-gate), and the learned rerank (jc-lr2's model) on the gentle list.
#   bash score_run.sh RUN_DIR SOLUTION.csv TEST_DRAW.parquet REPLAY_LIB POOL_POP.parquet LR_MODEL.txt OUT_DIR
set -euo pipefail
RUN=$1; SOL=$2; TEST=$3; LIB=$4; POP=$5; MODEL=$6; OUT=$7
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
python "$H/gate_variants.py" --run "$RUN" --out "$OUT/gate"
python "$H/replay_rerank.py" --run "$RUN" --lib "$LIB" --out "$OUT/rp" --v4 gentle="$OUT/gate/v4_gentle.csv"
python "$H/rerank_features.py" --run "$RUN" --main "$OUT/gate/v4_gentle.csv" --lib "$LIB" --pool-pop "$POP" \
    --solution "$SOL" --out "$OUT/feats.parquet"
python "$H/apply_lr.py" --feats "$OUT/feats.parquet" --model "$MODEL" --out "$OUT/lr2.csv"
SUBS=()
for f in "$RUN"/final_base2.csv "$OUT"/rp/rp_base2.csv "$OUT"/rp/rp_c2fix.csv "$OUT"/rp/rp_v4gentle.csv "$OUT"/lr2.csv; do
    [ -f "$f" ] && SUBS+=("$(basename "$f" .csv)=$f")
done
python "$H/mrr_eval.py" --solution "$SOL" --test "$TEST" --sub "${SUBS[@]}" --baseline rp_base2 | tee "$OUT/score.txt"
