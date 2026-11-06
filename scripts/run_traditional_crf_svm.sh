#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/run_traditional_crf_svm.sh <dataset_config> [traditional_crf_svm.py args...]" >&2
    exit 2
fi

CONFIG="$1"
shift
EXTRA_ARGS=("$@")
PYTHON="${PYTHON:-python}"

DS_NAME=$("$PYTHON" -c "from src.common.utils import load_yaml; print(load_yaml('$CONFIG')['name'])")
PDIR=$("$PYTHON" -c "from src.common.utils import load_yaml; print(load_yaml('$CONFIG')['processed_dir'])")
RUN_NAME="${RUN_NAME:-crf_svm}"
RUN_ROOT="outputs/$DS_NAME/$RUN_NAME"
mkdir -p "$RUN_ROOT/predictions" "$RUN_ROOT/reports"

"$PYTHON" scripts/traditional_crf_svm.py \
    --dataset_config "$CONFIG" \
    --run_name "$RUN_NAME" \
    --out_root outputs \
    --model_root outputs \
    --splits id_eval eval \
    "${EXTRA_ARGS[@]}"

"$PYTHON" -m src.evaluation.evaluate \
    --gold "$PDIR/id_eval.jsonl" \
    --pred "$RUN_ROOT/predictions/id_eval.jsonl" \
    --out_json "$RUN_ROOT/reports/id_eval.json" \
    --out_md "$RUN_ROOT/reports/id_eval.md" \
    --title "${RUN_NAME} (ID Eval)"

GROUPS_ARG=()
[[ -f "$PDIR/eval_groups.json" ]] && GROUPS_ARG=(--groups "$PDIR/eval_groups.json")
"$PYTHON" -m src.evaluation.evaluate \
    --gold "$PDIR/eval.jsonl" \
    --pred "$RUN_ROOT/predictions/eval.jsonl" \
    "${GROUPS_ARG[@]}" \
    --id_eval_report "$RUN_ROOT/reports/id_eval.json" \
    --out_json "$RUN_ROOT/reports/eval.json" \
    --out_md "$RUN_ROOT/reports/eval.md" \
    --title "${RUN_NAME} (OOD Eval)"

echo "=== done: $RUN_NAME, report: $RUN_ROOT/reports/eval.md ==="
