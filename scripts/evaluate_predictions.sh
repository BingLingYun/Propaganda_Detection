#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 2 ]]; then
    echo "Usage: bash scripts/evaluate_predictions.sh <dataset_config> <run_name> [run_name ...] [--collect]" >&2
    exit 2
fi

CONFIG="$1"
shift

COLLECT=0
RUN_NAMES=()
for arg in "$@"; do
    if [[ "$arg" == "--collect" ]]; then
        COLLECT=1
    else
        RUN_NAMES+=("$arg")
    fi
done

if [[ ${#RUN_NAMES[@]} -eq 0 ]]; then
    echo "No run_name provided." >&2
    exit 2
fi

PYTHON="${PYTHON:-python}"
PDIR=$("$PYTHON" -c "from src.common.utils import load_yaml; print(load_yaml('$CONFIG')['processed_dir'])")
DS_NAME=$("$PYTHON" -c "from src.common.utils import load_yaml; print(load_yaml('$CONFIG')['name'])")
OUTPUT_BASE="outputs/$DS_NAME"

for RUN_NAME in "${RUN_NAMES[@]}"; do
    RUN_ROOT="$OUTPUT_BASE/$RUN_NAME"
    REPORT_DIR="$RUN_ROOT/reports"
    mkdir -p "$REPORT_DIR"
    ID_PRED="$RUN_ROOT/predictions/id_eval.jsonl"
    EVAL_PRED="$RUN_ROOT/predictions/eval.jsonl"
    if [[ ! -f "$ID_PRED" ]]; then
        echo "Missing prediction file: $ID_PRED" >&2
        exit 1
    fi
    if [[ ! -f "$EVAL_PRED" ]]; then
        echo "Missing prediction file: $EVAL_PRED" >&2
        exit 1
    fi

    echo "=== evaluate: $RUN_NAME id_eval ==="
    "$PYTHON" -m src.evaluation.evaluate \
        --gold "$PDIR/id_eval.jsonl" \
        --pred "$ID_PRED" \
        --out_json "$REPORT_DIR/id_eval.json" \
        --out_md "$REPORT_DIR/id_eval.md" \
        --title "${RUN_NAME} (ID Eval)"

    GROUPS_ARG=()
    if [[ -f "$PDIR/eval_groups.json" ]]; then
        GROUPS_ARG=(--groups "$PDIR/eval_groups.json")
    fi

    echo "=== evaluate: $RUN_NAME eval ==="
    "$PYTHON" -m src.evaluation.evaluate \
        --gold "$PDIR/eval.jsonl" \
        --pred "$EVAL_PRED" \
        "${GROUPS_ARG[@]}" \
        --id_eval_report "$REPORT_DIR/id_eval.json" \
        --out_json "$REPORT_DIR/eval.json" \
        --out_md "$REPORT_DIR/eval.md" \
        --title "${RUN_NAME} (OOD Eval)"
done

if [[ "$COLLECT" == "1" ]]; then
    "$PYTHON" scripts/collect_results.py --reports_dir "$OUTPUT_BASE" --out_prefix summary_results
fi
