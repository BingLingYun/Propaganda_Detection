#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 2 ]]; then
    echo "Usage: bash scripts/run_single.sh <dataset_config> <model_config> [extra train.py args...]" >&2
    exit 2
fi

CONFIG="$1"
MODEL_CONFIG="$2"
shift 2
EXTRA_ARGS=("$@")
PYTHON="${PYTHON:-python}"
SEED="${SEED:-42}"
PRED_BATCH="${PRED_BATCH:-16}"
EXPLICIT_RUN_NAME=""
DEVICE_ARG="${DEVICE:-auto}"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
    if [[ "${EXTRA_ARGS[$i]}" == "--run_name" && $((i + 1)) -lt ${#EXTRA_ARGS[@]} ]]; then
        EXPLICIT_RUN_NAME="${EXTRA_ARGS[$((i + 1))]}"
    elif [[ "${EXTRA_ARGS[$i]}" == --run_name=* ]]; then
        EXPLICIT_RUN_NAME="${EXTRA_ARGS[$i]#--run_name=}"
    elif [[ "${EXTRA_ARGS[$i]}" == "--device" && $((i + 1)) -lt ${#EXTRA_ARGS[@]} ]]; then
        DEVICE_ARG="${EXTRA_ARGS[$((i + 1))]}"
    elif [[ "${EXTRA_ARGS[$i]}" == --device=* ]]; then
        DEVICE_ARG="${EXTRA_ARGS[$i]#--device=}"
    fi
done

DS_NAME=$("$PYTHON" -c "from src.common.utils import load_yaml; print(load_yaml('$CONFIG')['name'])")
PDIR=$("$PYTHON" -c "from src.common.utils import load_yaml; print(load_yaml('$CONFIG')['processed_dir'])")
if [[ -n "$EXPLICIT_RUN_NAME" ]]; then
    RUN_NAME="$EXPLICIT_RUN_NAME"
else
    RUN_NAME=$("$PYTHON" scripts/resolve_run_name.py --dataset_config "$CONFIG" --model_config "$MODEL_CONFIG" --seed "$SEED")
fi

echo "=== run: $RUN_NAME ==="
RUN_ROOT="outputs/$DS_NAME/$RUN_NAME"
mkdir -p "$RUN_ROOT/predictions" "$RUN_ROOT/reports"

"$PYTHON" scripts/train.py --dataset_config "$CONFIG" --model_config "$MODEL_CONFIG" \
    --seed "$SEED" --out_root outputs "${EXTRA_ARGS[@]}"

CKPT="$RUN_ROOT/checkpoints/best"

"$PYTHON" scripts/predict.py --checkpoint "$CKPT" --input "$PDIR/id_eval.jsonl" --split id_eval --out_dir "$RUN_ROOT/predictions" --device "$DEVICE_ARG" --batch_size "$PRED_BATCH"
"$PYTHON" scripts/predict.py --checkpoint "$CKPT" --input "$PDIR/eval.jsonl" --split eval --out_dir "$RUN_ROOT/predictions" --device "$DEVICE_ARG" --batch_size "$PRED_BATCH"

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
