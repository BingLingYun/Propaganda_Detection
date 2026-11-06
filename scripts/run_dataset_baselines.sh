#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/run_dataset_baselines.sh <dataset_config>" >&2
    echo "" >&2
    echo "Common env overrides:" >&2
    echo "  DEVICE=npu MODEL_NAME=xlm-roberta-base SEED=42 EPOCHS=20 PATIENCE=8 USE_FP16=1" >&2
    echo "  RUN_TRIVIAL=1 RUN_TRADITIONAL=1 RUN_MH=1 RUN_TWO_STAGE=1" >&2
    echo "  TRIVIAL_METHODS=\"majority\"" >&2
    echo "  MH_BATCH=4 MH_ACCUM=2 SPAN_BATCH=4 SPAN_ACCUM=2 STAGE2_BATCH=4 STAGE2_ACCUM=2 PRED_BATCH=4" >&2
    exit 2
fi

CONFIG="$1"
PYTHON="${PYTHON:-python}"
export PYTHON
DEVICE="${DEVICE:-npu}"
MODEL_NAME="${MODEL_NAME:-xlm-roberta-base}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
STRIDE="${STRIDE:-128}"
USE_FP16="${USE_FP16:-1}"
DEV_SAMPLE="${DEV_SAMPLE:-}"

RUN_TRIVIAL="${RUN_TRIVIAL:-1}"
RUN_TRADITIONAL="${RUN_TRADITIONAL:-1}"
RUN_MH="${RUN_MH:-1}"
RUN_TWO_STAGE="${RUN_TWO_STAGE:-1}"
TRIVIAL_METHODS="${TRIVIAL_METHODS:-majority}"

MH_BATCH="${MH_BATCH:-4}"
MH_ACCUM="${MH_ACCUM:-2}"
SPAN_BATCH="${SPAN_BATCH:-4}"
SPAN_ACCUM="${SPAN_ACCUM:-2}"
STAGE2_BATCH="${STAGE2_BATCH:-$SPAN_BATCH}"
STAGE2_ACCUM="${STAGE2_ACCUM:-$SPAN_ACCUM}"
PRED_BATCH="${PRED_BATCH:-4}"
CANDIDATE_CHUNK_SIZE="${CANDIDATE_CHUNK_SIZE:-8192}"
TARGET_RECALL="${TARGET_RECALL:-0.95}"
TRAD_CONTEXT_CHARS="${TRAD_CONTEXT_CHARS:-160}"
TRAD_CRF_MAX_ITERATIONS="${TRAD_CRF_MAX_ITERATIONS:-100}"
TRAD_CRF_C1="${TRAD_CRF_C1:-0.1}"
TRAD_CRF_C2="${TRAD_CRF_C2:-0.1}"
TRAD_CRF_VERBOSE="${TRAD_CRF_VERBOSE:-1}"
TRAD_MAX_TRAIN_DOCS="${TRAD_MAX_TRAIN_DOCS:-}"
TRAD_MAX_SVM_EXAMPLES="${TRAD_MAX_SVM_EXAMPLES:-}"
TRAD_SVM_BOUNDARY_PERTURB_RADIUS="${TRAD_SVM_BOUNDARY_PERTURB_RADIUS:-0}"
TRAD_SVM_OVERLAP_THRESHOLD="${TRAD_SVM_OVERLAP_THRESHOLD:-0.5}"
TRAD_SVM_OVERLAP_MODE="${TRAD_SVM_OVERLAP_MODE:-gold_coverage}"
TRAD_SVM_THRESHOLD="${TRAD_SVM_THRESHOLD:-0.0}"

DS_NAME=$("$PYTHON" -c "from src.common.utils import load_yaml; print(load_yaml('$CONFIG')['name'])")
PDIR=$("$PYTHON" -c "from src.common.utils import load_yaml; print(load_yaml('$CONFIG')['processed_dir'])")
OUTPUT_BASE="outputs/$DS_NAME"
LOG_BASE="logs/$DS_NAME"
mkdir -p "$OUTPUT_BASE" "$LOG_BASE"

MH_MODEL_CONFIG="${MH_MODEL_CONFIG:-configs/models/xlm_mhbio_crf.yaml}"

FP16_ARGS=()
if [[ "$USE_FP16" == "1" ]]; then
    FP16_ARGS=(--fp16)
elif [[ "$USE_FP16" == "0" ]]; then
    FP16_ARGS=(--no_fp16)
else
    echo "USE_FP16 must be 0 or 1, got: $USE_FP16" >&2
    exit 2
fi

DEV_SAMPLE_ARGS=()
if [[ -n "$DEV_SAMPLE" ]]; then
    DEV_SAMPLE_ARGS=(--dev_sample "$DEV_SAMPLE")
fi

TRADITIONAL_ARGS=(
    --context_chars "$TRAD_CONTEXT_CHARS"
    --crf_max_iterations "$TRAD_CRF_MAX_ITERATIONS"
    --crf_c1 "$TRAD_CRF_C1"
    --crf_c2 "$TRAD_CRF_C2"
    --svm_boundary_perturb_radius "$TRAD_SVM_BOUNDARY_PERTURB_RADIUS"
    --svm_overlap_threshold "$TRAD_SVM_OVERLAP_THRESHOLD"
    --svm_overlap_mode "$TRAD_SVM_OVERLAP_MODE"
    --svm_threshold "$TRAD_SVM_THRESHOLD"
)
if [[ "$TRAD_CRF_VERBOSE" == "1" ]]; then
    TRADITIONAL_ARGS+=(--crf_verbose)
fi
if [[ -n "$TRAD_MAX_TRAIN_DOCS" ]]; then
    TRADITIONAL_ARGS+=(--max_train_docs "$TRAD_MAX_TRAIN_DOCS")
fi
if [[ -n "$TRAD_MAX_SVM_EXAMPLES" ]]; then
    TRADITIONAL_ARGS+=(--max_svm_examples "$TRAD_MAX_SVM_EXAMPLES")
fi

COMMON_TRAIN_ARGS=(
    --device "$DEVICE"
    --model_name "$MODEL_NAME"
    --max_length "$MAX_LENGTH"
    --stride "$STRIDE"
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    "${DEV_SAMPLE_ARGS[@]}"
    "${FP16_ARGS[@]}"
)

RUN_NAMES=()

resolve_run_name() {
    local model_config="$1"
    local model_type="${2:-}"
    local args=(
        --dataset_config "$CONFIG"
        --model_config "$model_config"
        --seed "$SEED"
        --model_name "$MODEL_NAME"
    )
    if [[ -n "$model_type" ]]; then
        args+=(--model_type "$model_type")
    fi
    "$PYTHON" scripts/resolve_run_name.py "${args[@]}"
}

add_run_name() {
    local run_name="$1"
    RUN_NAMES+=("$run_name")
}

echo "############ dataset: $DS_NAME ############"
echo "processed_dir=$PDIR"
echo "device=$DEVICE model_name=$MODEL_NAME seed=$SEED epochs=$EPOCHS patience=$PATIENCE fp16=$USE_FP16"
echo "run_flags trivial=$RUN_TRIVIAL traditional=$RUN_TRADITIONAL mh=$RUN_MH two_stage=$RUN_TWO_STAGE"
echo "mh_model_config=$MH_MODEL_CONFIG"
echo "outputs=$OUTPUT_BASE"
echo "logs=$LOG_BASE"

if [[ "$RUN_TRIVIAL" == "1" ]]; then
    echo "############ baseline: majority ############"
    read -r -a TRIVIAL_METHOD_ARRAY <<< "$TRIVIAL_METHODS"
    if [[ ${#TRIVIAL_METHOD_ARRAY[@]} -eq 1 ]]; then
        TRIVIAL_RUN="${TRIVIAL_RUN:-majority}"
        mkdir -p "$LOG_BASE/$TRIVIAL_RUN"
        "$PYTHON" -m src.baselines.trivial_baselines \
            --dataset_config "$CONFIG" \
            --methods "${TRIVIAL_METHOD_ARRAY[@]}" \
            --run_name "$TRIVIAL_RUN" \
            --split id_eval eval \
            --out_root outputs \
            2>&1 | tee "$LOG_BASE/$TRIVIAL_RUN/predict.log"
        add_run_name "$TRIVIAL_RUN"
    else
        mkdir -p "$LOG_BASE/trivial"
        "$PYTHON" -m src.baselines.trivial_baselines \
            --dataset_config "$CONFIG" \
            --methods "${TRIVIAL_METHOD_ARRAY[@]}" \
            --split id_eval eval \
            --out_root outputs \
            2>&1 | tee "$LOG_BASE/trivial/predict.log"
        for method in "${TRIVIAL_METHOD_ARRAY[@]}"; do
            add_run_name "$method"
        done
    fi
fi

if [[ "$RUN_TRADITIONAL" == "1" ]]; then
    echo "############ baseline: crf_svm ############"
    TRAD_RUN="${TRAD_RUN:-crf_svm}"
    mkdir -p "$LOG_BASE/$TRAD_RUN"
    RUN_NAME="$TRAD_RUN" bash scripts/run_traditional_crf_svm.sh "$CONFIG" \
        "${TRADITIONAL_ARGS[@]}" \
        2>&1 | tee "$LOG_BASE/$TRAD_RUN/train_eval.log"
    add_run_name "$TRAD_RUN"
fi

if [[ "$RUN_MH" == "1" ]]; then
    echo "############ baseline: mhbio_crf ############"
    MH_RUN="${MH_RUN:-mhbio_crf}"
    mkdir -p "$LOG_BASE/$MH_RUN"
    SEED="$SEED" bash scripts/run_single.sh \
        "$CONFIG" \
        "$MH_MODEL_CONFIG" \
        --run_name "$MH_RUN" \
        "${COMMON_TRAIN_ARGS[@]}" \
        --batch_size "$MH_BATCH" \
        --grad_accumulation "$MH_ACCUM" \
        2>&1 | tee "$LOG_BASE/$MH_RUN/train_eval.log"
    add_run_name "$MH_RUN"
fi

if [[ "$RUN_TWO_STAGE" == "1" ]]; then
    echo "############ baseline: bilou_select ############"
    TWO_STAGE_RUN="${TWO_STAGE_RUN:-bilou_select}"
    PROPOSAL_RUN="$TWO_STAGE_RUN/proposal"
    CLASSIFIER_RUN="$TWO_STAGE_RUN/classifier"
    PIPELINE_RUN="$TWO_STAGE_RUN"
    PROPOSAL_ROOT="$OUTPUT_BASE/$PROPOSAL_RUN"
    CLASSIFIER_ROOT="$OUTPUT_BASE/$CLASSIFIER_RUN"
    PIPELINE_ROOT="$OUTPUT_BASE/$PIPELINE_RUN"
    mkdir -p \
        "$LOG_BASE/$PROPOSAL_RUN" "$LOG_BASE/$CLASSIFIER_RUN" "$LOG_BASE/$PIPELINE_RUN" \
        "$PROPOSAL_ROOT/proposals" "$PIPELINE_ROOT/predictions"

    "$PYTHON" scripts/train.py \
        --dataset_config "$CONFIG" \
        --model_config configs/models/xlm_bilou_select.yaml \
        --model_type ts_proposal \
        --seed "$SEED" \
        "${COMMON_TRAIN_ARGS[@]}" \
        --batch_size "$SPAN_BATCH" \
        --grad_accumulation "$SPAN_ACCUM" \
        --candidate_chunk_size "$CANDIDATE_CHUNK_SIZE" \
        --run_name "$PROPOSAL_RUN" \
        --out_root outputs \
        2>&1 | tee "$LOG_BASE/$PROPOSAL_RUN/train.log"

    "$PYTHON" scripts/tune_thresholds.py \
        --device "$DEVICE" \
        --checkpoint "$PROPOSAL_ROOT/checkpoints/best" \
        --gold "$PDIR/dev.jsonl" \
        --batch_size "$PRED_BATCH" \
        --target_recall "$TARGET_RECALL" \
        2>&1 | tee "$LOG_BASE/$PROPOSAL_RUN/tune.log"

    "$PYTHON" scripts/predict_proposals.py \
        --device "$DEVICE" \
        --checkpoint "$PROPOSAL_ROOT/checkpoints/best" \
        --input "$PDIR/train.jsonl" \
        --split train \
        --batch_size "$PRED_BATCH" \
        --out_dir "$PROPOSAL_ROOT/proposals" \
        2>&1 | tee "$LOG_BASE/$PROPOSAL_RUN/train_proposals.log"

    "$PYTHON" scripts/predict_proposals.py \
        --device "$DEVICE" \
        --checkpoint "$PROPOSAL_ROOT/checkpoints/best" \
        --input "$PDIR/dev.jsonl" \
        --split dev \
        --batch_size "$PRED_BATCH" \
        --out_dir "$PROPOSAL_ROOT/proposals" \
        2>&1 | tee "$LOG_BASE/$PROPOSAL_RUN/dev_proposals.log"

    cat "$PROPOSAL_ROOT/proposals/train.jsonl" \
        "$PROPOSAL_ROOT/proposals/dev.jsonl" \
        > "$PROPOSAL_ROOT/proposals/train_dev.jsonl"

    "$PYTHON" scripts/train.py \
        --dataset_config "$CONFIG" \
        --model_config configs/models/xlm_bilou_select.yaml \
        --model_type ts_classifier \
        --seed "$SEED" \
        "${COMMON_TRAIN_ARGS[@]}" \
        --proposal_file "$PROPOSAL_ROOT/proposals/train_dev.jsonl" \
        --batch_size "$STAGE2_BATCH" \
        --grad_accumulation "$STAGE2_ACCUM" \
        --run_name "$CLASSIFIER_RUN" \
        --out_root outputs \
        2>&1 | tee "$LOG_BASE/$CLASSIFIER_RUN/train.log"

    "$PYTHON" scripts/tune_thresholds.py \
        --device "$DEVICE" \
        --checkpoint "$CLASSIFIER_ROOT/checkpoints/best" \
        --gold "$PDIR/dev.jsonl" \
        --proposal_file "$PROPOSAL_ROOT/proposals/dev.jsonl" \
        --batch_size "$PRED_BATCH" \
        2>&1 | tee "$LOG_BASE/$CLASSIFIER_RUN/tune.log"

    "$PYTHON" scripts/predict_two_stage.py \
        --device "$DEVICE" \
        --proposal_checkpoint "$PROPOSAL_ROOT/checkpoints/best" \
        --classifier_checkpoint "$CLASSIFIER_ROOT/checkpoints/best" \
        --input "$PDIR/id_eval.jsonl" \
        --split id_eval \
        --batch_size "$PRED_BATCH" \
        --out_dir "$PIPELINE_ROOT/predictions" \
        2>&1 | tee "$LOG_BASE/$PIPELINE_RUN/id_eval_predict.log"

    "$PYTHON" scripts/predict_two_stage.py \
        --device "$DEVICE" \
        --proposal_checkpoint "$PROPOSAL_ROOT/checkpoints/best" \
        --classifier_checkpoint "$CLASSIFIER_ROOT/checkpoints/best" \
        --input "$PDIR/eval.jsonl" \
        --split eval \
        --batch_size "$PRED_BATCH" \
        --out_dir "$PIPELINE_ROOT/predictions" \
        2>&1 | tee "$LOG_BASE/$PIPELINE_RUN/eval_predict.log"

    add_run_name "$PIPELINE_RUN"
fi

echo "############ unified evaluation ############"
bash scripts/evaluate_predictions.sh "$CONFIG" "${RUN_NAMES[@]}" --collect \
    2>&1 | tee "$LOG_BASE/evaluate_collect.log"

echo "############ done: $DS_NAME ############"
echo "Summary: $OUTPUT_BASE/summary_results.md"
echo "Outputs: $OUTPUT_BASE/"
echo "Logs: $LOG_BASE/"
