#!/usr/bin/env bash
# ============================================================
# Step 1 (variant, local backend): Streaming inference WITHOUT GT history
# ------------------------------------------------------------
# Inference logic is identical to scripts/infer_no_gt_history.sh (per-second replay,
# using the model's own predictions as history, Silent not counted into history),
# except the backend is swapped from cloud API to a local Qwen3-VL model
# (transformers + qwen_vl_utils).
#
# Run in a suitable environment (with GPU):
#     conda activate <your_env>   # requires torch / transformers>=4.57 / qwen-vl-utils
#     bash scripts/infer_no_gt_history_local.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."   # move to release root

# Local checkpoint path (or HF hub id; pulls automatically if not downloaded)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
model_name="${MODEL_NAME:-Qwen3-VL-8B-Instruct}"

# Use a single GPU for local inference; adjust as needed
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

seed="${SEED:-78}"

# Raw annotation file (a list of per-video sessions), NOT the pre-split validated dir.
ANNOS_DIR="${ANNOS_DIR:-data/example/annotation_test_subset50.json}"
# Root that the annotation's 1s clips (e.g. 0009_0_1.mp4) are relative to.
VIDEO_BASE_DIR="${VIDEO_BASE_DIR:-data/processed/video_clips}"
# Source videos, only used to probe duration for trimming the silent tail.
VIDEO_DIR="${VIDEO_DIR:-data/example/videos_2fps}"
# Kept separate so the two variants don't mix.
OUTPUT_DIR="${OUTPUT_DIR:-output/infer_no_gt_history}"

TIME_WINDOW="${TIME_WINDOW:-60}"
CHUNK_SECONDS="${CHUNK_SECONDS:-1}"

FPS="${FPS:-2.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.0}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"          # or flash_attention_2 (requires flash-attn)
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"

# Set DEBUG=1 to also dump the intermediate per-second context history.
DEBUG="${DEBUG:-0}"
debug_flag=""
if [ "$DEBUG" = "1" ] || [ "$DEBUG" = "true" ]; then
    debug_flag="--debug"
fi

mkdir -p output/log
timestamp=$(date +%Y%m%d%H%M%S)

python infer_no_gt_history.py \
    --backend local \
    --model "$model_name" \
    --model_path "$MODEL_PATH" \
    --seed "$seed" \
    --fps "$FPS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --attn_implementation "$ATTN_IMPL" \
    --torch_dtype "$TORCH_DTYPE" \
    --ANNOS_DIR "$ANNOS_DIR" \
    --VIDEO_BASE_DIR "$VIDEO_BASE_DIR" \
    --VIDEO_DIR "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --time_window "$TIME_WINDOW" \
    --chunk_seconds "$CHUNK_SECONDS" \
    $debug_flag \
    2>&1 | tee "output/log/${model_name}_no_gt_history_${timestamp}.log"

echo "no-gt-history (local) inference done -> ${OUTPUT_DIR}/${model_name}"
