#!/usr/bin/env bash
# ============================================================
# Step 1 (local backend): Run inference with a local HF model
# ------------------------------------------------------------
# Inference logic is identical to scripts/infer.sh, except the backend is
# changed from cloud API to a local Qwen3-VL model (transformers + qwen_vl_utils).
#
# Run in a suitable environment (with GPU):
#     conda activate <your_env>   # requires torch / transformers>=4.57 / qwen-vl-utils
#     bash scripts/infer_local.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."   # move to release root

# Local checkpoint path (or HF hub id; pulls automatically if not downloaded)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
# Model name used for the output subdirectory (leave empty to auto-derive from the checkpoint dir name)
model_name="${MODEL_NAME:-Qwen3-VL-8B-Instruct}"

# Use a single GPU for local inference; adjust as needed
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Video reader backend for qwen-vl-utils: newer torchvision (>=0.22) removed io.read_video,
# so force decord here to avoid falling back to the broken torchvision backend.
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"

timestamp=$(date +%Y%m%d%H%M%S)

VIDEO_BASE_DIR="${VIDEO_BASE_DIR:-data/processed/video_clips}"
ANNOS_DIR="${ANNOS_DIR:-data/processed/validated}"
OUTPUT_DIR="${OUTPUT_DIR:-output/infer}"

FPS="${FPS:-2.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.0}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"          # or flash_attention_2 (requires flash-attn)
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"

mkdir -p output/log

# Set DEBUG=1 to also dump the intermediate per-turn context history.
DEBUG="${DEBUG:-0}"
debug_flag=""
if [ "$DEBUG" = "1" ] || [ "$DEBUG" = "true" ]; then
    debug_flag="--debug"
fi

python infer.py \
    --backend local \
    --model "$model_name" \
    --model_path "$MODEL_PATH" \
    --num_workers 1 \
    --fps "$FPS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --attn_implementation "$ATTN_IMPL" \
    --torch_dtype "$TORCH_DTYPE" \
    --VIDEO_BASE_DIR "$VIDEO_BASE_DIR" \
    --ANNOS_DIR "$ANNOS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $debug_flag \
    2>&1 | tee "output/log/${model_name}_${timestamp}.log"

echo "inference done -> ${OUTPUT_DIR}/${model_name}"
