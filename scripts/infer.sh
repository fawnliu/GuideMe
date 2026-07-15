#!/usr/bin/env bash
# ============================================================
# Step 1: Run inference
# ------------------------------------------------------------
# Runs a (Gemini / OpenAI-compatible) model over the streaming
# video annotations and writes per-file predictions.
#
# Run this from the release root:
#     cp api_setting.example.yaml api_setting.yaml   # then fill in api_key / api_host (one-time)
#     bash scripts/infer.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."   # move to release root

# API credentials for the cloud backend are loaded from api_setting.yaml at the
# repo root (gitignored; copy from api_setting.example.yaml). Env vars
# LLM_API_KEY / LLM_API_HOST still override it if set.


model_name="gemini-3-pro-preview"
workers=32
timestamp=$(date +%Y%m%d%H%M%S)


# Root directory that the video paths in the annotations are relative to.
# The annotation json files reference clips like "0006_0_12.mp4",
# so VIDEO_BASE_DIR must contain those clip files.
# Set this to wherever your video clips live.
VIDEO_BASE_DIR="${VIDEO_BASE_DIR:-data/processed/video_clips}"
ANNOS_DIR="${ANNOS_DIR:-data/processed/validated}"   # validated samples (Step 0 output)
OUTPUT_DIR="${OUTPUT_DIR:-output/infer}"             # per-file predictions

mkdir -p output/log

# Set DEBUG=1 to also dump the intermediate per-turn context history under
# output/debug/infer/<model>/. Default: off (only merged output).
DEBUG="${DEBUG:-0}"
debug_flag=""
if [ "$DEBUG" = "1" ] || [ "$DEBUG" = "true" ]; then
    debug_flag="--debug"
fi

python infer.py \
    --model "$model_name" \
    --seed "$timestamp" \
    --num_workers "$workers" \
    --VIDEO_BASE_DIR "$VIDEO_BASE_DIR" \
    --ANNOS_DIR "$ANNOS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $debug_flag \
    2>&1 | tee "output/log/${model_name}_${timestamp}.log"

echo "inference done -> ${OUTPUT_DIR}/${model_name}"

