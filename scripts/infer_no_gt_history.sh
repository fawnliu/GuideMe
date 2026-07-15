#!/usr/bin/env bash
# ============================================================
# Step 1 (variant): Streaming inference WITHOUT GT history
# ------------------------------------------------------------
# Unlike scripts/infer.sh (which reads the pre-split validated/ samples and
# uses the GROUND-TRUTH assistant turns as dialogue history), this variant:
#   - loads the RAW annotation file directly (a list of per-video sessions);
#   - replays each video second-by-second (start_t .. last_speak + silent tail);
#   - feeds the model its OWN previous predictions as history;
#   - drops predictions equal to "Silent" from the history;
#   - writes ONE JSON per video, saved as soon as that video finishes.
#
# The 1s video chunks referenced by the annotation must already exist under
# $VIDEO_BASE_DIR (produced by scripts/preprocess.sh). $VIDEO_DIR is only used
# to probe source-video duration for cropping the trailing silent window.
#
# Run this from the release root:
#     cp api_setting.example.yaml api_setting.yaml   # then fill in api_key / api_host (one-time)
#     bash scripts/infer_no_gt_history.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."   # move to release root

# API credentials for the cloud backend are loaded from api_setting.yaml at the
# repo root (gitignored; copy from api_setting.example.yaml). Env vars
# LLM_API_KEY / LLM_API_HOST still override it if set.


model_name="${MODEL_NAME:-gemini-3-pro-preview}"
seed="${SEED:-78}"

# Raw annotation file (a list of per-video sessions), NOT the pre-split validated dir.
ANNOS_DIR="${ANNOS_DIR:-data/example/annotation_test_subset50.json}"
# Root that the annotation's 1s clips (e.g. 0009_0_1.mp4) are relative to.
VIDEO_BASE_DIR="${VIDEO_BASE_DIR:-data/processed/video_clips}"
# Source videos, only used to probe duration for trimming the silent tail.
VIDEO_DIR="${VIDEO_DIR:-data/example/videos_2fps}"
# Kept separate from scripts/infer.sh output so the two variants don't mix.
OUTPUT_DIR="${OUTPUT_DIR:-output/infer_no_gt_history}"

TIME_WINDOW="${TIME_WINDOW:-60}"
CHUNK_SECONDS="${CHUNK_SECONDS:-1}"

# Set DEBUG=1 to also dump the intermediate per-second context history under
# output/debug/infer_no_gt_history/<model>/. Default: off (only merged output).
DEBUG="${DEBUG:-0}"
debug_flag=""
if [ "$DEBUG" = "1" ] || [ "$DEBUG" = "true" ]; then
    debug_flag="--debug"
fi

mkdir -p output/log
timestamp=$(date +%Y%m%d%H%M%S)

python infer_no_gt_history.py \
    --model "$model_name" \
    --seed "$seed" \
    --ANNOS_DIR "$ANNOS_DIR" \
    --VIDEO_BASE_DIR "$VIDEO_BASE_DIR" \
    --VIDEO_DIR "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --time_window "$TIME_WINDOW" \
    --chunk_seconds "$CHUNK_SECONDS" \
    $debug_flag \
    2>&1 | tee "output/log/${model_name}_no_gt_history_${timestamp}.log"

echo "no-gt-history inference done -> ${OUTPUT_DIR}/${model_name}"
