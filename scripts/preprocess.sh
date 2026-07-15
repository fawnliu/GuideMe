#!/usr/bin/env bash
# ============================================================
# Step 0: Data preparation (annotations -> samples -> clips)
# ------------------------------------------------------------
# The source videos are distributed already re-encoded to 2 fps, so this script
# reads them directly from $VIDEO_DIR (default data/example/videos_2fps).
#
# 1) process/data_proprecess.py  : split the raw annotation JSON (a single file
#                                  containing a list of samples) into type1/type2
#                                  streaming samples with rolling <video>
#                                  windows, writing one JSON per source video
#                                  (named after its `videos` entry, e.g.
#                                  0009.json)  ->  data/processed/save/
# 2) process/chunk_videos_1s.py  : cut every source video into dense, contiguous
#                                  1s chunks (0_1, 1_2, ...) -> $VIDEO_CLIPS_1S_DIR
#                                  (so inference can pick any streaming
#                                  granularity), then validate the chunks the
#                                  samples reference and drop samples with any
#                                  missing/unreadable chunk -> $VALIDATED_DIR.
#
# Inputs (defaults point at the bundled demo under data/example/):
#   - a single raw annotation JSON file $ANNO_FILE (a list of samples; a
#     directory of *.json files is also accepted for backward compatibility)
#   - 2 fps source videos under $VIDEO_DIR
#
# Run from the release root:
#     bash scripts/preprocess.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."   # move to release root

ANNO_FILE="${ANNO_FILE:-data/example/annotation_test_subset50.json}"   # raw annotation JSON (single file, a list of samples)
VIDEO_DIR="${VIDEO_DIR:-data/example/videos_2fps}"           # re-encoded videos (input to clipping)

SAVE_DIR="${SAVE_DIR:-data/processed/save}"                   # split samples (intermediate)
VALIDATED_DIR="${VALIDATED_DIR:-data/processed/validated}"    # validated samples (Step 1 input)
VIDEO_CLIPS_1S_DIR="${VIDEO_CLIPS_1S_DIR:-data/processed/video_clips}"  # dense 1s chunks == inference VIDEO_BASE_DIR
CHUNK_SECONDS="${CHUNK_SECONDS:-1}"
TIME_WINDOW="${TIME_WINDOW:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# 1) build per-second streaming samples (1s <video> chunks)
python process/data_proprecess.py \
    --annos_base_dir "$ANNO_FILE" \
    --output_dir "$SAVE_DIR" \
    --video_dir "$VIDEO_DIR" \
    --time_window "$TIME_WINDOW" \
    --chunk_seconds "$CHUNK_SECONDS"

# 2) cut dense 1s chunks over every source video (audio dropped), then
#    validate the referenced chunks and write the validated samples
python process/chunk_videos_1s.py \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$VIDEO_CLIPS_1S_DIR" \
    --chunk_seconds "$CHUNK_SECONDS" \
    --annos_save_dir "$SAVE_DIR" \
    --validated_annos_dir "$VALIDATED_DIR" \
    --num_workers "$NUM_WORKERS" \
    --remove_audio

echo "preprocess done -> $VALIDATED_DIR + $VIDEO_CLIPS_1S_DIR"
