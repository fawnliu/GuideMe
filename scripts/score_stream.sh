#!/usr/bin/env bash
# ============================================================
# Step 2: Merge + score (all-in-one score_stream.py)
# ------------------------------------------------------------
# Run the full scoring pipeline with the all-in-one score_stream.py:
#   Stage 1: merge infer jsons          (was tools/merge_infer_conv.py, passed in memory)
#   Stage 2: matching P/R/F1            (was score/metric_match_final.py)
#   Stage 3: LLM-judge (raw)            (was score/score_v2.py)
#   Stage 4: LLM-judge (matched pairs)  (was score/score_v2_matched.py)
#
# Output (intermediate results are all passed in memory; no more output/infer_merged/):
#   per-video jsons -> output/infer/{model}/            (produced by infer.py)
#   final results   -> output/eval/{model}/evaluation_results.txt
#                      [Temporal Alignment] soft P/R/F1 + score_matched
#                      [Response Behavior]  CS/NR/FA/PC + score
#   intermediate(debug) -> output/eval/debug/{model}/   (only with DEBUG=1)
#                      metrics_summary.json / v4.4*.db|csv / score_v2*.log
#
# Usage (from the release root, after running scripts/infer.sh):
#   cp api_setting.example.yaml api_setting.yaml   # then fill in api_key / api_host (one-time)
#   bash scripts/score_stream.sh                     # run the models in model_list below (all 4 steps)
#   bash scripts/score_stream.sh model_a model_b     # specify models on the command line (overrides model_list)
#   DEBUG=1 bash scripts/score_stream.sh             # also save intermediate artifacts under output/eval/debug/
# ============================================================
set -e
cd "$(dirname "$0")/.."   # move to release root

# ---------- API credentials (needed by stage 3/4 LLM judge) ----------
# Loaded from api_setting.yaml at the repo root (gitignored; copy from
# api_setting.example.yaml). Env vars LLM_API_KEY / LLM_API_HOST still override
# it if set, so no export is required here.

# ---------- Models to evaluate ----------
model_list=(
    "gemini-3-pro-preview"
)
# Model names passed on the command line take precedence over the list above
if [ "$#" -gt 0 ]; then
    model_list=("$@")
fi

# ---------- Directories ----------
infer_root="output/infer"           # per-video merged jsons (from infer.py)
eval_root="output/eval"             # all scoring outputs land here
# #### infer no gt results
# infer_root="output/infer_no_gt_history"     # per-video merged jsons (from infer_no_gt_history.py)
# eval_root="output/eval_no_gt_history"       # all scoring outputs land here
workers="${WORKERS:-32}"            # number of parallel LLM-judge threads
# Stage 3 only: a non-silent prediction within +/- this many seconds of a non-silent GT
# timestamp counts as PartlyCorrect (0 = strict same-frame rule).
pc_time_window="${PC_TIME_WINDOW:-2}"

# Set DEBUG=1 to also save intermediate artifacts under {eval_root}/debug/{model}/
# (metrics_summary.json, per-stage db/csv, score_v2*.log). Off by default: only
# {eval_root}/{model}/evaluation_results.txt is written.
DEBUG="${DEBUG:-0}"
debug_flag=""
if [ "$DEBUG" = "1" ] || [ "$DEBUG" = "true" ]; then
    debug_flag="--debug"
fi

# score_stream.py runs all 4 steps in a fixed order; internally it loops over models, prints each stage
# banner, and (with --debug) tees stage 3/4 output to {eval_root}/debug/{model}/score_v2*.log
python score_stream.py "${model_list[@]}" \
    --infer-root "${infer_root}" \
    --eval-root "${eval_root}" \
    --workers "${workers}" \
    --pc-time-window "${pc_time_window}" \
    ${debug_flag}

echo "scoring done -> ${eval_root}"
