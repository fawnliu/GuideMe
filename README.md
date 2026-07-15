# GuideMe: Multi-Domain Task Guidance and Intervention in Streaming Video (ECCV 2026)

<p align="center">
  <a href="https://arxiv.org/abs/2607.02991"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white" /></a>
  <a href="https://fawnliu.github.io/project/guideme/"><img src="https://img.shields.io/badge/Project-Page-blue?logo=googlechrome&logoColor=white" /></a>
  <img src="https://img.shields.io/badge/ECCV-2026-green" />
</p>

## Abstract

While multimodal Large Language Models (MLLMs) excel at offline video understanding, how far they are from serving as a real-time procedural coach remains unknown. Such a role requires an MLLM to continuously monitor the execution, detect mistakes, and provide corrective guidance in a closed-loop interaction. **GuideMe** is the first multi-domain benchmark for streaming video that supports training and evaluation of MLLMs for closed-loop interactive task guidance, covering the full loop of instruction, feedback, error detection, and correction—dimensions that prior procedural video datasets support only partially. Spanning cooking, object manipulation, daily-life guidance, and fitness, it assesses models in realistic scenarios where an assistant must monitor user actions, decide when to remain silent, and provide timely next-step instructions, completion feedback, error detection, and corrective guidance. Together with a three-facet evaluation framework that jointly measures sequence-level alignment, intervention timing, and content quality, our benchmark reveals that despite excelling at providing instructions, existing MLLMs consistently fail to identify execution errors and respond with corrective feedback.

## Dataset

The benchmark contains **2,458 video instances** with a combined duration of **223.7 hours**, yielding **47,775 streaming interaction samples** across four task domains. Video lengths range from 0.5 to 41.2 minutes (average 5.5 min, median 3.6 min).


| Split         | #Videos | Hours | Notes                                  |
| ------------- | ------- | ----- | -------------------------------------- |
| GuideMe-Train | 1,985   | 177.0 | Task-specific adaptation / fine-tuning |
| GuideMe-Test  | 473     | 46.7  | Held-out set used for all evaluations  |


Each interaction sample covers one of four guidance types: next-step instructions, completion feedback, error detection, and corrective guidance.

The four task domains map to the following dataset sources used throughout this
codebase: `captaincook4d` (cooking), `egoper` (daily-life guidance), `holoassist`
(object manipulation), and `fitness`.

> **Getting the data:** this repository ships with a small preview subset. See
> `[data/README.md](data/README.md)` for how to download the example data (the
> `alisn/GuideMe_Test_Subset50` dataset on Hugging Face), where to place the
> `annotation/` and `videos_2fps/` folders, and what every path under
> `data/example/` and `data/processed/` means.



## Usage

This repository provides the full **evaluation pipeline** for GuideMe:
data preparation → inference with an MLLM (cloud Gemini / OpenAI-compatible API,
or a local HF model) → three-facet scoring. Run everything from the repository root.

### Setup

The project is managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml`

- `uv.lock`, Python 3.12):

```bash
uv sync                       # creates .venv/ with all pinned dependencies
source .venv/bin/activate     # the scripts below invoke `python` directly
# also requires the `ffmpeg` binary on PATH (Step 0 video chunking)

# API credentials (only needed for the API backend and the LLM judge in Step 2):
# Copy the template once, then fill in your own key/host. api_setting.yaml is
# gitignored, so the secret is never committed; every infer/score script reads it.
cp api_setting.example.yaml api_setting.yaml
#   api_key:  "your-api-key"
#   api_host: "your-api-host"   # e.g. api.example.com (no scheme)
# (env vars LLM_API_KEY / LLM_API_HOST, if set, still override the file)
```

The locked environment already includes everything for the **local inference
backend** (Step 1): `torch`, `transformers==4.57.3`, `qwen-vl-utils`, `decord`.
Running it just additionally requires a GPU.

### Step 0 — Data preparation

Turns the raw annotations in `data/example/annotation_test_subset50.json` into
the preprocessed samples under `data/processed/validated/` (consumed by Step 1)
and cuts the source videos into dense 1s chunks. Run this before Step 1.
`scripts/preprocess.sh` runs two stages in order: **build samples → cut 1s
chunks + validate**.

> The source videos are distributed **already re-encoded to 2 fps** (H.265,
> fixed timestamps), so no local re-encoding is required. First make sure the
> example data is in place — follow `[data/README.md](data/README.md)` to
> download it and populate `data/example/annotation_test_subset50.json` and
> `data/example/videos_2fps/`.

**Inputs:**

- `--annos_base_dir` — the raw annotation JSON (default
`data/example/annotation_test_subset50.json`, already bundled). 
- `--video_dir` (default `data/example/videos_2fps`) — root of the source videos,
distributed already re-encoded to 2 fps. 
  ```
  data/example/videos_2fps/
  ├── 0006.mp4
  └── R0027-12-GoPro.mp4
  ```

**Outputs produced:**

- `data/processed/save/` — intermediate split samples (from `data_proprecess.py`).
- `--output_dir` (default `data/processed/video_clips`) — the **dense 1s video
chunks** covering every source video (`0006_0_1.mp4`, `0006_1_2.mp4`, ...), so
inference can pick any streaming granularity.
- `data/processed/validated/` — final samples, with any sample referencing an
unreadable chunk removed.

**Run it:**

```bash
# All defaults already point at the bundled data/example/ demo
# (annotation_test_subset50.json + videos_2fps/), so you can just run:
bash scripts/preprocess.sh

# To run on your own data, override via env vars, e.g.:
#   export ANNO_FILE=/path/to/annotations.json      # single JSON file (a list of samples)
#   export VIDEO_DIR=/path/to/videos_2fps           # 2 fps videos (clip input)
#   export VIDEO_CLIPS_1S_DIR=data/processed/video_clips
```

Or call the two stages directly with explicit flags:

```bash
python process/data_proprecess.py \
    --annos_base_dir data/example/annotation_test_subset50.json \
    --output_dir data/processed/save \
    --video_dir data/example/videos_2fps \
    --time_window 60 \
    --chunk_seconds 1

python process/chunk_videos_1s.py \
    --video_dir data/example/videos_2fps \
    --output_dir data/processed/video_clips \
    --chunk_seconds 1 \
    --annos_save_dir data/processed/save \
    --validated_annos_dir data/processed/validated \
    --num_workers 8 \
    --remove_audio
```

> **Important — video clip location:** the directory you pass as `--output_dir`
> in Step 0 is exactly the directory the inference step reads as `VIDEO_BASE_DIR`
> in Step 1. Keep them consistent (both default to `data/processed/video_clips`)
> so the chunks produced here are found during inference:
>
> ```
> Step 0  --output_dir  ==  Step 1  VIDEO_BASE_DIR   (default: data/processed/video_clips)
> ```



### Step 1 — Inference

Inference has two independent switches — the **dialogue-history mode** and the
**backend** — giving four launcher scripts:


| Script                                 | History fed to the model     | Backend        |
| -------------------------------------- | ---------------------------- | -------------- |
| `scripts/infer.sh`                     | ground-truth assistant turns | cloud API      |
| `scripts/infer_local.sh`               | ground-truth assistant turns | local HF model |
| `scripts/infer_no_gt_history.sh`       | model's own predictions      | cloud API      |
| `scripts/infer_no_gt_history_local.sh` | model's own predictions      | local HF model |


> **Cloud-API backends need credentials.** Before running `infer.sh` /
> `infer_no_gt_history.sh` (the `cloud API` rows above), create `api_setting.yaml`
> once — `cp api_setting.example.yaml api_setting.yaml` and fill in your `api_key`
> / `api_host`. It is gitignored, and every infer/score script reads it (see
> [Setup](#setup)). The two `*_local.sh` scripts need no API key.


**GT-history mode (**`infer.py`**)** — the standard protocol. Reads the pre-split
samples under `data/processed/validated/` (Step 0 output); at every turn the
dialogue history contains the **ground-truth** assistant responses, so each
prediction is conditioned on a correct context.

```bash
bash scripts/infer.sh            # cloud API backend
bash scripts/infer_local.sh      # local HF backend
```

Writes one merged JSON per video to `output/infer/<model_name>/`. By default
every file under `data/processed/validated/` is processed (pass
`--subset <file>` to restrict to a subset list).

**No-GT-history mode (**`infer_no_gt_history.py`**)** — a stricter, deployment-like
protocol. Instead of the pre-split samples it loads the raw annotation file
(`data/example/annotation_test_subset50.json`) directly, replays each video
**second-by-second** (from the start to the last speaking point plus a trailing
silent window), and feeds the model **its own previous predictions** as the
dialogue history; predictions equal to "Silent" are dropped from the history.
Errors therefore accumulate as they would in a real
streaming session.

```bash
bash scripts/infer_no_gt_history.sh          # cloud API backend
bash scripts/infer_no_gt_history_local.sh    # local HF backend
```

Writes one JSON per video to `output/infer_no_gt_history/<model_name>/` (kept
separate from the GT-history output so the two protocols don't mix). The output
format is identical to `infer.py`'s, so Step 2 scoring works unchanged — just
point it at the other directory (see below).

**Local backend** — both entry points accept `--backend local` to run a local
HuggingFace checkpoint (tested with Qwen3-VL-Instruct via `transformers` +
`qwen_vl_utils`, see `tools/local_backend.py`) instead of a cloud API; no
`LLM_API_KEY` is needed for this step. The `*_local.sh` scripts expose the
knobs as env vars:

```bash
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct MODEL_NAME=Qwen3-VL-8B-Instruct \
CUDA_VISIBLE_DEVICES=0 bash scripts/infer_local.sh
# also configurable: FPS, MAX_NEW_TOKENS, TEMPERATURE, ATTN_IMPL (sdpa /
# flash_attention_2), TORCH_DTYPE
```

Local inference is forced to a single worker (one GPU, sequential videos).

For any script, `DEBUG=1` additionally dumps the intermediate per-turn context
history under `output/debug/`.

### Step 2 — Score

```bash
# LLM judge is API-based; credentials come from api_setting.yaml (see Setup above)
bash scripts/score_stream.sh                     # models from the list in the script
bash scripts/score_stream.sh model_a model_b     # or pass model names explicitly
```

`scripts/score_stream.sh` drives the all-in-one `score_stream.py`, which runs
the three-facet evaluation in four stages (intermediate data is passed in
memory, nothing extra is written to disk):

1. **Merge** — merges per-turn inference files into one conversation per video.
2. **Matching P/R/F1** — sequence-level alignment + intervention timing:
  matching-based Precision / Recall / F1 (soft + hard) via min-weight
   bipartite matching, using `sentence-transformers/all-mpnet-base-v2`.
3. **LLM judge (raw stream)** — content quality: LLM-judge scoring of proactive
  responses on the raw merged stream (CorrectSilent / FalseAlarm / NoResponse /
   PartlyCorrect).
4. **LLM judge (matched pairs)** — the same content-quality scoring, but over
  the bipartite-matched (gen, ref) pairs from stage 2.

**Outputs:**

- `output/eval/<model_name>/evaluation_results.txt` — the final report, two
metric groups: **Temporal Alignment** (soft precision / recall / F1,
score_matched) and **Response Behavior** (CS / NR / FA / PC, score).
- `output/eval/debug/<model_name>/` — intermediate artifacts
(`metrics_summary.json`, judge `.csv` / `.db`, stage logs).

To score the **no-GT-history** results, switch the two root variables in
`scripts/score_stream.sh` to the variant directories (the commented lines are
already there):

```bash
infer_root="output/infer_no_gt_history"
eval_root="output/eval_no_gt_history"
```



## Notes

- All API keys were removed from the source; credentials are loaded from a local
`api_setting.yaml` at the repo root (copy `api_setting.example.yaml`; the file
is gitignored so it is never committed). The env vars `LLM_API_KEY` /
`LLM_API_HOST` still override it if set. The local inference backend needs no
API key (but Step 2's LLM judge always does).
- For the API backend, models whose name starts with `doubao`, `gpt`, `claude`,
`deepseek`, or `qwen` are called via the OpenAI-compatible endpoint; others
use the Gemini-native endpoint (see `tools/llm_utils.py`).



## Citation

If you find this work useful, please consider citing:

```bibtex
@article{liu2026guideme,
  title={GuideMe: Multi-Domain Task Guidance and Intervention in Streaming Video},
  author={Liu, Fang and Chen, Jinpeng and Xu, Ke and Liu, Yuhao and Guan, Huankang and Lu, Xudong and Yang, Bo and Hancke, Gerhard and Liu, Rui and Lau, Rynson WH},
  journal={arXiv preprint arXiv:2607.02991},
  year={2026}
}
```

