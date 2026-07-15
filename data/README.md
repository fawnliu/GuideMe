# GuideMe — Data

This directory holds the input data (`example/`) and all preprocessing outputs
(`processed/`) for the GuideMe evaluation pipeline.

> **Preview subset.** What we release here is a **small preview subset** of the
> GuideMe benchmark, for early-stage verification of the data format and the
> evaluation pipeline. The **full data will be released soon** (2,458 videos /
> 223.7 hours / 47,775 streaming interaction samples across cooking, daily-life
> guidance, object manipulation, and fitness).

## Download the example data

The example subset is hosted on Hugging Face:

- Dataset: **`alisn/GuideMe_Test_Subset50`**
- URL: https://huggingface.co/datasets/alisn/GuideMe_Test_Subset50

The videos are distributed **already re-encoded to 2 fps** (H.265, fixed
timestamps, audio kept), so no local re-encoding is needed. Download it, then
move the `annotation_test_subset50.json` file and the `videos_2fps/` folder
under `data/example/`:

```bash
# Option A: git-lfs
git lfs install
git clone https://huggingface.co/datasets/alisn/GuideMe_Test_Subset50

# Option B: huggingface_hub CLI
pip install -U "huggingface_hub[cli]"

hf download alisn/GuideMe_Test_Subset50 --repo-type dataset --local-dir ./data/example
```

After this, `data/example/` should contain both `annotation_test_subset50.json`
and `videos_2fps/` (see the layout below).

## `example/` — input data

```
data/example/
├── annotation_test_subset50.json   # 50 raw annotation samples (a single JSON list)
└── videos_2fps/                    # 50 corresponding source videos, already re-encoded to 2 fps (.mp4)
```

- **`annotation_test_subset50.json`** — a single JSON file holding a list of 50
  samples. Each sample references its source video through the `videos` field
  (e.g. `["0009.mp4"]`); the domain of each sample is carried inside the sample
  itself. `data_proprecess.py` reads this file and writes one JSON per source
  video (named after its `videos` entry, e.g. `0009.json`) into `processed/save/`.
- **`videos_2fps/`** — the source videos referenced by the annotations, already
  re-encoded to 2 fps (H.265, fixed timestamps, audio kept). The file name
  matches the `videos` field of the corresponding sample. This is the
  directory the clipping step reads as its `--video_dir`.

In this subset every sample has its referenced video available under
`videos_2fps/`, so the full pipeline (data preparation → inference → scoring)
can be run end-to-end.

## `processed/` — preprocessing outputs

These folders are **produced by the pipeline** (`scripts/preprocess.sh`, i.e.
`process/data_proprecess.py` followed by `process/chunk_videos_1s.py`). They do
not need to be downloaded — they are generated from `example/`.

```
data/processed/
├── save/            # intermediate streaming samples (output of data_proprecess.py)
├── video_clips/     # dense 1s video chunks over every source video (output of chunk_videos_1s.py)
└── validated/       # final samples, with any sample referencing an invalid chunk removed
```

- **`save/`** — intermediate split samples, one JSON per source video (named
  after its `videos` entry, e.g. `0009.json`). `data_proprecess.py` turns each
  raw annotation into per-second streaming samples with rolling `<video>`
  context windows sliced into 1s chunks. The chunk file names encode the
  start/end seconds (e.g. `0009_0_1.mp4`).
- **`video_clips/`** — the actual **dense 1s video chunks**, produced by
  `chunk_videos_1s.py` from the videos in `example/videos_2fps/`
  (`0009_0_1.mp4`, `0009_1_2.mp4`, ...). This is the directory the inference
  step reads as `VIDEO_BASE_DIR`.
- **`validated/`** — the **final samples** consumed by inference (Step 1). It is
  `save/` after dropping any sample that references a missing or unreadable chunk.

Relationship at a glance:

```
example/annotation_test_subset50.json ──data_proprecess.py──────────►  processed/save
example/videos_2fps                    ──chunk_videos_1s.py───────────►  processed/video_clips
processed/save                         ──chunk_videos_1s.py (validate)►  processed/validated
```
