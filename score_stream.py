#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_stream.py — all-in-one scoring pipeline (replaces the 4 scripts in scripts/score.sh).

Stages (run in fixed order, intermediate results passed in memory):
    Stage 1: build jsonl        <- tools/merge_infer_conv.py
    Stage 2: matching P/R/F1    <- score/metric_match_final.py
    Stage 3: LLM-judge (raw)    <- score/score_v2.py
    Stage 4: LLM-judge (matched)<- score/score_v2_matched.py

Output layout:
  {eval_root}/{model}/evaluation_results.txt   final results
      [Temporal Alignment] soft_precision / soft_recall / soft_F1 / score_matched
      [Response Behavior]  CS / NR / FA / PC / score
  {eval_root}/debug/{model}/                   intermediate artifacts (metrics/db/csv/logs), only with --debug

Usage (from the release root):
    export LLM_API_KEY="your-api-key"      # required by the stage 3/4 judge
    python score_stream.py gemini-3-pro-preview
    # no model name -> iterates over all model dirs under output/infer/
"""

import argparse
import glob
import importlib
import json
import os
import shutil
import sqlite3
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
import sentence_transformers as sbert
from joblib import Parallel, delayed
from json_repair import repair_json
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from PIL import Image

# Ensure the release root (this file's directory) is importable so that
# `score.prompts.*` and the judger class `llm_utils.*` resolve regardless of the
# working directory this script is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from score.prompts.en_llm_judge import get_prompt


# ===========================================================================
# Stage 1: build the combined {model}.jsonl
# (copied verbatim from tools/merge_infer_conv.py)
# ===========================================================================

def video_timestamps(video_path: str) -> tuple[int, int]:
    """Parse the (start, end) timestamps from the video filename, e.g. ..._146_206.mp4 -> (146, 206)."""
    stem = Path(video_path).stem
    parts = stem.split("_")
    try:
        return int(parts[-2]), int(parts[-1])
    except (ValueError, IndexError):
        return 0, 0


def merge_samples(samples: list[dict]) -> dict:
    """
    Merge multiple cumulative-conversation samples in a single JSON file into one conversation.

    Rules:
    - system prompt: take the one from the first sample, keep only one
    - first user text message: take the one from the first sample, keep only one
    - <video> messages: skip all of them
    - assistant messages: keep only the one containing a pred field (for each sample take the last one with pred)
    - videos: for each sample that contributed a pred take its videos[-1] (the segment newly introduced in that turn)
    - sorting: sort in ascending order by the start timestamp of the corresponding video
    """
    if not samples or len(samples) == 0:
        return {}

    first = samples[0]

    # extract system and the first user text message from the first sample
    system_msg = None
    first_user_msg = None
    for conv in first.get("conversations", []):
        if conv.get("from") == "system" and system_msg is None:
            system_msg = conv
        elif (conv.get("from") == "user"
              and conv.get("value") != "<video>"
              and first_user_msg is None):
            first_user_msg = conv

    # collect (pred_conv, video) pairs, to be sorted later
    pairs: list[tuple[dict, str]] = []

    for sample in samples:
        convs = sample.get("conversations", [])
        videos = sample.get("videos", [])

        # find the last assistant message containing pred
        pred_conv = None
        for conv in convs:
            if conv.get("from") == "assistant" and "pred" in conv:
                pred_conv = conv

        if pred_conv is None:
            continue

        # if the assistant turn has no timestamp, use the sample's endpoint_timestamp
        if "timestamp" not in pred_conv and "endpoint_timestamp" in sample:
            pred_conv = {**pred_conv, "timestamp": sample["endpoint_timestamp"]}

        if 'sample_type' in sample:
            pred_conv['sample_type'] = sample['sample_type']

        # the video segment newly introduced in this turn (the last video in the cumulative conversation is this turn's new segment)
        video = videos[-1] if videos else ""
        pairs.append((pred_conv, video))

    # sort in ascending order by (start, end) timestamp, breaking ties on end when start is equal
    pairs.sort(key=lambda p: video_timestamps(p[1]))

    merged_convs = []
    if first_user_msg:
        merged_convs.append(first_user_msg)
    merged_convs.extend(conv for conv, _ in pairs)

    gen_args = None

    # reuse the metadata fields of the first sample, replacing conversations.
    # do not save videos: there are too many 1s chunks for a single video, so no need to keep a video list after merging.
    merged = {k: v for k, v in first.items() if k not in ("conversations", "videos",
                                                            "endpoint_timestamp", "sample_type",
                                                            "inferred_goal", "inferred_knowledge")}
    merged["conversations"] = merged_convs
    if gen_args is not None:
        merged["gen_args"] = gen_args
    return merged


def run_merge(args):
    """The main() of the original tools/merge_infer_conv.py; the argparse part is moved to this file's main().

    args fields: model, input_root, output_root
    Change: merged samples are now returned directly in memory for stage 3/4 to consume;
    when output_root is None the jsonl is no longer written to disk (if a path is passed it still writes
    {output_root}/{model}/{model}.jsonl, with line-by-line identical content).
    Returns: {model_name: [merged_sample, ...]}
    """
    model_name = args.model

    INFERENCE_DIR = args.input_root
    OUTPUT_ROOT = args.output_root

    print(f"INFERENCE_DIR: {INFERENCE_DIR}")
    if OUTPUT_ROOT:
        print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    if model_name:
        model_names = [model_name]
    else:
        inference_root = Path(INFERENCE_DIR)
        model_names = sorted(
            d.name for d in inference_root.iterdir() if d.is_dir()
        )
        print(f"No model specified, found {len(model_names)} models: {model_names}\n")

    all_merged: dict[str, list[dict]] = {}
    for model_name in model_names:
        inference_dir = Path(INFERENCE_DIR) / model_name
        json_files = sorted(inference_dir.rglob("*.json"))
        print(f"===== Model: {model_name} =====")
        print(f"Found {len(json_files)} inference files in total, source directory: {inference_dir}\n")

        # write the jsonl only when output_root is set; otherwise keep in memory
        out = None
        jsonl_path = None
        if OUTPUT_ROOT:
            OUTPUT_DIR = f"{OUTPUT_ROOT}/{model_name}/"
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            jsonl_path = Path(OUTPUT_DIR) / f"{model_name}.jsonl"
            out = open(jsonl_path, "w", encoding="utf-8")

        merged_list: list[dict] = []
        total_pred = 0
        n_written = 0
        for json_file in tqdm.tqdm(json_files, desc=f"Merging jsons for {model_name} ..."):
            if json_file.stat().st_size == 0:
                continue

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    samples = json.load(f)
            except:
                print(f'{json_file} failed to read')
                continue

            if not samples:
                continue

            # infer.py has already merged each video into a dict; if what we got is a per-turn list, do a fallback merge once more
            merged = samples if isinstance(samples, dict) else merge_samples(samples)

            pred_count = sum(
                1 for conv in merged.get("conversations", [])
                if conv.get("from") == "assistant" and "pred" in conv
            )
            total_pred += pred_count

            if out is not None:
                out.write(json.dumps(merged, ensure_ascii=False) + "\n")
            merged_list.append(merged)
            n_written += 1

        if out is not None:
            out.close()
            print(f"\nTotal: {n_written} videos / {total_pred} preds, saved to: {jsonl_path}\n")
        else:
            print(f"\nTotal: {n_written} videos / {total_pred} preds (kept in memory only, not written to disk)\n")

        all_merged[model_name] = merged_list

    return all_merged


# ===========================================================================
# Stage 2: bipartite-matching Precision / Recall / F1
# (copied verbatim from score/metric_match_final.py)
#
# Streaming Video Captioning Evaluation Metrics
#
# Do min-weight bipartite matching between the model generation (gen) and the reference (ref), then compute Precision/Recall/F1.
#
# Flow: segment_by_gt_windows(segment by GT time windows) -> find_match(match within each segment)
#       -> merge_match_results -> compute_precision_recall_f1
#
# Match cost = text cost + distance cost:
#   - text cost: when both sides talk use the sentence-transformers semantic cost (1-|cos_sim|);
#              one side silent is 1; both sides silent is 0.
#   - distance cost: Gaussian decay 1 - exp(-sigma*|i-j|^2) (sigma=0.01).
#
# Match classification: matched(hit) / missed(missed detection) / redundant(redundant false alarm).
# Metrics: two sets of Precision/Recall/F1 — hard(filtered by semantic_score_threshold) and soft(using the
#      semantic score as a continuous weight).
# ===========================================================================

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FrameOutput:
    """Per-frame output from the StreamInferenceRunner."""

    gen: str
    """The generated text output."""
    ref: str | None = None
    """The reference text output."""
    image: Image.Image | None = None
    """The frame image. Useful for creating annotated videos."""
    text_inputs: list[tuple[str, str]] | None = None
    """The input texts for the current frame in the format (role, text)."""
    frame_idx_in_stream: int | None = None
    """The index of the frame in the video."""
    timestamp_in_stream: float | None = None
    """The timestamp of the frame in seconds."""

    def to_dict(self, ignore_keys: str | list[str] = "image") -> dict:
        ret = asdict(self)
        if ignore_keys:
            if isinstance(ignore_keys, str):
                ignore_keys = [ignore_keys]
            for k in ignore_keys:
                ret.pop(k, None)
        return ret


@dataclass
class MatchResult:
    matched: list[tuple[FrameOutput, FrameOutput]]
    missed: list[FrameOutput]
    redundant: list[FrameOutput]
    match_costs: list[float]
    semantic_scores: list[float]

    @classmethod
    def from_json(cls, d: dict) -> "MatchResult":
        return cls(
            matched=[(FrameOutput(**g), FrameOutput(**r)) for g, r in d["matched"]],
            missed=[FrameOutput(**m) for m in d["missed"]],
            redundant=[FrameOutput(**m) for m in d["redundant"]],
            match_costs=d["match_costs"],
            semantic_scores=d["semantic_scores"],
        )

    def to_json(self) -> dict:
        return {
            "matched": [(g.to_dict(), r.to_dict()) for g, r in self.matched],
            "missed": [m.to_dict() for m in self.missed],
            "redundant": [m.to_dict() for m in self.redundant],
            "match_costs": self.match_costs,
            "semantic_scores": self.semantic_scores,
        }

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_semantic_match_cost(
    strings: list[tuple[str, str]], model: sbert.SentenceTransformer, **kwargs
) -> torch.Tensor:
    """Semantic text similarity cost = 1 - |cos_sim(emb_a, emb_b)|."""
    all_strings = [s for pair in strings for s in pair]
    embeddings = model.encode(
        all_strings, convert_to_tensor=True, show_progress_bar=False, **kwargs
    )
    embeddings = embeddings.view(len(strings), 2, -1)
    cos_sim = sbert.util.pairwise_cos_sim(embeddings[:, 0], embeddings[:, 1])
    return 1 - cos_sim.abs()


def get_text_match_cost(
    eval_outputs: list[FrameOutput],
    sts_model: sbert.SentenceTransformer | None,
    no_talk_str: str = "",
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen_talk_times = torch.tensor(
        [-1.0 if o.gen == no_talk_str else 1.0 for o in eval_outputs]
    )
    ref_talk_times = torch.tensor(
        [-1.0 if o.ref == no_talk_str else 1.0 for o in eval_outputs]
    )
    cost = -gen_talk_times[None].T @ ref_talk_times[None]
    cost = (cost + 1) / 2

    if sts_model is not None:
        cmp_ids = []
        cmp_texts = []
        for i, fi in enumerate(eval_outputs):
            gen_txt = fi.gen
            if gen_txt != no_talk_str:
                for j, fj in enumerate(eval_outputs):
                    ref_txt = fj.ref
                    if ref_txt != no_talk_str:
                        cmp_ids.append((i, j))
                        cmp_texts.append((gen_txt, ref_txt))

        if cmp_texts:
            sem_costs = get_semantic_match_cost(cmp_texts, sts_model, **kwargs)
            for (i, j), c in zip(cmp_ids, sem_costs):
                cost[i, j] = c

    return cost, gen_talk_times


def get_gaussian_distance_cost(
    h: int, w: int, sigma: float = 0.01
) -> torch.Tensor:
    """Gaussian decay distance cost: 1 - exp(-sigma * dist^2).

    Returns a (h, w) tensor where entry (i, j) encodes how far apart
    frame i and frame j are, with smooth decay instead of a hard cutoff.
    """
    dist = torch.arange(1, h + 1)[:, None] - torch.arange(1, w + 1)
    dist = dist.float()
    return 1.0 - torch.exp(-sigma * dist ** 2)


# ---------------------------------------------------------------------------
# Bipartite matching
# ---------------------------------------------------------------------------

def find_match(
    eval_outputs: list[FrameOutput],
    sts_model: sbert.SentenceTransformer | None,
    gaussian_sigma: float = 0.01,
    no_talk_str: str = "",
    debug: bool = False,
    **kwargs,
) -> MatchResult:
    """Find optimal gen-to-ref matching via min-weight bipartite matching.

    Uses Gaussian decay distance cost: 1 - exp(-sigma * dist^2) to softly
    penalise temporally distant matches instead of a hard window cutoff.

    Returns a MatchResult containing matched/missed/redundant classifications
    and per-pair semantic scores.
    """
    match_cost, gen_talk_ids = get_text_match_cost(
        eval_outputs,
        sts_model,
        no_talk_str=no_talk_str,
        **kwargs,
    )

    dist_cost = get_gaussian_distance_cost(
        *match_cost.shape, sigma=gaussian_sigma
    )

    total_cost = match_cost + dist_cost

    gen_talk_pos_mask = gen_talk_ids == 1
    gen_talk_indices = gen_talk_pos_mask.nonzero().flatten()

    gen_to_ref_match: dict[int, int] = {}
    ref_be_matched: set[int] = set()

    if len(gen_talk_indices) > 0:
        gen_to_ref_costs = total_cost[gen_talk_pos_mask]
        dense = gen_to_ref_costs.numpy()
        dense = np.where(dense == 0.0, 1e-10, dense)
        gen_to_ref_costs_csr = csr_matrix(dense)
        try:
            idx_in_gen_talk, idx_in_ref = min_weight_full_bipartite_matching(
                gen_to_ref_costs_csr
            )
            idx_in_gen = gen_talk_indices[idx_in_gen_talk].numpy()
            gen_to_ref_match = {i: j for i, j in zip(idx_in_gen, idx_in_ref)}
            ref_be_matched = set(idx_in_ref)
        except ValueError:
            pass

    if debug:
        print("match_cost")
        print(match_cost)
        print("dist_cost")
        print(dist_cost)
        print("total_cost")
        print(total_cost)
        for i, j in zip(idx_in_gen, idx_in_ref):
            print(f"gen {i}: {eval_outputs[i].gen}")
            print(f"-> ref {j}: {eval_outputs[j].ref}")
            t, m, d = total_cost[i, j], match_cost[i, j], dist_cost[i, j]
            print(f"   total_cost: {t:.3f}, match_cost: {m:.3f}, dist_cost: {d:.3f}")

    matched, missed, redundant = [], [], []
    match_costs, semantic_scores = [], []
    for i, f in enumerate(eval_outputs):
        if i in gen_to_ref_match:
            ref_frame = eval_outputs[gen_to_ref_match[i]]
            if ref_frame.ref != no_talk_str:
                matched.append((f, ref_frame))
                match_costs.append(total_cost[i, gen_to_ref_match[i]].item())
                semantic_scores.append(1 - match_cost[i, gen_to_ref_match[i]].item())
            else:
                redundant.append(f)
        if f.ref != no_talk_str and i not in ref_be_matched:
            missed.append(f)

    return MatchResult(
        matched=matched,
        missed=missed,
        redundant=redundant,
        match_costs=match_costs,
        semantic_scores=semantic_scores,
    )


# ---------------------------------------------------------------------------
# Segment by GT time windows
# ---------------------------------------------------------------------------

def segment_by_gt_windows(
    eval_outputs: list[FrameOutput],
    no_talk_str: str = "",
) -> list[list[FrameOutput]]:
    """Split eval_outputs into segments bounded by midpoints of consecutive
    non-silent ref timestamps.

    Each segment contains exactly one non-silent ref and all surrounding
    frames up to the midpoint boundaries with its neighbours.
    """
    ref_times: list[tuple[int, float]] = []
    for i, o in enumerate(eval_outputs):
        if o.ref and o.ref != no_talk_str:
            t = o.timestamp_in_stream if o.timestamp_in_stream is not None else o.frame_idx_in_stream
            if t is not None:
                ref_times.append((i, t))

    if not ref_times:
        return [eval_outputs]

    mid_times = [
        (ref_times[k][1] + ref_times[k + 1][1]) / 2
        for k in range(len(ref_times) - 1)
    ]

    cut_indices = [0]
    for mid in mid_times:
        for i, o in enumerate(eval_outputs):
            t = o.timestamp_in_stream if o.timestamp_in_stream is not None else o.frame_idx_in_stream
            if t is not None and t >= mid:
                cut_indices.append(i)
                break
    cut_indices.append(len(eval_outputs))

    segments = []
    for k in range(len(cut_indices) - 1):
        seg = eval_outputs[cut_indices[k]:cut_indices[k + 1]]
        if seg:
            segments.append(seg)
    return segments


# ---------------------------------------------------------------------------
# Merge / segmented matching
# ---------------------------------------------------------------------------

def merge_match_results(results: list[MatchResult]) -> MatchResult:
    """Merge multiple MatchResults into one."""
    return MatchResult(
        matched=[p for mr in results for p in mr.matched],
        missed=[m for mr in results for m in mr.missed],
        redundant=[r for mr in results for r in mr.redundant],
        match_costs=[c for mr in results for c in mr.match_costs],
        semantic_scores=[s for mr in results for s in mr.semantic_scores],
    )


def find_match_by_segments(
    eval_outputs: list[FrameOutput],
    sts_model: sbert.SentenceTransformer | None,
    no_talk_str: str = "",
    debug: bool = False,
    **kwargs,
) -> tuple[MatchResult, list[MatchResult]]:
    """Segment eval_outputs by GT time windows, run find_match per segment,
    and return (merged_result, per_segment_results)."""
    segments = segment_by_gt_windows(eval_outputs, no_talk_str)

    seg_results: list[MatchResult] = []
    for i, seg in enumerate(segments):
        if debug:
            print(f"\n--- Segment {i} ({len(seg)} frames) ---")
        mr = find_match(
            seg,
            sts_model=sts_model,
            no_talk_str=no_talk_str,
            debug=debug,
            **kwargs,
        )
        seg_results.append(mr)

    merged = merge_match_results(seg_results)
    return merged, seg_results


# ---------------------------------------------------------------------------
# Precision / Recall / F1
# ---------------------------------------------------------------------------

def compute_precision_recall_f1(
    match_result: MatchResult | dict,
    semantic_score_threshold: float = 0.5,
    fps: int = 2,
) -> dict:
    """Compute precision / recall / F1 / jaccard and related metrics from a
    MatchResult (or its JSON dict representation).

    Parameters
    ----------
    match_result : MatchResult or dict
        Output of ``find_match`` (or its ``.to_json()`` dict).
    semantic_score_threshold : float
        Matched pairs with semantic score <= this are treated as mismatches.
    fps : int
        Frames per second, used to convert frame-index differences to seconds.

    Returns
    -------
    dict  with keys: jaccard_index, missing_rate, redundant_rate,
          semantic_score, time_diff, precision, recall, F1,
          num_matched, num_mismatched, num_missed, num_redundant.
    """
    if isinstance(match_result, MatchResult):
        mr = match_result.to_json()
    else:
        mr = match_result
    metrics: dict = {}

    # Soft metrics: no threshold filtering; use semantic scores as continuous weights
    sem_scores = np.array(mr["semantic_scores"])
    # Soft num_matched: the sum of the semantic scores
    soft_num_matched = np.sum(sem_scores)
    # keep the original denominator unchanged (used to measure the total quantity)
    denom_miss = len(mr["matched"]) + len(mr["missed"])
    denom_red = len(mr["matched"]) + len(mr["redundant"])
    soft_precision = soft_num_matched / denom_red if denom_red > 0 else 0
    soft_recall = soft_num_matched / denom_miss if denom_miss > 0 else 0
    metrics["soft_precision"] = soft_precision
    metrics["soft_recall"] = soft_recall
    if soft_precision + soft_recall == 0:
        soft_f1 = 0
    else:
        soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall)
    metrics["soft_f1"] = soft_f1

    matched_pairs = mr["matched"]
    sem_scores = mr["semantic_scores"]
    missed = mr["missed"]
    redundant = mr["redundant"]

    num_matched_before_filter = len(matched_pairs)
    num_missed = len(missed)
    num_redundant = len(redundant)

    gen_ref_pairs = {}
    time_diffs = []
    for idx, ((g, r), s) in enumerate(zip(matched_pairs, sem_scores)):
        if s > semantic_score_threshold:
            gidx = g.get("frame_idx_in_stream") if isinstance(g, dict) else g.frame_idx_in_stream
            ridx = r.get("frame_idx_in_stream") if isinstance(r, dict) else r.frame_idx_in_stream
            gen_text = g.get("gen") if isinstance(g, dict) else g.gen
            ref_text = r.get("ref") if isinstance(r, dict) else r.ref
            uid = f"{idx}_{gidx}<>{ridx}"
            gen_ref_pairs[uid] = (gen_text, ref_text, s)
            if gidx is not None and ridx is not None:
                time_diffs.append(abs(gidx - ridx) / fps)

    num_matched = len(gen_ref_pairs)
    num_mismatched = num_matched_before_filter - num_matched
    num_total = num_matched_before_filter + num_missed + num_redundant

    metrics["jaccard_index"] = num_matched / num_total if num_total > 0 else 0.0

    denom_miss = num_matched_before_filter + num_missed
    metrics["missing_rate"] = num_missed / denom_miss if denom_miss > 0 else 0.0

    denom_red = num_matched_before_filter + num_redundant
    metrics["redundant_rate"] = num_redundant / denom_red if denom_red > 0 else 0.0

    matched_semscores = [s for s in sem_scores if s >= semantic_score_threshold]
    metrics["semantic_score"] = float(np.mean(matched_semscores)) if matched_semscores else 0.0
    metrics["time_diff"] = float(np.mean(time_diffs)) if time_diffs else 0.0

    metrics["num_matched"] = num_matched
    metrics["num_mismatched"] = num_mismatched
    metrics["num_missed"] = num_missed
    metrics["num_redundant"] = num_redundant

    p = num_matched / denom_red if denom_red > 0 else 0.0
    r = num_matched / denom_miss if denom_miss > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    metrics["precision"] = p
    metrics["recall"] = r
    metrics["F1"] = f1

    return metrics


def run_metric_match(args):
    """The __main__ block of the original score/metric_match_final.py; the argparse part is moved to this file's main().

    args fields: model_name, subsetset, input_root, eval_root
    Changes:
      - the summary metrics are written to disk at {eval_root}/{model}/metrics_summary.json (the original logic only printed them),
        and are also used as a return value for main() to write the final evaluation_results.txt
      - metrics_info (the matched pairs details) is no longer written to disk, but passed directly as a return value
        in memory to stage 4 (run_score_v2_matched)
    Returns: (final_ourput summary dict, all_metrics_info dict)
    """
    from pprint import pprint

    SILENT_STRINGS = {"silent", "<|silent|>"}

    torch.set_printoptions(precision=3, sci_mode=False)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sts_model = sbert.SentenceTransformer(model_name, device=device)

    model_name = args.model_name

    input_root = args.input_root
    all_json_files = glob.glob(os.path.join(input_root, model_name, '**/*.json'), recursive=True)
    all_metrics = {}
    all_metrics_info = {}

    filename_to_subset = {}
    if args.subsetset:
        with open(args.subsetset, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        for line in lines:
            parts = line.split('/')
            subset = parts[0] if len(parts) > 1 else 'unknown'
            filename = parts[-1]
            filename_to_subset[filename] = subset
        subset_files = set(filename_to_subset.keys())
        all_json_files = [file for file in all_json_files if os.path.basename(file) in subset_files]

    print(f"Total json files: {len(all_json_files)}")
    for input_json in tqdm.tqdm(all_json_files, desc=f"Processing {model_name}, ({len(all_json_files)} files)"):
        video_id = os.path.basename(input_json)

        test_outputs: list[FrameOutput] = []
        with open(input_json, 'r') as f:
            data = json.load(f)
        conversations = data['conversations']
        for conv in conversations:
            if 'pred' not in conv:
                continue
            gen = "" if conv['pred'].lower().strip() in SILENT_STRINGS else conv['pred']
            ref = "" if conv['value'].lower().strip() in SILENT_STRINGS else conv['value']
            test_outputs.append(FrameOutput(
                gen=gen,
                ref=ref,
                frame_idx_in_stream=conv['timestamp'],
                timestamp_in_stream=float(conv['timestamp']),
            ))
        print(f"Total frames: {len(test_outputs)}")

        # Step 1: segmented bipartite matching
        merged_mr, seg_results = find_match_by_segments(
            test_outputs,
            sts_model=sts_model,
            gaussian_sigma=0.01,
            batch_size=64,
            debug=False,
        )

        # Step 2: per-segment metrics
        for i, mr in enumerate(seg_results):
            m = compute_precision_recall_f1(mr, semantic_score_threshold=0.5, fps=1)

        # Step 3: merged metrics
        torch.set_printoptions(profile="full", sci_mode=False, precision=3)

        final_metrics = compute_precision_recall_f1(
            merged_mr, semantic_score_threshold=0.5, fps=1
        )
        final_metrics['info'] = merged_mr.to_json()

        keeped_metrics = {k: v for k, v in final_metrics.items() if k in ['soft_precision', 'soft_recall', 'soft_f1', 'precision', 'recall', 'F1']}
        all_metrics[video_id] = keeped_metrics
        all_metrics_info[video_id] = final_metrics

    ## average metrics
    def _compute_avg_metrics(metrics_dict):
        return {
            'soft': {
                'avg_soft_precision': float(np.mean([v['soft_precision'] for v in metrics_dict.values()])),
                'avg_soft_recall': float(np.mean([v['soft_recall'] for v in metrics_dict.values()])),
                'avg_soft_f1': float(np.mean([v['soft_f1'] for v in metrics_dict.values()])),
            },
            'hard': {
                'avg_precision': float(np.mean([v['precision'] for v in metrics_dict.values()])),
                'avg_recall': float(np.mean([v['recall'] for v in metrics_dict.values()])),
                'avg_F1': float(np.mean([v['F1'] for v in metrics_dict.values()])),
            },
            'video_info': {
                'num_videos': len(metrics_dict),
            },
        }

    final_ourput = _compute_avg_metrics(all_metrics)
    final_ourput['model_name'] = model_name

    ## summary metrics: always print; write to disk as metrics_summary.json only in debug mode
    print("===== Averaged metrics =====")
    pprint(final_ourput)
    if getattr(args, "save_debug", True):
        eval_dir = os.path.join(args.eval_root, model_name)
        os.makedirs(eval_dir, exist_ok=True)
        summary_file = os.path.join(eval_dir, 'metrics_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(final_ourput, f, indent=4, ensure_ascii=False)
        print(f"Summary saved to {summary_file}")

    ## metrics_info (matched pairs details) is not written to disk, returned in memory to stage 4
    return final_ourput, all_metrics_info


# ===========================================================================
# Stage 3 / 4 shared part: LLM-judge proactive evaluation
# (copied verbatim from score/score_v2.py and score/score_v2_matched.py;
#   the definitions that are byte-for-byte identical in both files are kept only once)
# ===========================================================================

# --------------------------------------------------
# utility functions
# --------------------------------------------------

## the set from score_v2.py (has the extra "<|silent|>" and "silent." compared to the matched version)
PLACEHOLDERS_V2 = {
    "", "silent", "<NO_INFORMATION>", "<SILENT>", "<|silent|>", "silent.",
    "Alright, I'll send you a reminder then.",
    "Got it, I'll let you know at that moment.",
    "got it, i'll let you know.",
    "Understood, I will remind you when the time comes.",
    "收到，我会留意的。",
    "没问题，到时候提醒你。",
    "好的，到时候我会提醒你。",
    "没问题，到时候我会告诉你。",
    "No problem, I'll remind you then.",
    "Okay, I will alert you when it happens.",
    "I will make sure to remind you at that time.",
    "好的，那到时候我会提醒你。",
    "好的，到时候我会发提醒给你。",
    "Sure, I will let you know at that time.",
    "Noted, expect a reminder from me then.",
    "Ok, I will remind you then.",
    "Certainly, I will provide the reminder then.",
    "No problem, I'll remind you then.",
    "ok", "okay", "sure", "yes", "收到", "好的", "明白了", "got it, i will notify you at that moment."
}

## the set from score_v2_matched.py
PLACEHOLDERS_MATCHED = {
    "", "silent", "<NO_INFORMATION>", "<SILENT>",
    "Alright, I'll send you a reminder then.",
    "Got it, I'll let you know at that moment.",
    "got it, i'll let you know.",
    "Understood, I will remind you when the time comes.",
    "收到，我会留意的。",
    "没问题，到时候提醒你。",
    "好的，到时候我会提醒你。",
    "没问题，到时候我会告诉你。",
    "No problem, I'll remind you then.",
    "Okay, I will alert you when it happens.",
    "I will make sure to remind you at that time.",
    "好的，那到时候我会提醒你。",
    "好的，到时候我会发提醒给你。",
    "Sure, I will let you know at that time.",
    "Noted, expect a reminder from me then.",
    "Ok, I will remind you then.",
    "Certainly, I will provide the reminder then.",
    "No problem, I'll remind you then.",
    "ok", "okay", "sure", "yes", "收到", "好的", "明白了", "got it, i will notify you at that moment."
}

## the currently active set, switched at the run_score_v2 / run_score_v2_matched entry points
PLACEHOLDERS = PLACEHOLDERS_V2

def is_placeholder(text: str) -> bool:
    return text.strip().lower() in set(x.strip().lower() for x in PLACEHOLDERS)

# --------------------------------------------------
# LLM Judger (optional)
# --------------------------------------------------
def get_llm_prompt(question: str, model_output: str, reference_answer: str) -> str:
    return get_prompt(
        question=question,
        model_output=model_output,
        reference_answer=reference_answer
    )

class LLMJudger:
    def __init__(self, llm_client):
        self.llm = llm_client

    def judge(self, question: str, model_output: str, reference: str, retries=3) -> dict:
        inputs = [("user", get_llm_prompt(question, model_output, reference))]
        for _ in range(retries):
            try:
                result = self.llm.generate(inputs)
                resp = (result[0] if isinstance(result, list) else result).strip()
                parsed = json.loads(repair_json(resp))
                score = max(0.0, min(5.0, float(parsed["score"])))
                explanation = str(parsed.get("explanation", ""))
                return {"explanation": explanation, "score": score}
            except Exception:
                continue
        return {"explanation": "Judger LLM parsing error", "score": 0}

# --------------------------------------------------
# evaluate a single sample (conversations format)
# --------------------------------------------------
def evaluate_sample(
    sample: Dict[str, Any],
    llm_judger: Optional[LLMJudger],
    pc_time_window: float = 2.0,
) -> List[Dict[str, Any]]:
    """Categorize each assistant frame as CorrectSilent / NoResponse / FalseAlarm /
    PartlyCorrect and score it.

    pc_time_window: a non-silent prediction within +/- pc_time_window seconds of a
    non-silent GT timestamp is paired with that GT and counted as PartlyCorrect
    (LLM-judged against the paired GT), instead of the strict same-frame rule that
    would produce a NoResponse + FalseAlarm double penalty for near-miss timing.
    If several preds fall in one GT's window, the nearest is the PartlyCorrect match
    and the extra preds are ignored (redundant restatements near a real event, not
    false alarms). A pred only counts as FalseAlarm when no non-silent GT lies within
    its window. With pc_time_window == 0 the behavior is identical to the original
    per-frame logic.
    """
    sample_id = sample.get("video_id", sample.get("id", ""))
    scene_type = sample.get("type", "unknown")

    # take the first user message as the question
    question = ""
    for msg in sample["conversations"]:
        if msg["from"] == "user":
            question = msg["value"]
            break

    # collect assistant frames: (timestamp, ground_truth, pred)
    frames: List[Dict[str, Any]] = []
    for msg in sample["conversations"]:
        if msg["from"] != "assistant":
            continue
        frames.append({
            "timestamp": msg["timestamp"],
            "gt": msg["value"],
            "pred": msg.get("pred", ""),
        })

    # ---- pair non-silent preds with non-silent GTs within pc_time_window ----
    # greedy nearest-first: each pred / GT event is consumed at most once;
    # same-frame pairs have |dt| = 0 and always win first.
    pred_idx = [i for i, f in enumerate(frames) if not is_placeholder(f["pred"])]
    gt_idx = [j for j, f in enumerate(frames) if not is_placeholder(f["gt"])]

    candidates = []
    for i in pred_idx:
        for j in gt_idx:
            dt = abs(float(frames[i]["timestamp"]) - float(frames[j]["timestamp"]))
            if dt <= pc_time_window:
                candidates.append((dt, i, j))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))

    pred_to_gt: Dict[int, int] = {}
    gt_matched: set = set()
    for dt, i, j in candidates:
        if i in pred_to_gt or j in gt_matched:
            continue
        pred_to_gt[i] = j
        gt_matched.add(j)

    # preds that have at least one non-silent GT within the window (even if that GT was
    # matched to a closer pred). Such extra preds are redundant restatements of a real
    # event, not false alarms, so they are ignored rather than penalised.
    pred_has_gt_in_window = {i for _, i, _ in candidates}

    # ---- emit one record per event ----
    results = []
    for i, f in enumerate(frames):
        ground_truth, pred, timestamp = f["gt"], f["pred"], f["timestamp"]
        pred_silent = is_placeholder(pred)
        gt_silent = is_placeholder(ground_truth)

        explanation = ""
        if not pred_silent:
            if i in pred_to_gt:
                # matched to a GT within the window -> PartlyCorrect, judged against that GT
                matched_gt = frames[pred_to_gt[i]]["gt"]
                if llm_judger is None:
                    score_100 = 0.0
                    category = "Error (no LLM)"
                else:
                    raw = llm_judger.judge(question, pred, matched_gt)
                    score_100 = raw["score"] * 20.0
                    explanation = raw.get("explanation", "")
                    category = "PartlyCorrect"
                ground_truth = matched_gt
            elif i in pred_has_gt_in_window:
                # redundant extra response near a real GT that was matched by a closer
                # pred -> ignore (not a false alarm, not double-counted)
                continue
            else:
                # spoke with no GT anywhere in the window -> genuine false alarm
                score_100 = 0.0
                category = "FalseAlarm"
        else:
            if gt_silent:
                score_100 = 100.0
                category = "CorrectSilent"
            elif i in gt_matched:
                # this GT was answered by a nearby prediction (already counted as PartlyCorrect)
                continue
            else:
                score_100 = 0.0
                category = "NoResponse"

        results.append({
            "sample_id": sample_id,
            "scene_type": scene_type,
            "timestamp": timestamp,
            "answer": ground_truth,
            "response": pred,
            "score": score_100,
            "category": category,
            "explanation": explanation,
        })
    return results

# --------------------------------------------------
# main flow
# --------------------------------------------------
def load_llm(config):
    cls_path = config["class"]
    mod_name, cls_name = cls_path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return LLMJudger(llm_client=cls(**config.get("args", {})))

def worker(samples_chunk, llm_config, pc_time_window: float = 2.0):
    llm = load_llm(llm_config) if llm_config else None
    all_results = []
    for sample in tqdm.tqdm(samples_chunk):
        all_results.extend(evaluate_sample(sample, llm, pc_time_window=pc_time_window))
    return all_results

# --------------------------------------------------
# metric_match results -> samples conversion (only in score_v2_matched.py)
# --------------------------------------------------

def convert_matched_to_samples(
    metrics_data: dict,
    original_jsonl_path: str | None = None,
    original_samples: List[Dict] | None = None,
) -> List[Dict[str, Any]]:
    """Convert the matched pairs of metrics_info into samples consumable by evaluate_sample.

    Each matched pair = [gen_frame, ref_frame]:
      - gen_frame (item1): model output frame, gen_frame["gen"] = model prediction
      - ref_frame (item2): the matched reference frame, ref_frame["ref"] = GT reference
    The empty string "" is replaced with "silent".

    Change: besides reading the original samples from a jsonl file (original_jsonl_path),
    passing in-memory merged samples directly is also supported (original_samples, from stage 1).
    """
    orig_map: Dict[str, Dict] = {}
    if original_samples is not None:
        for s in original_samples:
            vid = s.get("video_id", "")
            orig_map[vid] = s
    elif original_jsonl_path and Path(original_jsonl_path).exists():
        with open(original_jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                s = json.loads(line)
                vid = s.get("video_id", "")
                orig_map[vid] = s

    samples = []
    for video_key, video_data in metrics_data.items():
        info = video_data.get("info", {})
        if not info:
            continue

        orig_sample = None
        for vid, orig in orig_map.items():
            if vid in video_key or video_key in vid:
                orig_sample = orig
                break

        question_text = ""
        scene_type = "unknown"
        if orig_sample:
            scene_type = orig_sample.get("type", "unknown")
            for msg in orig_sample.get("conversations", []):
                if msg["from"] == "user":
                    question_text = msg["value"]
                    break

        conversations: list[dict] = [
            {"from": "user", "value": question_text, "timestamp": 0}
        ]

        for gen_frame, ref_frame in info.get("matched", []):
            pred = gen_frame["gen"] if gen_frame["gen"] else "silent"
            ref = ref_frame["ref"] if ref_frame["ref"] else "silent"
            conversations.append({
                "from": "assistant",
                "value": ref,
                "pred": pred,
                "timestamp": gen_frame.get("timestamp_in_stream", 0),
            })

        samples.append({
            "video_id": video_key,
            "type": scene_type,
            "conversations": conversations,
        })

    return samples


# ===========================================================================
# Stage 3: LLM-judge proactive scoring on the raw jsonl
# (copied verbatim from the main() of score/score_v2.py)
# ===========================================================================

def run_score_v2(args, samples=None):
    """args fields: model_name, model_output, pred_model, output_dir, config,
    workers, collections
    Change: supports directly passing in-memory merged samples (from stage 1);
    when samples is None the original logic of loading from a jsonl file is kept.
    """
    global PLACEHOLDERS
    PLACEHOLDERS = PLACEHOLDERS_V2   # use the placeholder set from score_v2.py

    save_debug = bool(getattr(args, "save_debug", True))
    output_dir = Path(args.output_dir or f"output/eval/{args.pred_model}")
    if save_debug:
        output_dir.mkdir(exist_ok=True, parents=True)
        print(output_dir, '====')

    print(f"[1/4] Loading config: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    llm_config = config.get("judger")
    judger_info = llm_config.get("args", {}).get("model", llm_config.get("class", "not configured")) if llm_config else "not configured"
    print(f"      LLM Judger: {judger_info}")

    if samples is None:
        print(f"[2/4] Loading model outputs: {args.model_output}")
        with open(args.model_output) as f:
            samples = [json.loads(line) for line in f if line.strip()]
    else:
        print(f"[2/4] Using in-memory merged samples (produced by stage 1, not written to disk)")
    print(f"      {len(samples)} samples in total, evaluating in parallel with {args.workers} threads")

    pc_time_window = float(getattr(args, "pc_time_window", 2.0))
    print(f"      pc_time_window={pc_time_window}s (preds within +/- this window of a GT count as PartlyCorrect)")

    print(f"[3/4] Starting evaluation (model={args.model_name})...")
    chunks = [samples[i::args.workers] for i in range(args.workers)]
    all_results = Parallel(n_jobs=args.workers, backend='threading')(
        delayed(worker)(chunk, llm_config, pc_time_window) for chunk in chunks
    )
    all_results = [item for sublist in all_results for item in sublist]
    print(f"      Evaluation finished, {len(all_results)} evaluation records in total")

    print(f"[4/4] Saving results...")
    df = pd.DataFrame(all_results)
    if save_debug:
        db_path = output_dir / f"{args.collections}.db"
        df.to_sql(f"{args.model_name}", sqlite3.connect(db_path), if_exists="replace", index=False)
        print(f"      Details saved to: {db_path}  (table={args.model_name})")

    total = len(df)
    summary = {
        "model_name": args.model_name,
        "#samples": total,
        "final_score": round(df["score"].mean(), 1),
    }

    # category distribution (proportion + mean score), only compute the overall distribution merged across all scenes, no longer broken down by type
    def show_category_distribution(dataframe, subset_name, summary_dict):
        if len(dataframe) == 0:
            return
        for cat in dataframe["category"].unique():
            sub = dataframe[dataframe["category"] == cat]
            percent = round(len(sub) / len(dataframe) * 100, 1)
            score = round(sub["score"].mean(), 1)
            summary_dict[f"{cat}({subset_name})"] = f"{percent}%({score})"

    show_category_distribution(df, "all", summary)

    if save_debug:
        csv_path = output_dir / f"{args.collections}.csv"
        out_df = pd.DataFrame([summary])
        if csv_path.exists():
            out_df = pd.concat([pd.read_csv(csv_path), out_df], ignore_index=True)
        out_df.round(1).to_csv(csv_path, index=False)
        print(f"      Summary saved to: {csv_path}")

    print(f"\n✅ Done! final_score={summary['final_score']}")
    return summary


# ===========================================================================
# Stage 4: LLM-judge proactive scoring on the bipartite-matched pairs
# (copied verbatim from the main() of score/score_v2_matched.py)
# ===========================================================================

def run_score_v2_matched(args, orig_samples=None, metrics_data=None):
    """args fields: model_name, model_output, pred_model, metrics_info,
    output_dir, config, workers, collections
    Changes:
      - metrics_data: the return value of stage 2 (run_metric_match) is passed directly in memory,
        no longer relying on the on-disk metrics_info.json; when None the original logic of reading the file is kept.
      - orig_samples: the in-memory merged samples from stage 1, used to obtain question/scene_type;
        when None the original logic of reading the jsonl is kept.
    """
    global PLACEHOLDERS
    PLACEHOLDERS = PLACEHOLDERS_MATCHED   # use the placeholder set from score_v2_matched.py

    save_debug = bool(getattr(args, "save_debug", True))
    output_dir = Path(args.output_dir or f"output/eval/{args.pred_model}")
    if save_debug:
        output_dir.mkdir(exist_ok=True, parents=True)

    print(f"[1/4] Loading config: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    llm_config = config.get("judger")
    judger_info = llm_config.get("args", {}).get("model", llm_config.get("class", "not configured")) if llm_config else "not configured"
    print(f"      LLM Judger: {judger_info}")

    if metrics_data is None and args.metrics_info and Path(args.metrics_info).exists():
        print(f"[2/4] Loading metric_match matching results: {args.metrics_info}")
        with open(args.metrics_info) as f:
            metrics_data = json.load(f)

    if metrics_data is not None:
        if orig_samples is not None:
            print(f"[2/4] Using in-memory metric_match matching results + merged samples (question/scene_type)")
            samples = convert_matched_to_samples(metrics_data, original_samples=orig_samples)
        else:
            jsonl_path = args.model_output if args.model_output and Path(args.model_output).exists() else None
            if jsonl_path:
                print(f"      Also loading the original JSONL to get question/scene_type: {jsonl_path}")
            else:
                print(f"      Original JSONL not found ({args.model_output}), using default values for question/scene_type")
            samples = convert_matched_to_samples(metrics_data, jsonl_path)
        n_matched = sum(len(d["info"]["matched"]) for d in metrics_data.values() if "info" in d)
        print(f"      {len(samples)} videos in total, {n_matched} matched pairs")
        print(f"      Evaluating in parallel with {args.workers} threads")
    elif orig_samples is not None:
        print(f"[2/4] No metric_match matching results, falling back to the in-memory merged samples")
        samples = orig_samples
        print(f"      {len(samples)} samples in total, evaluating in parallel with {args.workers} threads")
    else:
        print(f"[2/4] Loading model outputs: {args.model_output}")
        with open(args.model_output) as f:
            samples = [json.loads(line) for line in f if line.strip()]
        print(f"      {len(samples)} samples in total, evaluating in parallel with {args.workers} threads")

    print(f"[3/4] Starting evaluation (model={args.model_name})...")
    chunks = [samples[i::args.workers] for i in range(args.workers)]
    # matched pairs are already temporally aligned by stage 2 -> keep strict per-frame rule
    all_results = Parallel(n_jobs=args.workers, backend='threading')(
        delayed(worker)(chunk, llm_config, 0.0) for chunk in chunks
    )
    all_results = [item for sublist in all_results for item in sublist]
    print(f"      Evaluation finished, {len(all_results)} evaluation records in total")

    print(f"[4/4] Saving results...")
    df = pd.DataFrame(all_results)
    if save_debug:
        db_path = output_dir / f"{args.collections}.db"
        df.to_sql(f"{args.model_name}", sqlite3.connect(db_path), if_exists="replace", index=False)
        print(f"      Details saved to: {db_path}  (table={args.model_name})")

    total = len(df)
    summary = {
        "model_name": args.model_name,
        "#samples": total,
        "final_score": round(df["score"].mean(), 1),
    }

    # group by scene type and compute mean scores
    for scene in sorted(df["scene_type"].dropna().unique()):
        subset = df[df["scene_type"] == scene]
        summary[scene] = round(subset["score"].mean(), 1) if len(subset) else 0.0

    # category distribution (proportion + mean score)
    def show_category_distribution(dataframe, subset_name, summary_dict):
        if len(dataframe) == 0:
            return
        for cat in dataframe["category"].unique():
            sub = dataframe[dataframe["category"] == cat]
            percent = round(len(sub) / len(dataframe) * 100, 1)
            score = round(sub["score"].mean(), 1)
            summary_dict[f"{cat}({subset_name})"] = f"{percent}%({score})"

    # category summary merged across all scenes
    show_category_distribution(df, "all", summary)

    # category distribution within each scene
    for scene in sorted(df["scene_type"].dropna().unique()):
        show_category_distribution(df[df["scene_type"] == scene], scene, summary)

    if save_debug:
        csv_path = output_dir / f"{args.collections}.csv"
        out_df = pd.DataFrame([summary])
        if csv_path.exists():
            out_df = pd.concat([pd.read_csv(csv_path), out_df], ignore_index=True)
        out_df.round(1).to_csv(csv_path, index=False)
        print(f"      Summary saved to: {csv_path}")

    print(f"\n✅ Done! final_score={summary['final_score']}")
    return summary


# ===========================================================================
# top-level orchestration (corresponds to the for loop in scripts/score.sh)
# ===========================================================================

class _Tee:
    """stdout is written to both the terminal and the log file (reproduces the `2>&1 | tee xxx.log` in score.sh)."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


@contextmanager
def _tee_stdout(path):
    tee = _Tee(path)
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        yield
    finally:
        sys.stdout = old_stdout
        tee.file.close()


def _stage_banner(model_name: str, stage: str, title: str):
    """separator print between stages, indicating which step is currently running."""
    line = "=" * 78
    print(f"\n{line}")
    print(f">>> [{model_name}] Stage {stage}: {title}")
    print(f"{line}\n", flush=True)


def _write_final_results_txt(path, model_name,
                             match_summary=None, v2_summary=None, v2m_summary=None,
                             save_debug=False):
    """Write the final evaluation results to a txt file, split into two major parts:

    - Temporal Alignment: soft precision / soft recall / soft F1 (stage 2)
                          + score_matched (the LLM-judge score on stage 4 matched pairs)
    - Response Behavior : the proportions (mean scores) of the four categories CS / NR / FA / PC (stage 3)
                          + score (the LLM-judge score on the stage 3 raw stream)
    """
    def fmt(v, nd=4):
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "N/A"

    lines = []
    lines.append("=" * 62)
    lines.append(f"Evaluation Results: {model_name}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 62)

    lines.append("")
    lines.append("[Temporal Alignment]")
    soft = (match_summary or {}).get("soft", {})
    lines.append(f"  soft_precision : {fmt(soft.get('avg_soft_precision'))}")
    lines.append(f"  soft_recall    : {fmt(soft.get('avg_soft_recall'))}")
    lines.append(f"  soft_F1        : {fmt(soft.get('avg_soft_f1'))}")
    score_matched = (v2m_summary or {}).get("final_score")
    lines.append(f"  score_matched  : {fmt(score_matched, nd=1)}   # LLM-judge on matched pairs (0-100)")

    lines.append("")
    lines.append("[Response Behavior]")
    v2 = v2_summary or {}
    def cat(name):
        return v2.get(f"{name}(all)", "N/A")
    lines.append(f"  CorrectSilent (CS) : {cat('CorrectSilent')}")
    lines.append(f"  NoResponse    (NR) : {cat('NoResponse')}")
    lines.append(f"  FalseAlarm    (FA) : {cat('FalseAlarm')}")
    lines.append(f"  PartlyCorrect (PC) : {cat('PartlyCorrect')}")
    lines.append(f"  score              : {fmt(v2.get('final_score'), nd=1)}   # LLM-judge on raw stream (0-100)")
    lines.append("")
    if save_debug:
        lines.append("(format: proportion%(mean score of that category); see the debug/ directory for intermediate artifacts)")
    else:
        lines.append("(format: proportion%(mean score of that category); rerun with --debug to also dump intermediate artifacts)")

    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nFinal results saved to: {path}")
    print(text)


def main():
    parser = argparse.ArgumentParser(
        description="all-in-one scoring pipeline: merge -> metric_match -> score_v2 -> score_v2_matched"
    )
    parser.add_argument("models", type=str, nargs="*", default=None,
                        help="list of model names; if not passed, automatically iterate over all model directories under {infer_root}")
    parser.add_argument("--infer-root", type=str, default="output/infer",
                        help="per-video merged jsons (the output of infer.py)")
    parser.add_argument("--eval-root", type=str, default="output/eval",
                        help="directory for all scoring outputs")
    ## stage 2 (metric_match) options
    parser.add_argument("--subsetset", type=str, default="",
                        help="optional subset txt (list of relative paths, e.g. fitness/0006.json); leave empty to evaluate all files")
    ## stage 3/4 (LLM judge) options, matching the argparse defaults of the original score_v2*.py
    parser.add_argument("--judge-model-name", type=str, default="gemini-3-pro-preview",
                        help="the --model_name of the original score_v2*.py (sqlite table name / model_name column in csv; "
                             "score.sh does not pass this parameter, so it is always the default)")
    parser.add_argument("--config", type=str, default="score/config/stream_config.yaml")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--pc-time-window", type=float, default=2.0,
                        help="stage 3 only: a non-silent prediction within +/- this many seconds of a "
                             "non-silent GT timestamp is paired with it and scored as PartlyCorrect, "
                             "instead of being counted as NoResponse + FalseAlarm; 0 = strict same-frame rule")
    parser.add_argument("--collections", type=str, default="v4.4",
                        help="the collections for stage 3 (db/csv filename)")
    parser.add_argument("--collections-matched", type=str, default="v4.4_matched",
                        help="the collections for stage 4 (db/csv filename)")
    parser.add_argument("--debug", action="store_true",
                        help="also save the intermediate artifacts to {eval_root}/debug/{model}/ "
                             "(metrics_summary.json, per-stage db/csv, score_v2*.log); "
                             "off by default, in which case only the final evaluation_results.txt is written")
    args = parser.parse_args()

    if args.models:
        model_list = args.models
    else:
        infer_root_path = Path(args.infer_root)
        model_list = sorted(d.name for d in infer_root_path.iterdir() if d.is_dir())
        print(f"No model specified, found {len(model_list)} models: {model_list}\n")

    for model_name in model_list:
        print(f"\n######## {model_name} ########")

        ## final results directory: {eval_root}/{model}/evaluation_results.txt
        ## intermediate artifacts directory: {eval_root}/debug/{model}/ (db/csv/log/summary json etc.)
        ##   only created/written when --debug is passed; otherwise everything stays in memory
        save_debug = bool(args.debug)
        eval_dir = f"{args.eval_root}/{model_name}"
        debug_root = f"{args.eval_root}/debug"
        debug_dir = f"{debug_root}/{model_name}"
        os.makedirs(eval_dir, exist_ok=True)
        if save_debug:
            os.makedirs(debug_dir, exist_ok=True)

        # 1) merge per-video jsons -> in-memory samples (the input for stage 3/4)
        _stage_banner(model_name, "1/4", "merge infer jsons -> samples (orig merge_infer_conv)")
        merged_samples = run_merge(argparse.Namespace(
            model=model_name,
            input_root=args.infer_root,
            output_root=None,   # merged jsonl not written to disk, passed in memory only
        ))[model_name]

        # 2) matching-based metrics (Precision / Recall / F1)
        #    read per-video jsons; when --debug the summary is written to disk at debug/metrics_summary.json,
        #    the matched pairs details (metrics_info) are passed in memory to stage 4
        _stage_banner(model_name, "2/4", "matching-based P/R/F1 (orig metric_match_final)")
        match_summary, metrics_info_data = run_metric_match(argparse.Namespace(
            model_name=model_name,
            subsetset=args.subsetset,
            input_root=args.infer_root,
            eval_root=debug_root,
            save_debug=save_debug,
        ))

        # 3) LLM-judge proactive scoring (raw merged samples)
        _stage_banner(model_name, "3/4", "LLM-judge proactive scoring, raw (orig score_v2)")
        with (_tee_stdout(os.path.join(debug_dir, "score_v2.log")) if save_debug else nullcontext()):
            v2_summary = run_score_v2(argparse.Namespace(
                model_name=args.judge_model_name,
                model_output=None,
                pred_model=model_name,
                output_dir=debug_dir,
                config=args.config,
                workers=args.workers,
                collections=args.collections,
                pc_time_window=args.pc_time_window,
                save_debug=save_debug,
            ), samples=merged_samples)

        # 4) LLM-judge proactive scoring on the bipartite-matched pairs
        #    metrics_info comes from the in-memory return value of stage 2 (not written to disk)
        _stage_banner(model_name, "4/4", "LLM-judge proactive scoring, matched (orig score_v2_matched)")
        with (_tee_stdout(os.path.join(debug_dir, "score_v2_matched.log")) if save_debug else nullcontext()):
            v2m_summary = run_score_v2_matched(argparse.Namespace(
                model_name=args.judge_model_name,
                model_output=None,
                pred_model=model_name,
                metrics_info=os.path.join(debug_dir, "metrics_info.json"),
                output_dir=debug_dir,
                config=args.config,
                workers=args.workers,
                collections=args.collections_matched,
                save_debug=save_debug,
            ), orig_samples=merged_samples, metrics_data=metrics_info_data)

        # final results: save only the two major metric categories to txt
        _write_final_results_txt(
            os.path.join(eval_dir, "evaluation_results.txt"),
            model_name,
            match_summary=match_summary,
            v2_summary=v2_summary,
            v2m_summary=v2m_summary,
            save_debug=save_debug,
        )

    print(f"\nscoring done -> {args.eval_root}")


if __name__ == "__main__":
    main()
