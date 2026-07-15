import os
import sys
import json
import base64
import time
import copy
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

sys.path.insert(0, os.path.dirname(__file__))
from tools.llm_utils import GeminiAPIGenerator
from tools.merge_infer_conv import merge_samples

# Video Base64 in-memory cache (thread-safe)
_video_b64_cache: dict[str, str] = {}
_video_b64_lock = threading.Lock()

# ============================================================
# Configuration (modify as needed)
# ============================================================
MAX_RETRIES      = 10    # Maximum retries when API calls fail
RETRY_BASE_DELAY = 10    # Retry wait base (seconds), exponential backoff
REQUEST_INTERVAL = 0.5  # Minimum interval between two adjacent API calls (seconds)
NUM_WORKERS      = 1   # Number of concurrent threads
BACKEND          = "api"  # "api" = cloud Gemini/OpenAI; "local" = local HF model
# ============================================================


def encode_video_b64(video_path: str) -> str:
    with _video_b64_lock:
        if video_path in _video_b64_cache:
            return _video_b64_cache[video_path]
    with open(video_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    with _video_b64_lock:
        _video_b64_cache[video_path] = data
    return data


def build_gemini_messages(conversations: list, videos: list, max_videos: int = 10) -> tuple[str, list[dict]]:
    """
    Convert conversations (excluding the last assistant message) to Gemini messages format.

    Return (system_prompt, messages), where the content of each message in messages is:
    - Plain text: str
    - Video content: list of Gemini parts (inlineData)
    - Mixed content: list of Gemini parts

    Adjacent user messages are merged into one (parts arrays are concatenated).
    """
    system_prompt = ""
    raw_messages: list[dict] = []
    video_idx = 0

    total_video_count = sum(1 for c in conversations if c.get("from") == "user" and c.get("value") == "<video>")
    skip_count = max(0, min(total_video_count, len(videos)) - max_videos)

    for conv in conversations:
        role_from = conv.get("from", "")
        value = conv.get("value", "")

        if role_from == "system":
            system_prompt = value

        elif role_from == "user":
            if value == "<video>":
                if video_idx >= len(videos) or video_idx < skip_count:
                    video_idx += 1
                    continue
                video_path = os.path.join(VIDEO_BASE_DIR, videos[video_idx])
                video_idx += 1
                if BACKEND == "local":
                    # The local backend uses file paths directly to avoid base64 encoding/decoding overhead
                    raw_messages.append({
                        "role": "user",
                        "content": [{"videoPath": video_path}],
                    })
                else:
                    video_b64 = encode_video_b64(video_path)
                    raw_messages.append({
                        "role": "user",
                        "content": [{"inlineData": {"mimeType": "video/mp4", "data": video_b64}}],
                    })
            else:
                raw_messages.append({
                    "role": "user",
                    "content": [{"text": value}],
                })

        elif role_from == "assistant":
            raw_messages.append({"role": "assistant", "content": value})

    # Merge adjacent user messages (concatenate parts lists)
    merged: list[dict] = []
    for msg in raw_messages:
        if merged and merged[-1]["role"] == "user" and msg["role"] == "user":
            prev = merged[-1]["content"]
            if isinstance(prev, str):
                prev = [{"text": prev}]
                merged[-1]["content"] = prev
            curr = msg["content"]
            if isinstance(curr, str):
                curr = [{"text": curr}]
            merged[-1]["content"].extend(curr)
        else:
            merged.append(msg)

    return system_prompt, merged


def save_json(output_path: str, obj) -> None:
    """Atomically write any JSON object to disk."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_path)


def call_api_with_retry(generator: GeminiAPIGenerator, system_prompt: str, messages: list) -> str | None:
    for attempt in range(MAX_RETRIES):
        try:
            return generator._call_api(system_prompt, messages)
        except Exception as e:
            err_msg = str(e).lower()
            if "request body too large" in err_msg or "only supports up to" in err_msg:
                print(f" [Skipped: unrecoverable error, not retrying: {e}]", end="", flush=True)
                return None
            wait = RETRY_BASE_DELAY ** (attempt + 1)
            print(f" [Retry {attempt + 1}/{MAX_RETRIES}; waiting {wait}s: {e}]", end="", flush=True)
            time.sleep(wait)
    return None


def process_sample(generator: GeminiAPIGenerator, sample: dict) -> dict:
    """
    Run inference on a single sample and return a new sample with the pred field added.
    If inference fails, return the original sample (without pred).
    """
    conversations = sample.get("conversations", [])
    videos = sample.get("videos", [])

    # Find the index of the last assistant message
    last_assistant_idx = -1
    for j, conv in enumerate(conversations):
        if conv.get("from") == "assistant":
            last_assistant_idx = j

    if last_assistant_idx == -1:
        print(" [Skipped: no assistant message]", end="")
        return sample

    # Context: all messages before the last assistant message
    context_convs = conversations[:last_assistant_idx]
    system_prompt, messages = build_gemini_messages(context_convs, videos)

    pred = call_api_with_retry(generator, system_prompt, messages)

    out_sample = copy.deepcopy(sample)
    if pred is not None:
        out_sample["conversations"][last_assistant_idx]["pred"] = pred
    else:
        print(" [Inference failed, skipped]", end="")

    return out_sample

import random 
def main():
    parser = argparse.ArgumentParser(description="Gemini video inference script")
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview",
                        help="Model name, such as gemini-3-pro-preview, gemini-2.5-flash-preview, etc.")
    parser.add_argument("--ANNOS_DIR", type=str, default="data/processed/validated",
                        help="Input directory, such as data/processed/validated, etc.")
    parser.add_argument("--VIDEO_BASE_DIR", type=str, default="data/processed/video_clips",
                        help="Video path root directory, such as data/processed/video_clips, etc.")
    parser.add_argument("--seed", type=int, default=78,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of concurrent threads")
    parser.add_argument("--output_dir", type=str, default="output/infer",
                        help="Output directory, such as output/infer, etc.")
    parser.add_argument("--subset", type=str,
                default="",
                        help="Optional: only infer samples listed in the subset txt file (relative paths, such as fitness/0006.json). Default is an empty string, infer all files under ANNOS_DIR.")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: additionally save intermediate per-turn context history to --debug_dir; disabled by default, only outputs the merged conversation.")
    parser.add_argument("--debug_dir", type=str, default="output/debug/infer",
                        help="Output root directory for debug intermediate results (only effective with --debug)")
    parser.add_argument("--backend", type=str, default="api", choices=["api", "local"],
                        help="Inference backend: api=cloud Gemini/OpenAI; local=local HF model (such as Qwen3-VL)")
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Local model checkpoint path or HF hub id (only effective with --backend local)")
    parser.add_argument("--max_new_tokens", type=int, default=8192,
                        help="Maximum number of generated tokens for the local backend")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for the local backend, 0 means greedy decoding")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Video frame sampling fps for the local backend")
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        help="Attention implementation for the local backend: sdpa / flash_attention_2 / eager")
    parser.add_argument("--torch_dtype", type=str, default="bf16",
                        help="Weight precision for the local backend: bf16 / fp16 / fp32")
    args = parser.parse_args()

    global ANNOS_DIR, VIDEO_BASE_DIR, BACKEND
    ANNOS_DIR = args.ANNOS_DIR
    VIDEO_BASE_DIR = args.VIDEO_BASE_DIR
    BACKEND = args.backend
    NUM_WORKERS = args.num_workers if args.num_workers else 1

    model_name = args.model
    if BACKEND == "local":
        # The local backend uses the checkpoint directory name as the output subdirectory to avoid reusing the default gemini name
        if not model_name or model_name == "gemini-3-pro-preview":
            model_name = os.path.basename(os.path.normpath(args.model_path))
    output_dir = f"{args.output_dir}/{model_name}"
    debug_dir = f"{args.debug_dir}/{model_name}" if args.debug else None

    if BACKEND == "local":
        from tools.local_backend import LocalQwenVLGenerator
        generator = LocalQwenVLGenerator(
            model_path=args.model_path,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            fps=args.fps,
        )
        if NUM_WORKERS != 1:
            print(f"[local backend] Single-machine HF inference does not support multithreaded concurrency; forcing num_workers=1 (original value {NUM_WORKERS})")
            NUM_WORKERS = 1
    else:
        generator = GeminiAPIGenerator(model=model_name)

    os.makedirs(output_dir, exist_ok=True)

    annos_root = Path(ANNOS_DIR)
    json_files = sorted(annos_root.rglob("*.json"))
    print(f"Model: {model_name}")
    print(f"Found {len(json_files)} JSON files, output to: {output_dir}\n")
    if debug_dir:
        print(f"debug mode enabled: intermediate context history will be saved to {debug_dir}\n")

    if args.subset:
        with open(args.subset, "r", encoding="utf-8") as f:
            subset_names = {line.strip() for line in f if line.strip()}
        json_files = [jf for jf in json_files if str(jf.relative_to(annos_root)) in subset_names]
        print(f"Subset mode: read {len(subset_names)} entries from txt, matched {len(json_files)} files\n")

    random.seed(args.seed)
    random.shuffle(json_files)

    for json_file in json_files:
        rel_path = json_file.relative_to(annos_root)
        output_path = os.path.join(output_dir, rel_path)
        print(f"=== Processing file: {rel_path} ===")
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"  {json_file.name}: already exists, skipped")
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            samples = json.load(f)

        n = len(samples)
        output_samples = [None] * n

        stats_before = generator.get_token_stats()

        def process_one(args):
            i, sample = args
            try:
                out_sample = process_sample(generator, sample)
                pred_text = out_sample["conversations"][
                    next(
                        j for j in range(len(out_sample["conversations"]) - 1, -1, -1)
                        if out_sample["conversations"][j].get("from") == "assistant"
                    )
                ].get("pred", "")
                if pred_text:
                    preview = pred_text[:60].replace("\n", " ")
                    print(f"  [{i + 1:>4}/{n}]  pred: {preview}", flush=True)
                else:
                    print(f"  [{i + 1:>4}/{n}]", flush=True)
                return i, out_sample
            except FileNotFoundError as e:
                print(f"  [{i + 1:>4}/{n}] [Skipped: video file does not exist -> {e}]", flush=True)
                return i, sample
            except Exception as e:
                print(f"  [{i + 1:>4}/{n}] [Error: {e}]", flush=True)
                return i, sample

        req_interval = 0.0 if BACKEND == "local" else REQUEST_INTERVAL
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {}
            for i, s in enumerate(samples):
                futures[executor.submit(process_one, (i, s))] = i
                if req_interval:
                    time.sleep(req_interval)
            for future in as_completed(futures):
                idx, out_sample = future.result()
                output_samples[idx] = out_sample

        stats_after = generator.get_token_stats()
        gen_args = {
            "model": model_name,
            "input_tokens": stats_after["input_tokens"] - stats_before["input_tokens"],
            "output_tokens": stats_after["output_tokens"] - stats_before["output_tokens"],
            "total_tokens": stats_after["total_tokens"] - stats_before["total_tokens"],
            "api_calls": stats_after["api_calls"] - stats_before["api_calls"],
        }

        if output_samples:
            for s in reversed(output_samples):
                if s is not None:
                    s["gen_args"] = gen_args
                    break

        # Debug mode: save intermediate per-turn context history to the debug directory
        if debug_dir:
            save_json(os.path.join(debug_dir, str(rel_path)), output_samples)

        # After inference for the current file is complete, directly merge into a single conversation as the final output
        merged = merge_samples([s for s in output_samples if s is not None])
        if gen_args:
            merged["gen_args"] = gen_args
        save_json(output_path, merged)
        print(f"  -> Saved: {output_path}  (tokens: {gen_args['total_tokens']})\n")

    stats = generator.get_token_stats()
    print(
        f"All done! Called API {stats['api_calls']} times, "
        f"input tokens: {stats['input_tokens']} , "
        f"output tokens: {stats['output_tokens']} , "
        f"total tokens: {stats['total_tokens']}"
    )


if __name__ == "__main__":

    main()
