import os
import sys
import json
import base64
import time
import copy
import random
import argparse
import threading

sys.path.insert(0, os.path.dirname(__file__))
from tools.llm_utils import GeminiAPIGenerator
from tools.merge_infer_conv import merge_samples
import process.data_proprecess as dpp

# In-memory cache for video Base64 (thread-safe)
_video_b64_cache: dict[str, str] = {}
_video_b64_lock = threading.Lock()

# Streaming inference without GT history: replay each raw-annotation video
# second by second, using the model's own previous predictions as history
# ("Silent" predictions are not added). Output format matches
# infer.py, so downstream merge/scoring scripts need no changes.

MAX_RETRIES      = 10    # Maximum number of retries when an API call fails
RETRY_BASE_DELAY = 10    # Retry wait base (seconds), exponential backoff
REQUEST_INTERVAL = 0.5   # Minimum interval between two consecutive API calls (seconds)
MAX_VIDEOS       = 60     # Number of most recent <video> blocks attached per request; overridden in main() by time_window//chunk_seconds
BACKEND          = "api"  # "api" = cloud Gemini/OpenAI; "local" = local HF model

VIDEO_BASE_DIR = "data/processed/video_clips"   # Root directory of 1s video blocks (referenced by add_video_context)


def encode_video_b64(video_path: str) -> str:
    with _video_b64_lock:
        if video_path in _video_b64_cache:
            return _video_b64_cache[video_path]
    with open(video_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    with _video_b64_lock:
        _video_b64_cache[video_path] = data
    return data


def build_gemini_messages(conversations: list, videos: list, max_videos: int | None = None) -> tuple[str, list[dict]]:
    """
    Convert conversations (excluding the last assistant message to be predicted) into Gemini messages format.

    Returns (system_prompt, messages), where the content of each message in messages is:
    - Plain text: str
    - Video content: list of Gemini parts (inlineData)
    - Mixed content: list of Gemini parts

    Adjacent user messages are merged into one (parts arrays concatenated).
    Only the most recent max_videos <video> blocks are kept (earlier ones are dropped to avoid an oversized request body).
    """
    if max_videos is None:
        max_videos = MAX_VIDEOS

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
                    # Local backend uses the file path directly, avoiding base64 encode/decode overhead
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


def call_api_with_retry(generator: GeminiAPIGenerator, system_prompt: str, messages: list) -> str | None:
    for attempt in range(MAX_RETRIES):
        try:
            return generator._call_api(system_prompt, messages)
        except Exception as e:
            err_msg = str(e).lower()
            if "request body too large" in err_msg or "only supports up to" in err_msg:
                print(f" [Skip: unrecoverable error, no retry: {e}]", end="", flush=True)
                return None
            wait = RETRY_BASE_DELAY ** (attempt + 1)
            print(f" [Retry {attempt + 1}/{MAX_RETRIES}, waiting {wait}s: {e}]", end="", flush=True)
            time.sleep(wait)
    return None


def is_silent_response(text: str | None) -> bool:
    """Determine whether the model reply is "Silent".

    None (inference failure) is also treated as silent, so it is not added to history. Leniently strips
    leading/trailing quotes and trailing punctuation before comparing with "silent".
    """
    if text is None:
        return True
    cleaned = text.strip().strip('"').strip("'").strip()
    cleaned = cleaned.rstrip(".!。！ ").strip()
    return cleaned.lower() == "silent"


def collect_events(conversations: list) -> tuple[dict[int, list[str]], list[tuple[int, str]]]:
    """Collect speaking points and user instructions from the raw annotation's conversations.

    Returns:
    - assistant_ts_values: {timestamp -> [GT assistant text, ...]} (speaking points)
    - user_events: [(timestamp, user text), ...] (user instructions, sorted by time)
    """
    assistant_ts_values: dict[int, list[str]] = {}
    user_events: list[tuple[int, str]] = []
    for c in conversations:
        if "timestamp" not in c:
            continue
        ts = int(c["timestamp"])
        role = c.get("from")
        value = c.get("value", "")
        if role == "assistant":
            assistant_ts_values.setdefault(ts, []).append(value)
        elif role == "user" and value != "<video>":
            user_events.append((ts, value))
    user_events.sort(key=lambda x: x[0])
    return assistant_ts_values, user_events


def build_second_sample(
    session_meta: dict,
    original_video_path: str,
    system_prompt: str,
    history_msgs: list[dict],
    t: int,
    is_speak: bool,
    gt_value: str,
) -> dict:
    """Construct the single sample for second t (structure identical to a validated sample).

    conversations = [system] + history (user instructions + the model's own non-silent predictions) + target assistant.
    Then reuse data_proprecess.add_video_context to insert the rolling 1s <video> window over [t-TIME_WINDOW, t].
    """
    convs: list[dict] = [{"from": "system", "value": system_prompt}]
    convs.extend(copy.deepcopy(history_msgs))
    convs.append({
        "from": "assistant",
        "value": gt_value if is_speak else "Silent",
        "timestamp": t,
    })

    sample = {k: v for k, v in session_meta.items() if k not in ("conversations", "videos")}
    sample["conversations"] = convs
    sample["videos"] = [original_video_path]
    sample["endpoint_timestamp"] = t
    sample["sample_type"] = 1 if is_speak else 2

    dpp.add_video_context(sample)  # Expand <video> in place and backfill the videos list
    return sample


def generate_pred(generator: GeminiAPIGenerator, sample: dict) -> str | None:
    """Predict on the target assistant at the end of the sample, writing pred back to that message. Returns the pred text."""
    conversations = sample["conversations"]
    last_assistant_idx = -1
    for j, conv in enumerate(conversations):
        if conv.get("from") == "assistant":
            last_assistant_idx = j
    if last_assistant_idx == -1:
        return None

    context_convs = conversations[:last_assistant_idx]
    system_prompt, messages = build_gemini_messages(context_convs, sample.get("videos", []))
    pred = call_api_with_retry(generator, system_prompt, messages)
    if pred is not None:
        conversations[last_assistant_idx]["pred"] = pred
    return pred


def is_complete_output(output_path: str) -> bool:
    """Determine whether a video's final output (merged conversation) is already complete, to support resumable runs.

    The final product is a single merged conversation dict; as long as the file is parseable and is a
    dict containing conversations, it is considered complete. (Compatible with the old per-second list format:
    only complete if the last entry carries gen_args.)
    """
    if not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
        return False
    try:
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    if isinstance(data, dict):
        return "conversations" in data
    if isinstance(data, list):
        return len(data) > 0 and isinstance(data[-1], dict) and "gen_args" in data[-1]
    return False


def save_json(output_path: str, obj) -> None:
    """Write an arbitrary JSON object to disk atomically."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_path)


def save_samples(output_path: str, output_samples: list[dict], gen_args: dict) -> None:
    """Write the current per-second sample list (intermediate context history) to disk atomically.

    Only called in debug mode: written once after each infer, continuously persisting the accumulated history / predictions.
    """
    if not output_samples:
        return
    if gen_args:
        output_samples[-1]["gen_args"] = gen_args
    save_json(output_path, output_samples)


def process_session(
    generator: GeminiAPIGenerator,
    session: dict,
    duration: float | None,
    model_name: str,
    debug_path: str | None = None,
) -> tuple[list[dict], dict]:
    """Run per-second streaming inference over one video's raw annotation, returning (per-second sample list, gen_args).

    When debug_path is non-empty (debug mode), after each second of infer completes, the intermediate per-second
    samples (with full context history) are written to debug_path; otherwise no intermediate results are written.
    """
    conversations = session.get("conversations", [])
    assistant_ts_values, user_events = collect_events(conversations)

    if not assistant_ts_values:
        print("  [Skip: no assistant speaking points]")
        return [], {}

    original_videos = session.get("videos", [])
    if not original_videos:
        print("  [Skip: no videos field]")
        return [], {}
    original_video_path = original_videos[0]

    assistant_ts = sorted(assistant_ts_values)
    user_ts = [ts for ts, _ in user_events]
    last_a_ts = max(assistant_ts)
    start_t = min(user_ts) if user_ts else min(assistant_ts)
    speak_set = set(assistant_ts)

    # Endpoint: last speaking point + a random [TAIL_MIN, TAIL_MAX] seconds of trailing silence (clipped to the video end if duration is known)
    tail = random.randint(dpp.TAIL_MIN, dpp.TAIL_MAX)
    if duration is not None and duration > 0:
        tail = min(tail, max(0, int(duration) - last_a_ts))
    end_t = last_a_ts + tail

    system_prompt = dpp.SYSTEM_PROMPT
    session_meta = {k: v for k, v in session.items() if k not in ("conversations", "videos")}

    stats_before = generator.get_token_stats()

    history_msgs: list[dict] = []
    output_samples: list[dict] = []
    u_idx = 0
    step = dpp.CHUNK_SECONDS

    total_steps = len(range(start_t, end_t + 1, step))
    print(f"  Per-second replay: t={start_t}..{end_t} ({total_steps} steps total, {len(speak_set)} speaking points)")

    for step_i, t in enumerate(range(start_t, end_t + 1, step)):
        # Merge into history any user instructions up to and including this moment that have not yet been added
        while u_idx < len(user_events) and user_events[u_idx][0] <= t:
            ts_u, val_u = user_events[u_idx]
            history_msgs.append({"from": "user", "value": val_u, "timestamp": ts_u})
            u_idx += 1

        is_speak = t in speak_set
        gt_value = " ".join(assistant_ts_values[t]) if is_speak else "Silent"

        sample = build_second_sample(
            session_meta=session_meta,
            original_video_path=original_video_path,
            system_prompt=system_prompt,
            history_msgs=history_msgs,
            t=t,
            is_speak=is_speak,
            gt_value=gt_value,
        )

        try:
            pred = generate_pred(generator, sample)
        except FileNotFoundError as e:
            print(f"  [{step_i + 1:>4}/{total_steps}] t={t} [Skip: video block does not exist -> {e}]", flush=True)
            pred = None
        except Exception as e:
            print(f"  [{step_i + 1:>4}/{total_steps}] t={t} [Error: {e}]", flush=True)
            pred = None

        output_samples.append(sample)

        if pred:
            preview = pred[:50].replace("\n", " ")
            print(f"  [{step_i + 1:>4}/{total_steps}] t={t:>4} pred: {preview}", flush=True)
        else:
            print(f"  [{step_i + 1:>4}/{total_steps}] t={t:>4} (no pred)", flush=True)

        # Only non-silent predictions are fed back into history (silent / failures are not added)
        if not is_silent_response(pred):
            history_msgs.append({"from": "assistant", "value": pred, "timestamp": t})

        # Debug mode: write the intermediate context history to disk once after each second of infer
        if debug_path:
            save_samples(debug_path, output_samples, {})

        if BACKEND != "local":
            time.sleep(REQUEST_INTERVAL)

    stats_after = generator.get_token_stats()
    gen_args = {
        "model": model_name,
        "input_tokens": stats_after["input_tokens"] - stats_before["input_tokens"],
        "output_tokens": stats_after["output_tokens"] - stats_before["output_tokens"],
        "total_tokens": stats_after["total_tokens"] - stats_before["total_tokens"],
        "api_calls": stats_after["api_calls"] - stats_before["api_calls"],
    }
    # Debug mode: write the intermediate result one more time at the end, with gen_args
    if debug_path:
        save_samples(debug_path, output_samples, gen_args)
    return output_samples, gen_args


def main():
    parser = argparse.ArgumentParser(
        description="Streaming video inference with no GT history: replay from the raw annotation second by second, history uses the model's own predictions, silent is not added to history."
    )
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview",
                        help="Model name, e.g. gemini-3-pro-preview, gemini-2.5-flash-preview, etc.")
    parser.add_argument("--ANNOS_DIR", type=str, default="data/example/annotation_test_subset50.json",
                        help="Raw annotation input: a single JSON file (whose content is an annotation list) or a directory containing multiple JSONs")
    parser.add_argument("--VIDEO_BASE_DIR", type=str, default="data/processed/video_clips",
                        help="Root directory of 1s video blocks (the <video> expanded from the annotation is relative to this directory)")
    parser.add_argument("--VIDEO_DIR", type=str, default="data/example/videos_2fps",
                        help="Source video directory, only used to probe duration for clipping trailing silence; if probing fails, use the full random tail")
    parser.add_argument("--seed", type=int, default=78, help="Random seed (sampling the trailing silence interval length)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="(Ignored) Streaming inference has step-by-step internal dependencies, videos are processed one by one in order")
    parser.add_argument("--output_dir", type=str, default="output/infer",
                        help="Output directory, e.g. output/infer, etc.")
    parser.add_argument("--time_window", type=int, default=60, help="Video context window length (seconds)")
    parser.add_argument("--chunk_seconds", type=int, default=1, help="Granularity of each <video> block (seconds), also the per-second step size")
    parser.add_argument("--tail_min", type=int, default=10, help="Minimum silence seconds to randomly extend after the last speaking point")
    parser.add_argument("--tail_max", type=int, default=15, help="Maximum silence seconds to randomly extend after the last speaking point")
    parser.add_argument("--subset", type=str, default="",
                        help="Optional: only infer the output names listed in the subset txt (e.g. 0009.json), one per line. Defaults to inferring all.")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: additionally save the intermediate per-second context history to --debug_dir; disabled by default, only outputs the merged conversation.")
    parser.add_argument("--debug_dir", type=str, default="output/debug/infer_no_gt_history",
                        help="Output root directory for intermediate debug results (only effective with --debug)")
    parser.add_argument("--backend", type=str, default="api", choices=["api", "local"],
                        help="Inference backend: api=cloud Gemini/OpenAI; local=local HF model (e.g. Qwen3-VL)")
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Local model checkpoint path or HF hub id (only effective with --backend local)")
    parser.add_argument("--max_new_tokens", type=int, default=8192,
                        help="Max number of generated tokens for the local backend")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for the local backend, 0 means greedy decoding")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Video frame sampling fps for the local backend")
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        help="Attention implementation for the local backend: sdpa / flash_attention_2 / eager")
    parser.add_argument("--torch_dtype", type=str, default="bf16",
                        help="Weight precision for the local backend: bf16 / fp16 / fp32")
    args = parser.parse_args()

    global VIDEO_BASE_DIR, MAX_VIDEOS, BACKEND
    VIDEO_BASE_DIR = args.VIDEO_BASE_DIR
    BACKEND = args.backend

    model_name = args.model
    if BACKEND == "local" and (not model_name or model_name == "gemini-3-pro-preview"):
        # Local backend uses the checkpoint directory name as the output subdirectory, avoiding the default gemini name
        model_name = os.path.basename(os.path.normpath(args.model_path))
    output_dir = f"{args.output_dir}/{model_name}"
    debug_dir = f"{args.debug_dir}/{model_name}" if args.debug else None

    # Configure the global parameters of the data_proprecess module (reused by add_video_context / tail logic)
    dpp.TIME_WINDOW = args.time_window
    dpp.CHUNK_SECONDS = max(1, args.chunk_seconds)

    # Number of most recent <video> blocks attached per request = total blocks in the window (derived from time_window / chunk_seconds)
    MAX_VIDEOS = max(1, args.time_window // dpp.CHUNK_SECONDS)
    dpp.TAIL_MIN = args.tail_min
    dpp.TAIL_MAX = args.tail_max
    dpp.SYSTEM_PROMPT = dpp.build_system_prompt(args.time_window)

    random.seed(args.seed)

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
    else:
        generator = GeminiAPIGenerator(model=model_name)
    os.makedirs(output_dir, exist_ok=True)

    items = dpp.load_annotations(args.ANNOS_DIR)
    print(f"Model: {model_name}")
    print(f"Read {len(items)} annotations from {args.ANNOS_DIR}, output to: {output_dir}\n")
    if debug_dir:
        print(f"Debug mode enabled: intermediate context history will be saved to {debug_dir}\n")

    subset_names: set[str] | None = None
    if args.subset:
        with open(args.subset, encoding="utf-8") as f:
            subset_names = {line.strip() for line in f if line.strip()}
        print(f"Subset mode: read {len(subset_names)} output names from txt\n")

    for index, item in enumerate(items):
        out_name = dpp.output_name_from_item(item, index)   # e.g. 0009.json
        if subset_names is not None and out_name not in subset_names:
            continue

        output_path = os.path.join(output_dir, out_name)
        print(f"=== Processing video: {out_name} ===")
        if is_complete_output(output_path):
            print(f"  {out_name}: already complete, skipping")
            continue

        # Probe the source video duration, used to clip the trailing silence interval
        src_videos = item.get("videos") or []
        duration = None
        if src_videos:
            duration = dpp.get_duration_seconds(os.path.join(args.VIDEO_DIR, src_videos[0]))

        # In debug mode write the intermediate per-second results to the debug directory; in non-debug mode write no intermediate results
        debug_path = os.path.join(debug_dir, out_name) if debug_dir else None
        output_samples, gen_args = process_session(
            generator, copy.deepcopy(item), duration=duration,
            model_name=model_name, debug_path=debug_path,
        )

        if not output_samples:
            print(f"  {out_name}: no samples, skipping save\n")
            continue

        # After all infer for the current video is complete, merge directly into a single conversation as the final product
        merged = merge_samples([s for s in output_samples if s is not None])
        if gen_args:
            merged["gen_args"] = gen_args
        save_json(output_path, merged)
        print(f"  -> Saved: {output_path}  (tokens: {gen_args.get('total_tokens', 0)})\n")

        # Release this video's base64 cache to avoid memory bloat over long runs
        with _video_b64_lock:
            _video_b64_cache.clear()

    stats = generator.get_token_stats()
    print(
        f"All done! Called the API {stats['api_calls']} times in total, "
        f"input tokens: {stats['input_tokens']}, "
        f"output tokens: {stats['output_tokens']}, "
        f"total tokens: {stats['total_tokens']}"
    )


if __name__ == "__main__":
    main()
