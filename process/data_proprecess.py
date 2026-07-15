import os
import json
import copy
import random
import argparse
from pathlib import Path

# ============================================================
# Configuration (filled from command-line arguments in main(); see build_system_prompt / parse_args below)
# ============================================================
TIME_WINDOW = 60                     # video context window length (seconds)
CHUNK_SECONDS = 1                    # granularity (seconds) of each <video> chunk / per-second step
TAIL_MIN = 10                        # min extra silent seconds after the last answer
TAIL_MAX = 15                        # max extra silent seconds after the last answer
SYSTEM_PROMPT = ""                   # built in main() based on TIME_WINDOW


def build_system_prompt(time_window: int) -> str:
    return f'''You are an AI assistant observing a continuous live video stream of a user performing a step-by-step task. 
Note that you only see the most recent {time_window} seconds of the video stream. The user will describe their goal at the beginning of the conversation. 

Your role is to provide timely, context-aware, concise, and conversational guidance: 
when the user should proceed to the next step, tell them proactively; 
when the user completes a step correctly, acknowledge it; 
when the user makes a mistake, point it out clearly, and give the user the correct instruction. 

At all other times, output only a single word: "Silent".

NOTE: 
- You should be very concise and to the point.
- You should only guide the user through ONE step at a time. Never mention or anticipate future steps beyond the immediate next one.
- You should not repeat the same instruction or feedback too many times.
- You should not give the user too much information.
'''

# ============================================================


def add_suffix_to_path(video_path: str, start: int, end: int) -> str:
    """Insert _{start}_{end} before the extension of the video path."""
    base, ext = os.path.splitext(video_path)
    return f"{base}_{start}_{end}{ext}"


_DURATION_CACHE: dict = {}


def get_duration_seconds(video_path: str):
    """Return the source video duration in seconds (ffprobe, then PyAV).

    Returns None when the file is missing or no backend is available, so the
    caller can fall back to the full random tail.
    """
    if not video_path or not os.path.exists(video_path):
        return None
    if video_path in _DURATION_CACHE:
        return _DURATION_CACHE[video_path]

    duration = None
    try:
        import ffmpeg
        info = ffmpeg.probe(video_path)
        dur = info.get("format", {}).get("duration")
        duration = float(dur) if dur is not None else None
    except Exception:
        duration = None

    if duration is None:
        try:
            import av
            with av.open(video_path) as container:
                if container.duration:
                    duration = float(container.duration) / av.time_base
        except Exception:
            duration = None

    _DURATION_CACHE[video_path] = duration
    return duration


def add_video_context(sample: dict) -> dict:
    """
    Insert <video> user messages into the conversation based on endpoint_timestamp and TIME_WINDOW,
    and update the videos field with the corresponding list of clips.

    The rolling window [max(0, endpoint - TIME_WINDOW), endpoint] is sliced into
    fixed-length ``CHUNK_SECONDS`` (default 1s) <video> chunks, and the original
    conversation messages are interleaved at their timestamps. This way the
    inference stage sees the video as a dense stream of 1s chunks that can be
    consumed at any granularity (the clips live in ``video_clips``).

    Logic:
    - Video window start video_start = max(0, endpoint_timestamp - TIME_WINDOW)
    - Walk the conversation; before every message whose timestamp falls in
      (prev_boundary, endpoint_ts], emit the 1s <video> chunks that fill the gap
      [prev_boundary, msg_ts], then the message itself.
    - After the walk, emit the trailing 1s <video> chunks covering
      [prev_boundary, endpoint_ts].
    """
    endpoint_ts = int(sample["endpoint_timestamp"])
    video_start = max(0, endpoint_ts - TIME_WINDOW)

    original_videos = sample.get("videos", [])
    if not original_videos:
        return sample
    original_video_path = original_videos[0]

    conversations = sample["conversations"]
    new_conversations = []
    new_videos = []
    state = {"prev": video_start}

    def emit_chunks(upto: int) -> None:
        """Emit fixed-length <video> chunks from state['prev'] up to ``upto``."""
        prev = state["prev"]
        while prev < upto:
            end = min(prev + CHUNK_SECONDS, upto)
            clip_path = add_suffix_to_path(original_video_path, prev, end)
            new_conversations.append({"from": "user", "value": "<video>"})
            new_videos.append(clip_path)
            prev = end
        state["prev"] = prev

    for conv in conversations:
        if "timestamp" not in conv:
            new_conversations.append(conv)
            continue
        ts = int(conv["timestamp"])
        # Only fill 1s video chunks when the message timestamp is inside the window and beyond the current boundary
        if ts >= video_start and ts > state["prev"]:
            emit_chunks(ts)
        new_conversations.append(conv)

    # Fill trailing 1s video chunks covering [prev_boundary, endpoint_ts]
    if endpoint_ts > state["prev"]:
        emit_chunks(endpoint_ts)

    sample["conversations"] = new_conversations
    sample["videos"] = new_videos
    return sample


def process_annotation(data: dict, duration: float = None) -> list:
    """
    Process a single annotation (one dict) and return the list of per-second samples (one inference per second).

    Endpoints run from the "first question time" (timestamp of the first user
    instruction) to "the last answer timestamp + random [TAIL_MIN, TAIL_MAX]
    seconds", generating one sample per second (the random trailing silent span
    tests whether the model stays silent after the task ends).

    When the true video ``duration`` is known, the tail is trimmed to the end of
    the video to avoid referencing 1s chunks beyond the video that don't exist:
    - Each sample keeps the full conversation up to (and including) the endpoint, filling the rolling video window with 1s chunks.
    - If that second has an assistant turn (speaking point), its GT reply is the target (sample_type=1).
    - Otherwise append {"from":"assistant","value":"Silent"} at the end as the target
      (silent point, sample_type=2).

    Samples of adjacent seconds differ only in their history context (sliding window + conversation so far).
    """
    conversations = data.get("conversations", [])

    # Collect assistant (speaking point) timestamps and the first question time
    assistant_timestamps = [
        int(c["timestamp"]) for c in conversations
        if c.get("from") == "assistant" and "timestamp" in c
    ]
    if not assistant_timestamps:
        return []

    user_timestamps = [
        int(c["timestamp"]) for c in conversations
        if c.get("from") == "user" and c.get("value") != "<video>" and "timestamp" in c
    ]
    last_a_timestamp = max(assistant_timestamps)
    # Start: first question time; fall back to the first assistant time if there is no user instruction
    start_t = min(user_timestamps) if user_timestamps else min(assistant_timestamps)
    speak_timestamps = set(assistant_timestamps)
    # End: extend a random 10~15s past the last answer (pure silent tail)
    tail = random.randint(TAIL_MIN, TAIL_MAX)
    # When the video duration is known, trim the tail to the video end (may be under 5s, even 0)
    if duration is not None and duration > 0:
        available_tail = max(0, int(duration) - last_a_timestamp)
        tail = min(tail, available_tail)
    end_t = last_a_timestamp + tail

    samples = []
    for t in range(start_t, end_t + 1, CHUNK_SECONDS):
        new_item = copy.deepcopy(data)
        new_item["conversations"] = [
            conv for conv in conversations if int(conv["timestamp"]) <= t
        ]
        new_item["endpoint_timestamp"] = t
        is_speak = t in speak_timestamps
        new_item["sample_type"] = 1 if is_speak else 2
        new_item["conversations"].insert(0, {"from": "system", "value": SYSTEM_PROMPT})
        new_item = add_video_context(new_item)
        # Silent second: append a Silent assistant message at the end as the target (with timestamp for scoring alignment)
        if not is_speak:
            new_item["conversations"].append(
                {"from": "assistant", "value": "Silent", "timestamp": t}
            )
        samples.append(new_item)

    return samples


def output_name_from_item(item: dict, index: int) -> str:
    """
    Derive the output filename from the annotation item's videos field (named after the video, extension changed to .json).
    E.g. videos=["0009.mp4"] -> "0009.json"; falls back to video_id or the index when missing.
    """
    videos = item.get("videos") or []
    if videos:
        stem = os.path.splitext(os.path.basename(videos[0]))[0]
    elif item.get("video_id"):
        stem = str(item["video_id"])
    else:
        stem = f"sample_{index:06d}"
    return f"{stem}.json"


def load_annotations(input_path: str) -> list:
    """
    Load input annotations. Two forms are supported:
    - A single full JSON file: content is a list of annotation items (or a single dict).
    - A directory: each *.json under it is one annotation dict (legacy behavior).
    Returns a list of annotation item dicts.
    """
    p = Path(input_path)
    if p.is_dir():
        items = []
        for json_file in sorted(p.rglob("*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                items.append(json.load(f))
        return items

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split raw annotation JSON into per-second streaming samples (one inference per second), inserting sliding-window <video> messages as 1s chunks."
    )
    parser.add_argument("--annos_base_dir", type=str, default="data/example/annotation_test_subset50.json",
                        help="Raw annotation input: a single full JSON file (a list of annotations) or a directory of JSON files")
    parser.add_argument("--output_dir", type=str, default="data/processed/save",
                        help="Output directory for the split samples")
    parser.add_argument("--video_dir", type=str, default="data/example/videos_2fps",
                        help="Source video directory, used to probe duration for trimming the trailing silent span (falls back to the full random tail on failure)")
    parser.add_argument("--time_window", type=int, default=60,
                        help="Video context window length (seconds)")
    parser.add_argument("--chunk_seconds", type=int, default=1,
                        help="Granularity (seconds) of each <video> chunk in the rolling window, and the per-second generation step")
    parser.add_argument("--tail_min", type=int, default=10,
                        help="Min random silent seconds appended after the last answer")
    parser.add_argument("--tail_max", type=int, default=15,
                        help="Max random silent seconds appended after the last answer")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (for sampling the trailing silent span length)")
    return parser.parse_args()


def main():
    args = parse_args()

    global TIME_WINDOW, CHUNK_SECONDS, TAIL_MIN, TAIL_MAX, SYSTEM_PROMPT
    TIME_WINDOW = args.time_window
    CHUNK_SECONDS = max(1, args.chunk_seconds)
    TAIL_MIN = args.tail_min
    TAIL_MAX = args.tail_max
    SYSTEM_PROMPT = build_system_prompt(TIME_WINDOW)
    random.seed(args.seed)

    annos_base_dir = args.annos_base_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    items = load_annotations(annos_base_dir)
    print(f"Loaded {len(items)} annotations")

    for index, item in enumerate(items):
        out_name = output_name_from_item(item, index)     # named after the video, e.g. 0009.json
        output_path = Path(output_dir) / out_name
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Probe source video duration to trim the trailing silent span
        src_videos = item.get("videos") or []
        duration = get_duration_seconds(os.path.join(args.video_dir, src_videos[0])) if src_videos else None

        samples = process_annotation(copy.deepcopy(item), duration=duration)

        speak = sum(1 for s in samples if s.get("sample_type") == 1)
        silent = sum(1 for s in samples if s.get("sample_type") == 2)
        print(f"  {out_name}: speaking={speak}, silent={silent}, total={len(samples)}")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"    Saved to: {output_path}")

    print(f"\nAll done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
