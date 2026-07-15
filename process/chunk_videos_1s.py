"""
Cut source videos into dense, fixed-length (default 1 second) chunks AND
validate + filter the annotation samples in a single step:

  1. Cut a *dense, contiguous* chunk stream covering the whole video:

        {stem}_0_1.mp4, {stem}_1_2.mp4, {stem}_2_3.mp4, ...

     so the inference stage can freely reconstruct any streaming granularity
     (step once per second, or merge N consecutive chunks into a window)
     without re-cutting videos on the fly.
  2. Validate the chunks referenced by the samples in ``--annos_save_dir`` and
     drop any sample referencing a missing/unreadable chunk, writing the
     surviving samples to ``--validated_annos_dir``.

Naming follows the existing convention ``{stem}_{start}_{end}{ext}`` used
throughout the pipeline, so ``merge_infer_conv.video_timestamps`` and friends
keep working.

Requires the ``ffmpeg`` binary on PATH plus the ``av`` and ``ffmpeg-python``
packages.
"""

import os
import json
import math
import argparse
import multiprocessing as mp
from functools import partial
from glob import glob

import av
import ffmpeg
import tqdm


VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm")


# ------------------------------------------------------------------ #
#  Cutting / validation helpers
# ------------------------------------------------------------------ #

def is_video_valid(path: str) -> bool:
    """Return True if the file exists and has a readable video stream."""
    try:
        container = av.open(path, "r")
        next(stream for stream in container.streams if stream.type == "video")
        container.close()
        return True
    except Exception:
        return False


def split_video(video_file: str, start_time: int, end_time: int, output_file: str,
                remove_audio: bool = False) -> str:
    """Cut a video with ffmpeg; skip if the target already exists and is valid, re-cut if it exists but is invalid.

    When ``remove_audio`` is True the audio track is dropped from the output clip (default: keep audio).
    """
    if os.path.exists(output_file):
        if is_video_valid(output_file):
            print(f"[skip] {output_file}")
            return output_file
        # Exists but invalid (e.g. a corrupted file from an interrupted previous run); remove and re-cut
        print(f"[re-split] {output_file}, existing file is invalid")
        os.remove(output_file)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_kwargs = {"t": int(end_time) - int(start_time), "vcodec": "libx264"}
    if remove_audio:
        output_kwargs["an"] = None          # drop the audio stream entirely
    else:
        output_kwargs["acodec"] = "aac"
    (
        ffmpeg
        .input(video_file, ss=int(start_time))
        .output(output_file, **output_kwargs)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
    return output_file


def validate_chunk(video_paths: list, output_dir: str) -> list:
    """Validate a batch of video clips and return the list of invalid paths."""
    invalid_list = []
    for video_path in tqdm.tqdm(video_paths, desc="validating", leave=False):
        full_path = os.path.join(output_dir, video_path)
        if not (os.path.exists(full_path) and is_video_valid(full_path)):
            invalid_list.append(video_path)
    return invalid_list


def collect_all_video_paths(annos_dir: str):
    """
    Recursively walk all JSON files under annos_dir and collect all unique clip paths.
    Return {json_file: [video_path, ...]} together with the global set of unique paths.
    """
    file_to_paths = {}
    all_paths = set()

    for fpath in sorted(glob(os.path.join(annos_dir, "**", "*.json"), recursive=True)):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            continue

        paths = []
        for item in data:
            for vp in item.get("videos", []):
                paths.append(vp)
                all_paths.add(vp)
        file_to_paths[fpath] = paths

    return file_to_paths, all_paths


def validate_all_videos(all_paths: set, clips_dir: str, num_workers: int) -> set:
    """Validate all video clips in parallel and return the set of invalid paths."""
    path_list = sorted(all_paths)
    chunk_size = max(1, (len(path_list) + num_workers - 1) // num_workers)
    chunks = [path_list[i:i + chunk_size] for i in range(0, len(path_list), chunk_size)]

    invalid_set = set()
    print(f"Start validating {len(path_list)} video clips...")
    with mp.Pool(num_workers) as pool:
        func = partial(validate_chunk, output_dir=clips_dir)
        for result in tqdm.tqdm(pool.imap_unordered(func, chunks),
                                total=len(chunks), desc="validating"):
            invalid_set.update(result)

    print(f"Validation done, number of invalid clips: {len(invalid_set)}")
    return invalid_set


def filter_and_save(annos_dir: str, validated_dir: str, invalid_set: set):
    """Drop samples referencing missing/invalid clips and save to validated_dir (keeping the subdirectory structure).

    Keep only samples that reference at least one video and whose referenced
    clips are all valid; samples with empty videos or any invalid clip are dropped.
    """
    os.makedirs(validated_dir, exist_ok=True)

    for fpath in sorted(glob(os.path.join(annos_dir, "**", "*.json"), recursive=True)):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        new_data = [
            item for item in data
            if item.get("videos")
            and not any(vp in invalid_set for vp in item["videos"])
        ]

        rel_path = os.path.relpath(fpath, annos_dir)
        save_path = os.path.join(validated_dir, rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)


def get_duration_seconds(video_path: str) -> float:
    """Return the video duration in seconds, trying ffprobe first, then PyAV."""
    try:
        info = ffmpeg.probe(video_path)
        dur = info.get("format", {}).get("duration")
        if dur is not None:
            return float(dur)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video" and stream.get("duration"):
                return float(stream["duration"])
    except Exception:
        pass

    try:
        with av.open(video_path) as container:
            if container.duration:
                return float(container.duration) / av.time_base
            for stream in container.streams:
                if stream.type == "video" and stream.duration and stream.time_base:
                    return float(stream.duration * stream.time_base)
    except Exception:
        pass
    return 0.0


def chunk_one_video(
    video_rel: str,
    video_dir: str,
    output_dir: str,
    chunk_seconds: int,
    remove_audio: bool,
) -> tuple:
    """Cut a single source video into consecutive fixed-length chunks.

    Returns (video_rel, num_chunks, num_failed).
    """
    ori_path = os.path.join(video_dir, video_rel)
    if not os.path.exists(ori_path):
        print(f"[skip] {ori_path}, source video not found")
        return video_rel, 0, 0

    duration = get_duration_seconds(ori_path)
    if duration <= 0:
        print(f"[skip] {ori_path}, could not determine duration")
        return video_rel, 0, 0

    dirpart = os.path.dirname(video_rel)
    basename = os.path.basename(video_rel)
    stem, ext = os.path.splitext(basename)

    num_chunks = int(math.ceil(duration / chunk_seconds))
    failed = 0
    for i in range(num_chunks):
        start = i * chunk_seconds
        end = (i + 1) * chunk_seconds
        clip_name = f"{stem}_{start}_{end}{ext}"
        out_path = os.path.join(output_dir, dirpart, clip_name) if dirpart else os.path.join(output_dir, clip_name)
        try:
            split_video(ori_path, start, end, out_path, remove_audio=remove_audio)
        except Exception as e:
            print(f"Error splitting {video_rel} [{start},{end}]: {e}")
            failed += 1
    return video_rel, num_chunks, failed


def collect_source_videos(video_dir: str) -> list:
    """Recursively collect source videos, returned as paths relative to video_dir."""
    rels = []
    for path in sorted(glob(os.path.join(video_dir, "**", "*"), recursive=True)):
        if os.path.isfile(path) and path.lower().endswith(VIDEO_EXTS):
            rels.append(os.path.relpath(path, video_dir))
    return rels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cut every source video into dense fixed-length (default 1s) chunks, "
                    "then validate + filter the annotation samples in one step."
    )
    parser.add_argument("--video_dir", type=str, default="data/example/videos_2fps",
                        help="root directory of the source (uncut) videos")
    parser.add_argument("--output_dir", type=str, default="data/processed/video_clips",
                        help="output directory for the fixed-length chunks (== inference VIDEO_BASE_DIR)")
    parser.add_argument("--chunk_seconds", type=int, default=1,
                        help="length of each chunk in seconds (default: 1)")
    parser.add_argument("--annos_save_dir", type=str, default="data/processed/save",
                        help="directory of JSON files produced by data_proprecess.py (samples to validate)")
    parser.add_argument("--validated_annos_dir", type=str, default="data/processed/validated",
                        help="output directory for the validated JSON files")
    parser.add_argument("--num_workers", type=int, default=8, help="number of parallel processes")
    parser.add_argument("--remove_audio", action="store_true", default=False,
                        help="drop the audio track from the cut chunks (default: keep audio)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Step 1: Collect source videos")
    videos = collect_source_videos(args.video_dir)
    print(f"{len(videos)} source videos under {args.video_dir}")

    print("=" * 60)
    print(f"Step 2: Cut {args.chunk_seconds}s chunks "
          f"(audio {'removed' if args.remove_audio else 'kept'}) using {args.num_workers} processes")

    func = partial(
        chunk_one_video,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        chunk_seconds=args.chunk_seconds,
        remove_audio=args.remove_audio,
    )

    total_chunks = 0
    total_failed = 0
    with mp.Pool(args.num_workers) as pool:
        for _rel, n_chunks, n_fail in tqdm.tqdm(
            pool.imap_unordered(func, videos), total=len(videos), desc="chunking"
        ):
            total_chunks += n_chunks
            total_failed += n_fail

    print(f"{total_chunks} chunks written to {args.output_dir} (cut failures: {total_failed})")

    print("=" * 60)
    print("Step 3: Collect the chunk paths referenced by the samples")
    file_to_paths, all_paths = collect_all_video_paths(args.annos_save_dir)
    print(f"{len(file_to_paths)} JSON files, {len(all_paths)} unique referenced chunks")

    print("=" * 60)
    print("Step 4: Validate the referenced chunks")
    invalid_set = validate_all_videos(all_paths, args.output_dir, args.num_workers)

    print("=" * 60)
    print("Step 5: Filter and save the validated samples")
    filter_and_save(args.annos_save_dir, args.validated_annos_dir, invalid_set)

    print("=" * 60)
    print(f"All done! Chunks -> {args.output_dir}; validated JSON -> {args.validated_annos_dir}")


if __name__ == "__main__":
    main()
