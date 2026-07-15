import os
import json
from pathlib import Path



def video_timestamps(video_path: str) -> tuple[int, int]:
    """Parse (start, end) timestamps from the video filename, e.g. ..._146_206.mp4 → (146, 206)."""
    stem = Path(video_path).stem
    parts = stem.split("_")
    try:
        return int(parts[-2]), int(parts[-1])
    except (ValueError, IndexError):
        return 0, 0


def merge_samples(samples: list[dict]) -> dict:
    """
    Merge the multiple cumulative conversation samples in one JSON file into a single conversation.

    Rules:
    - system prompt: take the first sample's, keep only one
    - first user text message: take the first sample's, keep only one
    - <video> messages: skip all
    - assistant messages: keep only the one with a pred field (last pred-bearing turn per sample)
    - videos: each pred-contributing sample contributes its videos[-1] (the clip newly introduced that turn)
    - ordering: ascending by the start timestamp of the corresponding video
    """
    if not samples or len(samples) == 0:
        return {}

    first = samples[0]

    # Extract the system and first user text message from the first sample
    system_msg = None
    first_user_msg = None
    for conv in first.get("conversations", []):
        if conv.get("from") == "system" and system_msg is None:
            system_msg = conv
        elif (conv.get("from") == "user"
              and conv.get("value") != "<video>"
              and first_user_msg is None):
            first_user_msg = conv

    # Collect (pred_conv, video) pairs to sort later
    pairs: list[tuple[dict, str]] = []

    for sample in samples:
        convs = sample.get("conversations", [])
        videos = sample.get("videos", [])

        # Find the last pred-bearing assistant message
        pred_conv = None
        for conv in convs:
            if conv.get("from") == "assistant" and "pred" in conv:
                pred_conv = conv

        if pred_conv is None:
            continue

        # If the assistant turn has no timestamp, use the sample's endpoint_timestamp
        if "timestamp" not in pred_conv and "endpoint_timestamp" in sample:
            pred_conv = {**pred_conv, "timestamp": sample["endpoint_timestamp"]}

        if 'sample_type' in sample:
            pred_conv['sample_type'] = sample['sample_type']

        # The clip newly introduced this turn (the last video in the cumulative conversation)
        video = videos[-1] if videos else ""
        pairs.append((pred_conv, video))

    # Sort ascending by (start, end) timestamps; tie-break on end when start is equal
    pairs.sort(key=lambda p: video_timestamps(p[1]))

    merged_convs = []
    if first_user_msg:
        merged_convs.append(first_user_msg)
    merged_convs.extend(conv for conv, _ in pairs)

    # Extract gen_args (from the last sample carrying that field)
    gen_args = None

    # Reuse the first sample's metadata fields, replacing conversations.
    # Do not keep videos: one video has too many 1s chunks; the merged output doesn't need the video list.
    merged = {k: v for k, v in first.items() if k not in ("conversations", "videos",
                                                            "endpoint_timestamp", "sample_type",
                                                            "inferred_goal", "inferred_knowledge")}
    merged["conversations"] = merged_convs
    if gen_args is not None:
        merged["gen_args"] = gen_args
    return merged


def main():
    import argparse

    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, nargs="?", default=None)
    parser.add_argument("--input_root", type=str, default="output/infer",
                        help="Inference results directory")
    parser.add_argument("--output_root", type=str, default="output/infer_merged",
                        help="Output directory")
    args = parser.parse_args()
    model_name = args.model

    INFERENCE_DIR = args.input_root
    OUTPUT_ROOT = args.output_root

    print(f"INFERENCE_DIR: {INFERENCE_DIR}")
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    if model_name:
        model_names = [model_name]
    else:
        inference_root = Path(INFERENCE_DIR)
        model_names = sorted(
            d.name for d in inference_root.iterdir() if d.is_dir()
        )
        print(f"No model specified; found {len(model_names)} models: {model_names}\n")

    for model_name in model_names:
        OUTPUT_DIR = f"{OUTPUT_ROOT}/{model_name}/"
        inference_dir = Path(INFERENCE_DIR) / model_name
        json_files = sorted(inference_dir.rglob("*.json"))
        print(f"===== Model: {model_name} =====")
        print(f"Found {len(json_files)} inference files, source directory: {inference_dir}\n")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        jsonl_path = Path(OUTPUT_DIR) / f"{model_name}.jsonl"

        total_pred = 0
        n_written = 0
        ## Only save the final jsonl (one line per video); no longer save individual json files
        with open(jsonl_path, "w", encoding="utf-8") as out:
            for json_file in tqdm.tqdm(json_files, desc=f"Building jsonl for {model_name} ..."):
                if json_file.stat().st_size == 0:
                    continue

                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        samples = json.load(f)
                except:
                    print(f'{json_file} failed to load')
                    continue

                if not samples:
                    continue

                # infer.py already merges each video into a dict; if we get a per-turn list, merge it as a fallback
                merged = samples if isinstance(samples, dict) else merge_samples(samples)

                pred_count = sum(
                    1 for conv in merged.get("conversations", [])
                    if conv.get("from") == "assistant" and "pred" in conv
                )
                total_pred += pred_count

                out.write(json.dumps(merged, ensure_ascii=False) + "\n")
                n_written += 1

        print(f"\nTotal: {n_written} videos / {total_pred} preds, saved to: {jsonl_path}\n")



if __name__ == "__main__":
    main()
