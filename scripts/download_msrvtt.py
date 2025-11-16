#!/usr/bin/env python

"""
download_msrvtt.py

Download MSRVTT (test_1k) from HuggingFace and organize it for TRIANGLE:

Creates (automatically, if missing):

    <root>/video_test/
    <root>/audio_test/

Videos saved as:
    <root>/video_test/videoXXXX.mp4

Audio extracted via ffmpeg as:
    <root>/audio_test/videoXXXX.mp3

Usage:
    python download_msrvtt.py --root datasets/MSRVTT --split test_1k
"""

import argparse
import os
import pathlib
import subprocess
from datasets import load_dataset


def run(cmd):
    """Helper: run shell commands with printed output."""
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def safe_mkdir(path):
    """Create directory if missing."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_video_path(v):
    """Extract underlying file path from HF Video object/dict."""
    for attr in ("path", "filename"):
        if hasattr(v, attr):
            return getattr(v, attr)
    if isinstance(v, dict):
        return v.get("path") or v.get("filename")
    return None


def extract_audio(video_dir, audio_dir):
    """Extract mp3 audio for each video using ffmpeg."""
    safe_mkdir(audio_dir)

    for fname in sorted(os.listdir(video_dir)):
        if not fname.lower().endswith((".mp4", ".webm", ".mkv", ".avi")):
            continue

        stem = pathlib.Path(fname).stem
        in_path = os.path.join(video_dir, fname)
        out_path = os.path.join(audio_dir, stem + ".mp3")

        if os.path.exists(out_path):
            print(f"[audio] exists, skipping: {out_path}")
            continue

        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-vn", "-acodec", "libmp3lame", out_path
        ]
        run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        required=True,
        help="Directory where MSRVTT dataset will be created (video_test/ and audio_test/)."
    )
    parser.add_argument(
        "--split",
        default="test_1k",
        help="HuggingFace configuration (default: test_1k)"
    )
    args = parser.parse_args()

    # --- Ensure directories exist ---
    root = os.path.abspath(args.root)
    video_dir = os.path.join(root, "video_test")
    audio_dir = os.path.join(root, "audio_test")

    safe_mkdir(root)
    safe_mkdir(video_dir)

    # --- Download dataset ---
    print(f"Downloading MSRVTT ({args.split}) from HuggingFace...")
    ds = load_dataset("friedrichor/MSR-VTT", args.split)

    split_name = "test"
    if split_name not in ds:
        raise RuntimeError(f"Expected 'test' split. Got: {list(ds.keys())}")

    # --- Copy videos ---
    for row in ds[split_name]:
        vid_id = row["video_id"]
        v = row["video"]
        src = get_video_path(v)

        if src is None:
            print(f"[WARN] No video path for {vid_id}, skipping.")
            continue

        ext = pathlib.Path(src).suffix or ".mp4"
        dst = os.path.join(video_dir, vid_id + ext)

        if os.path.exists(dst):
            print(f"[video] exists, skipping: {dst}")
            continue

        print(f"[video] copying: {src} -> {dst}")
        run(["cp", src, dst])

    # --- Extract audio ---
    print(f"\nExtracting audio into: {audio_dir}")
    extract_audio(video_dir, audio_dir)

    print("\nMSRVTT download complete.")
    print(f"Videos saved in: {video_dir}")
    print(f"Audio saved in:  {audio_dir}")


if __name__ == "__main__":
    main()
