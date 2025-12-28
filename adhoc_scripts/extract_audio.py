#!/usr/bin/env python

"""
extract_msrvtt_audio.py

Extract audio tracks from MSRVTT videos and save as MP3 files.

- Input:  <video_root>/videoXXXX.ext  (mp4, avi, etc.)
- Output: <audio_root>/videoXXXX.mp3

Handles:
- Skips videos that have NO audio stream (common in MSR-VTT)
- Skips audio files that already exist

Usage:
  python extract_msrvtt_audio.py \
      --video_root datasets/MSRVTT/video \
      --audio_root datasets/MSRVTT/audio
"""

import argparse
import os
import pathlib
import subprocess


def run(cmd):
    """Run a shell command and print it."""
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def has_audio_stream(path):
    """
    Return True if ffprobe finds at least one audio stream in the file.
    Requires ffprobe (part of ffmpeg) to be available on PATH.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip() != ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_root",
        required=True,
        help="Directory containing MSRVTT videos (e.g. datasets/MSRVTT/video).",
    )
    parser.add_argument(
        "--audio_root",
        required=True,
        help="Directory to write extracted audio (e.g. datasets/MSRVTT/audio).",
    )
    args = parser.parse_args()

    video_root = os.path.abspath(args.video_root)
    audio_root = os.path.abspath(args.audio_root)

    safe_mkdir(audio_root)

    # Accepted video extensions
    exts = (".mp4", ".webm", ".mkv", ".avi", ".mov")

    files = sorted(
        f for f in os.listdir(video_root)
        if f.lower().endswith(exts)
    )

    if not files:
        print(f"[WARN] No video files found in {video_root}")
        return

    print(f"Found {len(files)} video files in {video_root}")
    print(f"Extracting audio into {audio_root}")

    n_audio = 0
    n_no_audio = 0

    for fname in files:
        stem = pathlib.Path(fname).stem          # e.g. "video7020"
        in_path = os.path.join(video_root, fname)
        out_path = os.path.join(audio_root, stem + ".mp3")

        if os.path.exists(out_path):
            print(f"[audio] exists, skipping: {out_path}")
            n_audio += 1
            continue

        # Check if file has at least one audio stream
        if not has_audio_stream(in_path):
            print(f"[WARN] No audio stream in {in_path}, skipping.")
            n_no_audio += 1
            continue

        cmd = [
            "ffmpeg",
            "-y",
            "-i", in_path,
            "-vn",
            "-acodec", "libmp3lame",
            out_path,
        ]
        run(cmd)
        n_audio += 1

    print("\nDone extracting audio.")
    print(f"Videos with audio extracted: {n_audio}")
    print(f"Videos without any audio:    {n_no_audio}")
    print(f"Audio files saved in:        {audio_root}")


if __name__ == "__main__":
    main()
