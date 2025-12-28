#!/usr/bin/env python
"""
download_vast.py

Download and extract VAST pretrained model (tar.gz) into:

output/
   vast/
      pretrain_vast/
      vision_captioner/
      audio_captioner/
"""

import argparse
import os
import subprocess
import sys


def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=".", help="Root directory of your repo")
    args = parser.parse_args()

    base = os.path.abspath(args.base_dir)

    # required output structure
    vast_root = os.path.join(base, "output", "vast")
    os.makedirs(vast_root, exist_ok=True)

    # final file path
    tar_path = os.path.join(vast_root, "vast_pretrained.tar.gz")

    print("[INFO] Preparing to download VAST pretrained model...")

    # ensure gdown exists
    try:
        import gdown
    except ImportError:
        print("[INFO] Installing gdown ...")
        run([sys.executable, "-m", "pip", "install", "--user", "gdown"])
        import gdown

    # Google Drive ID
    file_id = "1ZkeZpis2Fggy4MyTFPqQj37MgPZxdJ53"
    url = f"https://drive.google.com/uc?id={file_id}"

    # download tar.gz
    if not os.path.exists(tar_path):
        print(f"[INFO] Downloading tar.gz file from: {url}")
        gdown.download(url, tar_path, quiet=False)
    else:
        print("[INFO] File already exists. Skipping download.")

    # extract tar.gz
    print("[INFO] Extracting tar.gz file...")
    run(["tar", "-xzf", tar_path, "-C", vast_root])

    print("\n[INFO] DONE — Extracted structure should now be:")
    print("output/vast/pretrain_vast/")
    print("output/vast/vision_captioner/")
    print("output/vast/audio_captioner/")
    print()


if __name__ == "__main__":
    main()
