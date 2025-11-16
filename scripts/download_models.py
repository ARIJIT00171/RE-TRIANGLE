#!/usr/bin/env python
"""
download_models.py

Download TRIANGLE model assets on Snellius:

- EVA-CLIP (video encoder)
- BERT base (text encoder)
- TRIANGLE pretraining checkpoint (triangle_pretraining.zip from Google Drive)

BEATs is NOT handled here (you said you'll upload it manually).

Usage:
  python download_models.py --base_dir .

Assumptions:
  - You already have `transformers` and `gdown` installed in your env.
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
    parser.add_argument(
        "--base_dir",
        default=".",
        help="Root dir of TRIANGLE repo (where pretrained_weights/ and triangle_pretraining/ will live)",
    )
    args = parser.parse_args()

    base = os.path.abspath(args.base_dir)
    os.makedirs(base, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1) Create directory structure
    # -------------------------------------------------------------------------
    pw_dir = os.path.join(base, "pretrained_weights")
    clip_dir = os.path.join(pw_dir, "clip")
    beats_dir = os.path.join(pw_dir, "beats")
    bert_dir = os.path.join(pw_dir, "bert")
    for d in (clip_dir, beats_dir, bert_dir):
        os.makedirs(d, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2) EVA-CLIP (video encoder)
    # -------------------------------------------------------------------------
    evaclip_url = "https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA01_CLIP_g_14_psz14_s11B.pt"
    evaclip_out = os.path.join(clip_dir, "EVA01_CLIP_g_14_psz14_s11B.pt")
    if not os.path.exists(evaclip_out):
        print(f"Downloading EVA-CLIP to {evaclip_out}")
        run(["wget", "-O", evaclip_out, evaclip_url])
    else:
        print("EVA-CLIP already exists, skipping.")

    # -------------------------------------------------------------------------
    # 3) BEATs (audio encoder) – skipped, you upload manually
    # -------------------------------------------------------------------------
    beats_out = os.path.join(beats_dir, "BEATs_iter3_plus_AS2M.pt")
    if os.path.exists(beats_out):
        print(f"BEATs model already present at {beats_out}.")
    else:
        print(
            f"[INFO] BEATs model NOT downloaded here. Please manually place "
            f"'BEATs_iter3_plus_AS2M.pt' at: {beats_out}"
        )

    # -------------------------------------------------------------------------
    # 4) BERT base (text encoder) via transformers
    # -------------------------------------------------------------------------
    try:
        from transformers import BertModel, BertTokenizer
    except ImportError as e:
        print(
            "[ERROR] transformers not installed in your environment. "
            "Please install it in your env before running this script."
        )
        raise e

    bert_subdir = os.path.join(bert_dir, "bert-base-uncased")
    if not os.path.exists(os.path.join(bert_subdir, "config.json")):
        print(f"Downloading bert-base-uncased into {bert_subdir}")
        bert = BertModel.from_pretrained("bert-base-uncased")
        tok = BertTokenizer.from_pretrained("bert-base-uncased")
        bert.save_pretrained(bert_subdir)
        tok.save_pretrained(bert_subdir)
    else:
        print("BERT weights already present, skipping.")

    # -------------------------------------------------------------------------
    # 5) TRIANGLE pretraining checkpoint (VAST-150k)
    # -------------------------------------------------------------------------
    ckpt_root = os.path.join(base, "triangle_pretraining")
    if not os.path.exists(ckpt_root):
        print("Downloading triangle_pretraining from Google Drive with gdown")
        try:
            import gdown  # type: ignore
        except ImportError:
            run([sys.executable, "-m", "pip", "install", "--user", "gdown"])
            import gdown  # type: ignore

        url = "https://drive.google.com/uc?id=1T-wuY-CzUp_PF8UuhKDqXEAL86obUpzj"
        zip_path = os.path.join(base, "triangle_pretraining.zip")
        if not os.path.exists(zip_path):
            gdown.download(url, zip_path, quiet=False)
        print("Unzipping triangle_pretraining.zip ...")
        run(["unzip", "-d", base, zip_path])
    else:
        print("triangle_pretraining folder already exists, skipping.")


if __name__ == "__main__":
    main()