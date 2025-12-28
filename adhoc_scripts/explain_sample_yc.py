import os
import json
import argparse
import random
import warnings
import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.utils_for_fast_inference import (
    get_args,
    VisionMapper,
    AudioMapper,
    build_batch
)
from model.build_model import build_model
from utils.utils import area_computation

warnings.filterwarnings("ignore")
os.environ["LOCAL_RANK"] = "0"


# ===============================================================
#                    ARGUMENT PARSER
# ===============================================================
def get_script_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrain_dir", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)

    # NEW: optional list of video_ids (comma-separated)
    parser.add_argument(
        "--video_ids",
        type=str,
        default=None,
        help="Comma-separated list of video_ids to restrict sampling"
    )

    return parser.parse_args()


# ===============================================================
#                    MAIN SCRIPT
# ===============================================================
def main():
    script_args = get_script_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------
    # Load model args + model
    # ---------------------------------------------------------------
    args = get_args(script_args.pretrain_dir)

    model, _, _ = build_model(args, device=device)
    model.to("cuda")
    model.eval()

    visionMapper = VisionMapper(args.data_cfg.train[0], args)
    audioMapper = AudioMapper(args.data_cfg.train[0], args)

    tasks = "ret%tva"

    # ---------------------------------------------------------------
    # Load JSON annotations
    # ---------------------------------------------------------------
    with open(script_args.json_path, "r") as f:
        annotations = json.load(f)

    print(f"Loaded {len(annotations)} annotations.")

    # ---------------------------------------------------------------
    # Optional filtering by video_ids
    # ---------------------------------------------------------------
    if script_args.video_ids is not None:
        allowed_ids = set(v.strip() for v in script_args.video_ids.split(","))
        annotations = [
            ann for ann in annotations
            if ann["video_id"] in allowed_ids
        ]
        print(f"Filtered to {len(annotations)} annotations using video_ids.")

        if len(annotations) == 0:
            raise RuntimeError("No annotations left after video_id filtering.")

    # ---------------------------------------------------------------
    # Random sampling
    # ---------------------------------------------------------------
    random.seed(42)
    sampled = random.sample(
        annotations,
        min(script_args.num_samples, len(annotations))
    )

    text_list = []
    video_list = []
    audio_list = []
    used_ids = []
    used_paths = []

    # ---------------------------------------------------------------
    # Resolve paths and validate existence
    # ---------------------------------------------------------------
    for ann in sampled:
        vid = ann["video_id"]
        desc = ann["desc"]

        video_path = os.path.join(script_args.video_dir, f"{vid}.mp4")
        audio_path = os.path.join(script_args.audio_dir, f"{vid}.mp3")

        if not os.path.exists(video_path):
            print(f"[WARNING] Missing video: {video_path}")
            continue

        if not os.path.exists(audio_path):
            print(f"[WARNING] Missing audio: {audio_path}")
            continue

        text_list.append(desc)
        video_list.append(video_path)
        audio_list.append(audio_path)

        used_ids.append(vid)
        used_paths.append({
            "video": video_path,
            "audio": audio_path
        })

    print(f"Final usable samples: {len(text_list)}")
    print("IDs used:", used_ids)

    if len(text_list) == 0:
        raise RuntimeError("No valid samples found. Check directory paths.")

    # ---------------------------------------------------------------
    # Build batch and run inference
    # ---------------------------------------------------------------
    batch = build_batch(args, text_list, video_list, audio_list)

    with torch.no_grad():
        evaluation_dict = model(batch, tasks, compute_loss=False)

    feat_t = evaluation_dict["feat_t"]
    feat_v = evaluation_dict["feat_v"]
    feat_a = evaluation_dict["feat_a"]

    # ---------------------------------------------------------------
    # Save features + metadata
    # ---------------------------------------------------------------
    save_dir = "./saved_features_sample"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(feat_t.cpu(), os.path.join(save_dir, "feat_text.pt"))
    torch.save(feat_v.cpu(), os.path.join(save_dir, "feat_video.pt"))
    torch.save(feat_a.cpu(), os.path.join(save_dir, "feat_audio.pt"))

    with open(os.path.join(save_dir, "ids.json"), "w") as f:
        json.dump(used_ids, f, indent=2)

    with open(os.path.join(save_dir, "captions.json"), "w") as f:
        json.dump(text_list, f, indent=2)

    with open(os.path.join(save_dir, "paths.json"), "w") as f:
        json.dump(used_paths, f, indent=2)

    print("Saved features and metadata to:", save_dir)
    print("Feature shapes:", feat_t.shape, feat_v.shape, feat_a.shape)

    # ---------------------------------------------------------------
    # Compute AREA and save
    # ---------------------------------------------------------------
    area = area_computation(feat_t, feat_v, feat_a)
    area_cpu = area.detach().cpu()

    torch.save(area_cpu, os.path.join(save_dir, "area.pt"))
    print("AREA matrix saved.")


# ===============================================================
#                    ENTRY POINT
# ===============================================================
if __name__ == "__main__":
    main()
