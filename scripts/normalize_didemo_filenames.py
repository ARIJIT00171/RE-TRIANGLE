import os
import json


def load_video_ids(annotation_dir):
    ann_files = [
        "descs_ret_train.json",
        "descs_ret_test.json",
    ]
    ids = set()
    for fname in ann_files:
        path = os.path.join(annotation_dir, fname)
        if not os.path.exists(path):
            print(f"[WARN] Annotation file not found: {path}")
            continue
        with open(path, "r") as f:
            data = json.load(f)
        for item in data:
            vid = item.get("video_id")
            if vid:
                ids.add(vid)
    print(f"Loaded {len(ids)} unique video_ids from annotations.")
    return ids


def normalize_dir(root, ids, target_ext):
    """
    root: directory with files (video or audio).
    ids: set of valid video_ids from annotations.
    target_ext: '.mp4' for video, '.mp3' for audio.
    """
    root = os.path.abspath(root)
    print(f"Normalizing files in: {root}")
    if not os.path.isdir(root):
        print(f"[WARN] Directory does not exist: {root}")
        return

    renamed = 0
    skipped = 0

    for fname in os.listdir(root):
        old_path = os.path.join(root, fname)
        if not os.path.isfile(old_path):
            continue

        # Part before the first dot -> expected video_id
        base = fname.split(".")[0]

        if base not in ids:
            skipped += 1
            continue

        new_name = base + target_ext
        new_path = os.path.join(root, new_name)

        if old_path == new_path:
            # Already normalized
            continue

        if os.path.exists(new_path):
            print(f"[WARN] Target already exists, skipping rename: {new_path}")
            skipped += 1
            continue

        print(f"Renaming: {fname} -> {new_name}")
        os.rename(old_path, new_path)
        renamed += 1

    print(f"Done in {root}: renamed={renamed}, skipped={skipped}")


def main():
    # Treat current working directory as TRIANGLE root
    triangle_root = os.path.abspath(os.getcwd())
    print(f"Triangle root assumed as: {triangle_root}")

    ann_dir = os.path.join(triangle_root, "datasets", "annotations", "didemo")
    video_root = os.path.join(triangle_root, "datasets", "DiDeMo", "video")
    audio_root = os.path.join(triangle_root, "datasets", "DiDeMo", "audio")

    print(f"Annotation dir: {ann_dir}")
    ids = load_video_ids(ann_dir)

    # Normalize videos to <video_id>.mp4
    normalize_dir(video_root, ids, ".mp4")

    # Normalize audios to <video_id>.mp3
    normalize_dir(audio_root, ids, ".mp3")


if __name__ == "__main__":
    main()