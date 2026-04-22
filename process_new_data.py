"""
process_new_data.py — Deduplicate, split, and merge new data into the existing dataset.

Workflow:
  1. Recursively find every image inside new_dataset/ (handles nested dirs)
  2. Hash every image already in datasets/train/ and datasets/valid/
  3. Skip duplicates (including intra-batch) to prevent data leakage
  4. Split unique new images 80/20 into train/valid
  5. Copy them into datasets/train/ and datasets/valid/
  6. Print a summary report

Usage:
    python process_new_data.py
    python process_new_data.py --new-dir my_new_data/
    python process_new_data.py --ratio 0.15 --seed 99
"""

import argparse
import hashlib
import os
import random
import shutil

# ── Config ──────────────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DATASETS_DIRS = [
    os.path.join("datasets", "train", "images"),
    os.path.join("datasets", "valid", "images"),
]

TRAIN_IMAGES = os.path.join("datasets", "train", "images")
TRAIN_LABELS = os.path.join("datasets", "train", "labels")
VALID_IMAGES = os.path.join("datasets", "valid", "images")
VALID_LABELS = os.path.join("datasets", "valid", "labels")


# ── Helpers ─────────────────────────────────────────────────────────

def md5_file(filepath: str) -> str:
    """Compute the MD5 hash of a file (reads in 8 KB chunks)."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def is_image(filename: str) -> bool:
    """Check if a filename has a recognised image extension."""
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTS


def collect_hashes(directory: str) -> set:
    """Return a set of MD5 hashes for every image in `directory`."""
    hashes = set()
    if not os.path.isdir(directory):
        return hashes
    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath) and is_image(fname):
            hashes.add(md5_file(fpath))
    return hashes


def find_label_for_image(image_path: str) -> str | None:
    """
    Given an image path, find the corresponding .txt label file.

    Handles two common layouts:
      A) Flat — label sits next to the image:
           some_dir/image001.jpg  →  some_dir/image001.txt
      B) images/labels split — label is in a sibling 'labels/' folder:
           .../images/image001.jpg  →  .../labels/image001.txt
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    parent = os.path.dirname(image_path)

    # Strategy B: sibling labels/ directory
    if os.path.basename(parent).lower() == "images":
        labels_dir = os.path.join(os.path.dirname(parent), "labels")
        candidate = os.path.join(labels_dir, stem + ".txt")
        if os.path.isfile(candidate):
            return candidate

    # Strategy A: label next to image
    candidate = os.path.join(parent, stem + ".txt")
    if os.path.isfile(candidate):
        return candidate

    return None


def discover_images(root_dir: str) -> list[tuple[str, str]]:
    """
    Recursively walk `root_dir` and return a list of
    (image_path, label_path) for every image that has a matching label.

    Images without a label are reported but skipped.
    """
    pairs = []
    no_label = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if not is_image(fname):
                continue
            img_path = os.path.join(dirpath, fname)
            lbl_path = find_label_for_image(img_path)
            if lbl_path is None:
                no_label += 1
                continue
            pairs.append((img_path, lbl_path))

    return pairs, no_label


# ── CLI ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deduplicate & merge new data into dataset")
    p.add_argument("--new-dir", type=str, default="new_dataset",
                    help="Folder containing new raw images + labels")
    p.add_argument("--ratio", type=float, default=0.20,
                    help="Fraction of new images to assign to validation (default: 0.20)")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducible splits")
    return p.parse_args()


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    new_dir = args.new_dir

    # ── 1. Validate paths ───────────────────────────────────────────
    if not os.path.isdir(new_dir):
        print(f"[ERROR] New data folder not found: {new_dir}")
        print("        Place new images + labels there and re-run.")
        return

    for d in [TRAIN_IMAGES, TRAIN_LABELS, VALID_IMAGES, VALID_LABELS]:
        os.makedirs(d, exist_ok=True)

    # ── 2. Hash existing dataset images ─────────────────────────────
    print("[1/5] Hashing existing dataset images ...")
    existing_hashes = set()
    for d in DATASETS_DIRS:
        h = collect_hashes(d)
        print(f"       {d}: {len(h)} images")
        existing_hashes.update(h)
    print(f"       Total existing hashes: {len(existing_hashes)}")

    # ── 3. Discover new images recursively & deduplicate ────────────
    print("\n[2/5] Scanning new data & deduplicating ...")
    all_pairs, skipped_no_label = discover_images(new_dir)
    print(f"       Found {len(all_pairs)} image+label pairs in '{new_dir}'")

    unique_pairs = []
    duplicates = 0

    for img_path, lbl_path in all_pairs:
        file_hash = md5_file(img_path)
        if file_hash in existing_hashes:
            duplicates += 1
            continue
        # Track hash to catch intra-batch duplicates too
        existing_hashes.add(file_hash)
        unique_pairs.append((img_path, lbl_path))

    total_unique = len(unique_pairs)
    print(f"       Duplicates skipped    : {duplicates}")
    print(f"       No label (skipped)    : {skipped_no_label}")
    print(f"       Unique new images     : {total_unique}")

    if total_unique == 0:
        print("\n[DONE] No new unique images to add.")
        return

    # ── 4. Split 80/20 ──────────────────────────────────────────────
    print(f"\n[3/5] Splitting {total_unique} images "
          f"({1 - args.ratio:.0%} train / {args.ratio:.0%} valid) ...")
    random.shuffle(unique_pairs)
    val_count = max(1, int(total_unique * args.ratio))
    val_set   = unique_pairs[:val_count]
    train_set = unique_pairs[val_count:]

    print(f"       Train: {len(train_set)}")
    print(f"       Valid: {len(val_set)}")

    # ── 5. Copy files into datasets/ ────────────────────────────────
    def copy_files(file_list, img_dst, lbl_dst, label):
        """Copy image+label pairs into their destination directories."""
        moved = 0
        for img_path, lbl_path in file_list:
            img_name = os.path.basename(img_path)
            lbl_name = os.path.splitext(img_name)[0] + ".txt"

            dst_img = os.path.join(img_dst, img_name)
            dst_lbl = os.path.join(lbl_dst, lbl_name)

            # Handle filename collisions by appending a suffix
            if os.path.exists(dst_img):
                stem = os.path.splitext(img_name)[0]
                ext  = os.path.splitext(img_name)[1]
                counter = 1
                while os.path.exists(dst_img):
                    new_name = f"{stem}_{counter}"
                    dst_img = os.path.join(img_dst, new_name + ext)
                    dst_lbl = os.path.join(lbl_dst, new_name + ".txt")
                    counter += 1

            shutil.copy2(img_path, dst_img)
            shutil.copy2(lbl_path, dst_lbl)
            moved += 1
        print(f"       {label}: {moved} images + labels copied")
        return moved

    print(f"\n[4/5] Merging into datasets/ ...")
    train_added = copy_files(train_set, TRAIN_IMAGES, TRAIN_LABELS, "train")
    valid_added = copy_files(val_set,   VALID_IMAGES, VALID_LABELS, "valid")

    # ── 6. Summary ──────────────────────────────────────────────────
    final_train = len([f for f in os.listdir(TRAIN_IMAGES) if is_image(f)])
    final_valid = len([f for f in os.listdir(VALID_IMAGES) if is_image(f)])

    print(f"\n[5/5] Summary Report")
    print("=" * 50)
    print(f"  Duplicates skipped    : {duplicates}")
    print(f"  No label (skipped)    : {skipped_no_label}")
    print(f"  New images -> train   : {train_added}")
    print(f"  New images -> valid   : {valid_added}")
    print("-" * 50)
    print(f"  Final train/images    : {final_train}")
    print(f"  Final valid/images    : {final_valid}")
    print(f"  Total dataset size    : {final_train + final_valid}")
    print("=" * 50)
    print("\n[DONE] New data merged successfully.")


if __name__ == "__main__":
    main()
