"""
merge_datasets.py — Merge old + new datasets, then split 80/20.

Merges:
  datasets/old_dataset/train/{images,labels}/  (487 files)
  datasets/train/{images,labels}/              (308 files)
  datasets/valid/{images,labels}/              (76 files — moved back to train first)

Into a single pool, then re-splits into train/valid (80/20).
"""

import os
import random
import shutil

# ── Config ──────────────────────────────────────────────────────────
MERGED_DIR   = os.path.join("datasets", "_merged")
MERGED_IMG   = os.path.join(MERGED_DIR, "images")
MERGED_LBL   = os.path.join(MERGED_DIR, "labels")

FINAL_TRAIN_IMG = os.path.join("datasets", "train", "images")
FINAL_TRAIN_LBL = os.path.join("datasets", "train", "labels")
FINAL_VALID_IMG = os.path.join("datasets", "valid", "images")
FINAL_VALID_LBL = os.path.join("datasets", "valid", "labels")

DATA_YAML = os.path.join("config", "data.yaml")

SOURCES = [
    ("datasets/old_dataset/train/images", "datasets/old_dataset/train/labels"),
    ("datasets/train/images",             "datasets/train/labels"),
    ("datasets/valid/images",             "datasets/valid/labels"),
]

RATIO = 0.20
SEED  = 42


def copy_files(img_src, lbl_src, img_dst, lbl_dst, prefix):
    """Copy images + labels from source to destination, prefixing to avoid name clashes."""
    count = 0
    if not os.path.isdir(img_src):
        return count

    for f in os.listdir(img_src):
        name, ext = os.path.splitext(f)
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue

        new_name = f"{prefix}_{name}"
        # Copy image
        shutil.copy2(os.path.join(img_src, f), os.path.join(img_dst, new_name + ext))
        # Copy label
        lbl_file = os.path.join(lbl_src, name + ".txt")
        if os.path.isfile(lbl_file):
            shutil.copy2(lbl_file, os.path.join(lbl_dst, new_name + ".txt"))
        count += 1
    return count


def main():
    random.seed(SEED)

    # ── 1. Create merged staging area ───────────────────────────────
    os.makedirs(MERGED_IMG, exist_ok=True)
    os.makedirs(MERGED_LBL, exist_ok=True)

    print("Merging datasets into staging area...")
    total = 0
    for i, (img_src, lbl_src) in enumerate(SOURCES):
        prefix = f"ds{i}"
        n = copy_files(img_src, lbl_src, MERGED_IMG, MERGED_LBL, prefix)
        print(f"  [{prefix}] {img_src}: {n} images copied")
        total += n

    print(f"\n  Total merged: {total} images")

    # ── 2. Collect all image stems ──────────────────────────────────
    all_files = []
    for f in os.listdir(MERGED_IMG):
        name, ext = os.path.splitext(f)
        if ext.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            all_files.append((name, ext))
    all_files.sort()

    # ── 3. Shuffle & split ──────────────────────────────────────────
    random.shuffle(all_files)
    val_count = max(1, int(len(all_files) * RATIO))
    val_files   = all_files[:val_count]
    train_files = all_files[val_count:]

    print(f"\n  Split: train={len(train_files)}, valid={val_count} ({RATIO:.0%})")

    # ── 4. Clear old train/valid and populate ───────────────────────
    for d in [FINAL_TRAIN_IMG, FINAL_TRAIN_LBL, FINAL_VALID_IMG, FINAL_VALID_LBL]:
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    def copy_set(file_list, img_dst, lbl_dst):
        for stem, ext in file_list:
            shutil.copy2(os.path.join(MERGED_IMG, stem + ext), os.path.join(img_dst, stem + ext))
            lbl = os.path.join(MERGED_LBL, stem + ".txt")
            if os.path.isfile(lbl):
                shutil.copy2(lbl, os.path.join(lbl_dst, stem + ".txt"))

    copy_set(train_files, FINAL_TRAIN_IMG, FINAL_TRAIN_LBL)
    copy_set(val_files,   FINAL_VALID_IMG, FINAL_VALID_LBL)

    # ── 5. Cleanup staging + old_dataset ────────────────────────────
    shutil.rmtree(MERGED_DIR, ignore_errors=True)
    old_ds = os.path.join("datasets", "old_dataset")
    if os.path.isdir(old_ds):
        shutil.rmtree(old_ds)
        print("  Cleaned up old_dataset/")

    # ── 6. Rewrite data.yaml ────────────────────────────────────────
    yaml_content = """\
# ============================================================
# YOLOv8 Dataset Config  (merged dataset — auto-generated)
# ============================================================

path: ../datasets
train: train/images
val: valid/images

nc: 2
names:
  0: Damaged
  1: package
"""
    os.makedirs(os.path.dirname(DATA_YAML), exist_ok=True)
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # ── 7. Verify ───────────────────────────────────────────────────
    t_img = len(os.listdir(FINAL_TRAIN_IMG))
    v_img = len(os.listdir(FINAL_VALID_IMG))
    print(f"\n  Final counts:")
    print(f"    train/images: {t_img}")
    print(f"    valid/images: {v_img}")
    print(f"    total:        {t_img + v_img}")
    print(f"\n  Updated {DATA_YAML}")
    print("  Done!")


if __name__ == "__main__":
    main()
