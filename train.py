"""
train.py — Train a YOLOv8 model on the Parcel / Damage dataset.

Usage
-----
    python train.py                         # defaults (yolov8n, 100 epochs)
    python train.py --model yolov8s.pt --epochs 200 --batch 32

The script uses the Ultralytics high-level API exclusively — no custom
training loops, no manual loss calculation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


# ── Defaults ─────────────────────────────────────────────────────────
DEFAULT_MODEL = "yolov8n.pt"               # Nano — fast prototyping
DEFAULT_DATA  = "config/data.yaml"
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ  = 640
DEFAULT_BATCH  = 16
DEFAULT_DEVICE = ""                        # "" → auto-select GPU / CPU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 training for Parcel Damage Detection",
    )
    parser.add_argument("--model",   type=str, default=DEFAULT_MODEL,  help="Pre-trained weights to fine-tune")
    parser.add_argument("--data",    type=str, default=DEFAULT_DATA,   help="Path to data.yaml")
    parser.add_argument("--epochs",  type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--imgsz",   type=int, default=DEFAULT_IMGSZ,  help="Input image size")
    parser.add_argument("--batch",   type=int, default=DEFAULT_BATCH,  help="Batch size")
    parser.add_argument("--device",  type=str, default=DEFAULT_DEVICE, help="Device: '' (auto), '0', 'cpu'")
    parser.add_argument("--name",    type=str, default="train",        help="Run name (saved under runs/detect/)")
    parser.add_argument("--resume",  action="store_true",              help="Resume from last checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Load model ───────────────────────────────────────────────
    print(f"\n📦  Loading model: {args.model}")
    model = YOLO(args.model)

    # ── 2. Train ────────────────────────────────────────────────────
    print(f"🚀  Starting training — {args.epochs} epochs @ {args.imgsz}px\n")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        name=args.name,
        resume=args.resume,
        # ── Augmentation & scheduling (sensible defaults) ────────
        hsv_h=0.015,          # hue shift
        hsv_s=0.7,            # saturation shift
        hsv_v=0.4,            # value shift
        degrees=5.0,          # rotation ±5°
        translate=0.1,        # translation fraction
        scale=0.5,            # scale gain
        flipud=0.0,           # no vertical flip (parcels have orientation)
        fliplr=0.5,           # horizontal flip
        mosaic=1.0,           # mosaic augmentation
        patience=20,          # early-stopping patience
        save=True,
        save_period=10,       # checkpoint every 10 epochs
        plots=True,           # generate training curves
        verbose=True,
    )

    # ── 3. Validate ─────────────────────────────────────────────────
    print("\n✅  Training complete — running validation …")
    metrics = model.val()
    print(f"    mAP50  : {metrics.box.map50:.4f}")
    print(f"    mAP50-95: {metrics.box.map:.4f}")

    best_weights = Path(f"runs/detect/{args.name}/weights/best.pt")
    print(f"\n🏆  Best weights saved to: {best_weights.resolve()}\n")


if __name__ == "__main__":
    main()
