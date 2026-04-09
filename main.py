"""
main.py — Video inference pipeline with YOLOv8 tracking & severity analysis.

The script reads a video file, runs YOLOv8 object tracking (ByteTrack) on
every frame, calculates damage severity relative to the parent parcel, and
writes an annotated video to disk.

Usage
-----
    python main.py                                           # use settings.yaml defaults
    python main.py --input data/videos/clip.mp4 --output output/clip_result.mp4
    python main.py --weights runs/detect/train/weights/best.pt --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import yaml
from ultralytics import YOLO

from src.severity import Detection, classify_severity, find_parent_parcel
from src.visualiser import draw_damage, draw_hud, draw_parcel


# ── Config helpers ───────────────────────────────────────────────────
def load_settings(path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load project settings from YAML, falling back to empty dict."""
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Nested dict getter: _get(cfg, 'model', 'confidence', default=0.35)."""
    node = cfg
    for k in keys:
        if isinstance(node, dict):
            node = node.get(k, default)
        else:
            return default
    return node


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args(cfg: Dict[str, Any]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parcel Damage Detection — Video Inference")
    p.add_argument("--input",      type=str, default=_get(cfg, "video", "input",             default="data/videos/sample.mp4"))
    p.add_argument("--output",     type=str, default=_get(cfg, "video", "output",            default="output/result.mp4"))
    p.add_argument("--weights",    type=str, default=_get(cfg, "model", "weights",           default="runs/detect/train/weights/best.pt"))
    p.add_argument("--conf",       type=float, default=_get(cfg, "model", "confidence",      default=0.35))
    p.add_argument("--iou",        type=float, default=_get(cfg, "model", "iou_threshold",   default=0.45))
    p.add_argument("--imgsz",      type=int,   default=_get(cfg, "model", "imgsz",           default=640))
    p.add_argument("--device",     type=str,   default=_get(cfg, "model", "device",          default=""))
    p.add_argument("--tracker",    type=str,   default=_get(cfg, "tracker", "config",        default="bytetrack.yaml"))
    p.add_argument("--show",       action="store_true", default=_get(cfg, "video", "show_live", default=False))
    p.add_argument("--no-save",    action="store_true", help="Skip saving output video")
    return p.parse_args()


# ── Detection extraction ────────────────────────────────────────────
def extract_detections(result: Any) -> List[Detection]:
    """
    Convert a single Ultralytics Results object into a flat list of
    ``Detection`` dataclass instances.
    """
    detections: List[Detection] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    # .xyxy → (x1,y1,x2,y2), .cls → class id, .conf → confidence
    xyxy   = boxes.xyxy.cpu().numpy()
    cls    = boxes.cls.cpu().numpy().astype(int)
    conf   = boxes.conf.cpu().numpy()
    # Track IDs may be None on the first frame or if tracking is lost
    ids    = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(cls)

    for i in range(len(cls)):
        det = Detection(
            class_id=int(cls[i]),
            track_id=int(ids[i]) if ids[i] is not None else None,
            bbox=tuple(int(v) for v in xyxy[i]),
            confidence=float(conf[i]),
        )
        detections.append(det)

    return detections


# ── Main pipeline ────────────────────────────────────────────────────
def run(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    # --- Severity thresholds from config ---
    minor_max    = float(_get(cfg, "severity", "minor_max",    default=0.10))
    moderate_max = float(_get(cfg, "severity", "moderate_max", default=0.25))
    parcel_cls   = int(_get(cfg, "classes", "parcel", default=0))
    damage_cls   = int(_get(cfg, "classes", "damage", default=1))

    # ── 1. Load model ───────────────────────────────────────────────
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"❌  Weights not found: {weights_path.resolve()}")
        print("    Train a model first with  python train.py")
        sys.exit(1)

    print(f"📦  Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    # ── 2. Open input video ─────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"❌  Cannot open video: {args.input}")
        sys.exit(1)

    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎬  Input : {args.input}  ({w}×{h} @ {fps:.1f} FPS, {total} frames)")

    # ── 3. Prepare output writer ────────────────────────────────────
    writer: Optional[cv2.VideoWriter] = None
    if not args.no_save:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        codec = _get(cfg, "video", "codec", default="mp4v")
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        print(f"💾  Output: {out_path.resolve()}")

    # ── 4. Frame-by-frame processing ────────────────────────────────
    frame_idx = 0
    print("\n▶  Processing …\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # --- Run tracking inference ---
        results = model.track(
            source=frame,
            persist=True,                         # keep track IDs across frames
            tracker=args.tracker,                  # bytetrack.yaml
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device if args.device else None,
            verbose=False,                         # suppress per-frame logs
        )

        # --- Parse detections ---
        detections = extract_detections(results[0])
        parcels = [d for d in detections if d.class_id == parcel_cls]
        damages = [d for d in detections if d.class_id == damage_cls]

        # --- Draw intact parcels ---
        for p in parcels:
            draw_parcel(frame, p)

        # --- Severity analysis & draw damages ---
        # Strategy: If a Damaged box overlaps a package box, use area ratio.
        # If Damaged is detected alone (common — dataset uses mutually exclusive
        # labels), estimate severity from detection confidence instead.
        for dmg in damages:
            parent = find_parent_parcel(dmg, parcels, overlap_threshold=0.3)
            if parent is not None:
                # Case 1: Both Damaged region + parent package detected
                severity, ratio = classify_severity(
                    dmg.area, parent.area,
                    minor_max=minor_max,
                    moderate_max=moderate_max,
                )
            else:
                # Case 2: Only Damaged detected (whole-box label)
                # Map confidence to severity — higher confidence = more obvious damage
                conf = dmg.confidence
                if conf >= 0.75:
                    severity, ratio = "Severe", conf
                elif conf >= 0.50:
                    severity, ratio = "Moderate", conf
                else:
                    severity, ratio = "Minor", conf

            draw_damage(frame, dmg, severity, ratio)

        # --- HUD overlay ---
        draw_hud(frame, frame_idx, len(parcels), len(damages))

        # --- Display / write ---
        if args.show:
            cv2.imshow("Parcel Damage Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("⏹  Stopped by user.")
                break

        if writer is not None:
            writer.write(frame)

        # Progress every 100 frames
        if frame_idx % 100 == 0 or frame_idx == total:
            pct = (frame_idx / total * 100) if total > 0 else 0
            print(f"   [{frame_idx}/{total}]  {pct:.1f}%")

    # ── 5. Cleanup ──────────────────────────────────────────────────
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n✅  Done — processed {frame_idx} frames.")
    if writer is not None:
        print(f"    Output saved to: {Path(args.output).resolve()}\n")


# ── Entry point ──────────────────────────────────────────────────────
def main() -> None:
    cfg  = load_settings()
    args = parse_args(cfg)
    run(args, cfg)


if __name__ == "__main__":
    main()
