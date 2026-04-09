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

    # Tracking ledger — aggregates per-ID results across all frames
    # { track_id: { "class": str, "severity": str, "confidence": float,
    #               "frames_seen": int, "first_frame": int, "last_frame": int } }
    track_ledger: Dict[int, Dict[str, Any]] = {}

    # Severity ranking for keeping the worst case per track
    severity_rank = {"Minor": 1, "Moderate": 2, "Severe": 3}

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

        # --- Record intact parcels in ledger ---
        for p in parcels:
            draw_parcel(frame, p)
            if p.track_id is not None:
                if p.track_id not in track_ledger:
                    track_ledger[p.track_id] = {
                        "class": "package", "status": "Intact",
                        "severity": "None", "confidence": p.confidence,
                        "frames_seen": 0, "first_frame": frame_idx, "last_frame": frame_idx,
                    }
                entry = track_ledger[p.track_id]
                entry["frames_seen"] += 1
                entry["last_frame"] = frame_idx
                entry["confidence"] = max(entry["confidence"], p.confidence)

        # --- Severity analysis & draw damages ---
        for dmg in damages:
            parent = find_parent_parcel(dmg, parcels, overlap_threshold=0.3)
            if parent is not None:
                severity, ratio = classify_severity(
                    dmg.area, parent.area,
                    minor_max=minor_max,
                    moderate_max=moderate_max,
                )
            else:
                conf = dmg.confidence
                if conf >= 0.75:
                    severity, ratio = "Severe", conf
                elif conf >= 0.50:
                    severity, ratio = "Moderate", conf
                else:
                    severity, ratio = "Minor", conf

            draw_damage(frame, dmg, severity, ratio)

            # Record damage in ledger
            if dmg.track_id is not None:
                if dmg.track_id not in track_ledger:
                    track_ledger[dmg.track_id] = {
                        "class": "Damaged", "status": "Damaged",
                        "severity": severity, "confidence": dmg.confidence,
                        "frames_seen": 0, "first_frame": frame_idx, "last_frame": frame_idx,
                    }
                entry = track_ledger[dmg.track_id]
                entry["frames_seen"] += 1
                entry["last_frame"] = frame_idx
                entry["confidence"] = max(entry["confidence"], dmg.confidence)
                # Keep the worst severity across frames
                if severity_rank.get(severity, 0) > severity_rank.get(entry["severity"], 0):
                    entry["severity"] = severity
                entry["status"] = "Damaged"

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
        print(f"    Output saved to: {Path(args.output).resolve()}")

    # ── 6. Final Verdict Report ─────────────────────────────────────
    # Count frames where damage / intact parcels were detected
    total_damaged_frames = sum(1 for v in track_ledger.values()
                               if v["status"] == "Damaged")
    total_intact_frames  = sum(1 for v in track_ledger.values()
                               if v["status"] == "Intact")
    total_damage_frame_hits = sum(v["frames_seen"] for v in track_ledger.values()
                                  if v["status"] == "Damaged")
    total_intact_frame_hits = sum(v["frames_seen"] for v in track_ledger.values()
                                  if v["status"] == "Intact")

    # Find peak confidence and worst severity
    peak_conf = 0.0
    worst_severity = "None"
    for v in track_ledger.values():
        if v["status"] == "Damaged":
            peak_conf = max(peak_conf, v["confidence"])
            if severity_rank.get(v["severity"], 0) > severity_rank.get(worst_severity, 0):
                worst_severity = v["severity"]

    # Determine overall verdict via frame-count majority
    is_damaged = total_damage_frame_hits > total_intact_frame_hits

    print("\n" + "=" * 70)
    print("  DAMAGE ASSESSMENT REPORT")
    print("=" * 70)
    print(f"  Frames analyzed       : {frame_idx}")
    print(f"  Frames with damage    : {total_damage_frame_hits}")
    print(f"  Frames intact         : {total_intact_frame_hits}")
    print(f"  Peak confidence       : {peak_conf:.1%}")
    print("-" * 70)

    if is_damaged:
        print(f"  VERDICT : DAMAGED")
        print(f"  Severity: {worst_severity}")
        print(f"  Action  : FLAG FOR INSPECTION")
    else:
        print(f"  VERDICT : NOT DAMAGED")
        print(f"  Action  : CLEAR FOR DELIVERY")

    print("=" * 70)

    # Save CSV report
    report_path = Path(args.output).parent / "report.csv"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"frames_analyzed,{frame_idx}\n")
        f.write(f"frames_with_damage,{total_damage_frame_hits}\n")
        f.write(f"frames_intact,{total_intact_frame_hits}\n")
        f.write(f"peak_confidence,{peak_conf:.4f}\n")
        f.write(f"verdict,{'DAMAGED' if is_damaged else 'NOT DAMAGED'}\n")
        f.write(f"severity,{worst_severity if is_damaged else 'None'}\n")
    print(f"\n  Report saved to: {report_path.resolve()}\n")

    if not track_ledger:
        print("\n  No parcels detected in the video.\n")


# ── Entry point ──────────────────────────────────────────────────────
def main() -> None:
    cfg  = load_settings()
    args = parse_args(cfg)
    run(args, cfg)


if __name__ == "__main__":
    main()
