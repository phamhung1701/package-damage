"""
logistics_liability_tracker.py — Zone-based liability tracking with SQLite logging.

Runs YOLOv8 + ByteTrack on a live webcam feed.  When a NEW track ID appears,
the system **simulates an external barcode-scanner event** by generating a
mock Package ID (e.g. "PKG-VN-839172").  The mock ID is pinned to the tracked
bounding box for the duration of its visibility.

While a package is on-screen the tracker records its *worst observed* damage
severity (Normal → Minor → Severe).  Once a package leaves the frame (track
lost for > LOST_FRAMES_THRESHOLD consecutive frames), its final status is
written to an SQLite database so that downstream analytics can determine
*which zone* caused the damage.

Hardware note
-------------
Uses ``half=True`` (FP16) for the GTX 1660 — acceptable precision trade-off
that roughly doubles throughput on Turing-architecture GPUs.

Usage
-----
    python logistics_liability_tracker.py --zone inbound
    python logistics_liability_tracker.py --zone outbound --source 1
    python logistics_liability_tracker.py --zone storage  --conf 0.40
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import string
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from src.severity import Detection, classify_severity, find_parent_parcel


# ═════════════════════════════════════════════════════════════════════════
#  Constants
# ═════════════════════════════════════════════════════════════════════════

VALID_ZONES = ("inbound", "storage", "outbound")

# Number of consecutive frames a track must be absent before we consider
# the package as having "left the frame" and flush its record to the DB.
LOST_FRAMES_THRESHOLD = 30

# Severity ranking — higher = worse.  "Normal" is the default for an
# undamaged package; it can only escalate upward.
SEVERITY_RANK = {"Normal": 0, "Minor": 1, "Moderate": 2, "Severe": 3}
RANK_TO_LABEL = {v: k for k, v in SEVERITY_RANK.items()}

# Default severity thresholds (mirrored from settings.yaml / severity.py)
MINOR_MAX = 0.10
MODERATE_MAX = 0.25

# Class IDs — must match config/data.yaml
DAMAGE_CLS = 0   # "Damaged"
PARCEL_CLS = 1   # "package"

# Database path
DB_PATH = "liability_log.db"


# ═════════════════════════════════════════════════════════════════════════
#  Mock External Barcode Scanner
# ═════════════════════════════════════════════════════════════════════════
#
#  DESIGN DECISION:
#  The physical camera does NOT have barcode-reading capability.  In a
#  production environment an external hardware scanner (e.g. Honeywell /
#  Zebra industrial scanner mounted beside the camera) would fire an
#  event containing the scanned Package ID via serial / TCP.
#
#  For development & demo purposes we *simulate* that event here:  every
#  time ByteTrack assigns a **new** track ID, we generate a randomised
#  Package ID and treat it as if the external scanner just reported it.
#  This keeps the rest of the pipeline (DB logging, HUD display, etc.)
#  identical to what the production version would look like.
# ═════════════════════════════════════════════════════════════════════════

def generate_mock_package_id() -> str:
    """
    Simulate receiving a Package ID from an external barcode scanner.

    Format: PKG-VN-XXXXXX  (6 random digits)
    In production, this would be replaced by a callback/listener on the
    hardware scanner's serial or TCP interface.
    """
    digits = "".join(random.choices(string.digits, k=6))
    return f"PKG-VN-{digits}"


# ═════════════════════════════════════════════════════════════════════════
#  Per-Package Tracking Record
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class PackageRecord:
    """In-memory record for a package currently being tracked on-screen."""
    package_id: str                     # Mock barcode (pinned to this track)
    track_id: int                       # ByteTrack integer ID
    zone: str                           # CLI --zone value
    first_seen: str = ""                # ISO-8601 timestamp
    worst_severity: str = "Normal"      # escalates: Normal → Minor → Moderate → Severe
    worst_rank: int = 0                 # numeric rank of worst_severity
    frames_seen: int = 0               # total frames this track appeared in
    frames_since_last_seen: int = 0    # consecutive frames *not* seen (for lost detection)

    def update_severity(self, severity: str) -> None:
        """Escalate to a worse severity if applicable — never downgrades."""
        rank = SEVERITY_RANK.get(severity, 0)
        if rank > self.worst_rank:
            self.worst_rank = rank
            self.worst_severity = severity

    def mark_seen(self) -> None:
        """Called every frame in which this track ID is detected."""
        self.frames_seen += 1
        self.frames_since_last_seen = 0

    def mark_absent(self) -> None:
        """Called every frame in which this track ID is NOT detected."""
        self.frames_since_last_seen += 1

    @property
    def is_lost(self) -> bool:
        return self.frames_since_last_seen >= LOST_FRAMES_THRESHOLD


# ═════════════════════════════════════════════════════════════════════════
#  SQLite Database
# ═════════════════════════════════════════════════════════════════════════

def init_database(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Create (or open) the SQLite liability database and ensure the
    PackageHistory table exists.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS PackageHistory (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            PackageID     TEXT    NOT NULL,
            Zone          TEXT    NOT NULL,
            Timestamp     TEXT    NOT NULL,
            Final_Status  TEXT    NOT NULL
        )
    """)
    conn.commit()
    print(f"[DB] Database ready: {Path(db_path).resolve()}")
    return conn


def log_package(conn: sqlite3.Connection, record: PackageRecord) -> None:
    """
    Write a completed PackageRecord to the database.

    Called when a package's track has been lost for > LOST_FRAMES_THRESHOLD
    frames (i.e. the package has left the camera's field of view).
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO PackageHistory (PackageID, Zone, Timestamp, Final_Status) VALUES (?, ?, ?, ?)",
        (record.package_id, record.zone, timestamp, record.worst_severity),
    )
    conn.commit()
    print(
        f"[DB] Logged: {record.package_id} | Zone={record.zone} | "
        f"Status={record.worst_severity} | Frames={record.frames_seen}"
    )


# ═════════════════════════════════════════════════════════════════════════
#  Threaded Video Capture  (reused pattern from realtime_webcam.py)
# ═════════════════════════════════════════════════════════════════════════

class VideoCaptureThread:
    """
    Continuously grabs frames from a cv2.VideoCapture source in a
    background daemon thread.  Only the *latest* frame is kept, so the
    inference loop never processes stale/queued frames.
    """

    def __init__(self, source: int = 0):
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source {source}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> "VideoCaptureThread":
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.cap.release()

    def _reader(self) -> None:
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


# ═════════════════════════════════════════════════════════════════════════
#  Detection Extraction  (adapted from main.py)
# ═════════════════════════════════════════════════════════════════════════

def extract_detections(result) -> List[Detection]:
    """Convert a single Ultralytics Results object into Detection instances."""
    detections: List[Detection] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    cls  = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    ids  = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(cls)

    for i in range(len(cls)):
        detections.append(Detection(
            class_id=int(cls[i]),
            track_id=int(ids[i]) if ids[i] is not None else None,
            bbox=tuple(int(v) for v in xyxy[i]),
            confidence=float(conf[i]),
        ))
    return detections


# ═════════════════════════════════════════════════════════════════════════
#  Visualisation Helpers
# ═════════════════════════════════════════════════════════════════════════

# Colour palette (BGR)
COL_NORMAL   = (0, 200, 0)      # green
COL_MINOR    = (0, 200, 200)    # yellow-ish
COL_MODERATE = (0, 140, 255)    # orange
COL_SEVERE   = (0, 0, 255)      # red
COL_TEXT     = (255, 255, 255)   # white
COL_HUD_BG  = (30, 30, 30)      # dark overlay

SEVERITY_COLOURS = {
    "Normal":   COL_NORMAL,
    "Minor":    COL_MINOR,
    "Moderate": COL_MODERATE,
    "Severe":   COL_SEVERE,
}


def draw_tracked_package(
    frame: np.ndarray,
    det: Detection,
    record: PackageRecord,
    severity: str,
) -> None:
    """
    Draw a bounding box around a tracked package with its pinned mock
    Package ID and current severity label.
    """
    x1, y1, x2, y2 = det.bbox
    colour = SEVERITY_COLOURS.get(severity, COL_NORMAL)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    # Label: Package ID + severity
    label = f"{record.package_id} | {severity}"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - baseline - 6), (x1 + tw + 4, y1), colour, cv2.FILLED)
    cv2.putText(frame, label, (x1 + 2, y1 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT, 1, cv2.LINE_AA)


def draw_damage_box(
    frame: np.ndarray,
    det: Detection,
    severity: str,
    ratio: float,
) -> None:
    """Draw a damage detection with severity annotation."""
    x1, y1, x2, y2 = det.bbox
    colour = SEVERITY_COLOURS.get(severity, COL_SEVERE)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    label = f"DMG | {severity} ({ratio:.1%})"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y2), (x1 + tw + 4, y2 + th + baseline + 6), colour, cv2.FILLED)
    cv2.putText(frame, label, (x1 + 2, y2 + th + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1, cv2.LINE_AA)


def draw_zone_hud(
    frame: np.ndarray,
    zone: str,
    fps: float,
    active_tracks: int,
    total_logged: int,
) -> None:
    """Render a translucent zone-status HUD in the top-left corner."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (380, 120), COL_HUD_BG, cv2.FILLED)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    lines = [
        f"ZONE: {zone.upper()}",
        f"FPS: {fps:.1f}  |  Active Tracks: {active_tracks}",
        f"Packages Logged to DB: {total_logged}",
        "Press 'q' to quit",
    ]
    y = 30
    for line in lines:
        cv2.putText(frame, line, (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT, 1, cv2.LINE_AA)
        y += 24


# ═════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Logistics Liability Tracker — Zone-based package damage logging"
    )
    p.add_argument(
        "--zone", type=str, required=True, choices=VALID_ZONES,
        help="Inspection zone the camera is monitoring (inbound / storage / outbound)",
    )
    p.add_argument(
        "--weights", type=str,
        default="runs/detect/train_merged/weights/best.pt",
        help="Path to trained YOLOv8 weights",
    )
    p.add_argument(
        "--source", type=int, default=0,
        help="Camera index (0 = default webcam)",
    )
    p.add_argument(
        "--conf", type=float, default=0.35,
        help="Minimum detection confidence threshold",
    )
    p.add_argument(
        "--iou", type=float, default=0.45,
        help="NMS IoU threshold",
    )
    p.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size",
    )
    p.add_argument(
        "--db", type=str, default=DB_PATH,
        help="Path to SQLite database file",
    )
    p.add_argument(
        "--lost-threshold", type=int, default=LOST_FRAMES_THRESHOLD,
        help="Frames a track must be absent before logging to DB",
    )
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════
#  Main Loop
# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    zone = args.zone
    lost_threshold_override = args.lost_threshold

    # ── 1. Validate weights ─────────────────────────────────────────
    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights.resolve()}")
        print("        Train a model first with:  python train.py")
        return

    # ── 2. Load model — GTX 1660 optimisation (FP16) ────────────────
    import torch

    use_gpu = torch.cuda.is_available()
    device  = "0" if use_gpu else "cpu"
    # half=True provides ~2× throughput on GTX 1660 (Turing arch)
    use_fp16 = use_gpu

    print(f"[INFO] Zone        : {zone.upper()}")
    print(f"[INFO] Model       : {weights}")
    print(f"[INFO] Device      : {'CUDA GPU (FP16)' if use_gpu else 'CPU (FP32)'}")
    print(f"[INFO] Lost thresh : {lost_threshold_override} frames")

    model = YOLO(str(weights))

    # Warm-up: run a dummy inference to initialise CUDA kernels
    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    model.predict(dummy, device=device, half=use_fp16, verbose=False)
    print("[INFO] Model loaded & warmed up")

    # ── 3. Initialise SQLite database ───────────────────────────────
    conn = init_database(args.db)
    total_logged = 0

    # ── 4. Start threaded camera ────────────────────────────────────
    print(f"[INFO] Opening camera {args.source} …")
    stream = VideoCaptureThread(source=args.source).start()
    time.sleep(1.0)  # let the camera auto-expose

    print("[INFO] Running — press 'q' to quit\n")

    # ── 5. Tracking state ───────────────────────────────────────────
    #
    #  active_records:  { ByteTrack_ID → PackageRecord }
    #
    #  When ByteTrack assigns a NEW track ID we:
    #    1. Generate a mock Package ID  (simulated external scanner event)
    #    2. Create a PackageRecord pinned to that track
    #
    #  When a track is not seen for > lost_threshold frames:
    #    1. Log the PackageRecord (worst severity) to the SQLite DB
    #    2. Remove it from active_records
    #
    active_records: Dict[int, PackageRecord] = {}

    fps_counter = 0
    fps_time = time.perf_counter()
    display_fps = 0.0

    # ── 6. Inference loop ───────────────────────────────────────────
    try:
        while True:
            frame = stream.read()
            if frame is None:
                continue

            # --- YOLOv8 + ByteTrack (persist=True keeps IDs across frames) ---
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                half=use_fp16,          # GTX 1660 FP16 optimisation
                verbose=False,
            )

            # --- Parse detections ---
            detections = extract_detections(results[0])
            parcels = [d for d in detections if d.class_id == PARCEL_CLS]
            damages = [d for d in detections if d.class_id == DAMAGE_CLS]

            # Set of track IDs observed THIS frame
            seen_track_ids: set[int] = set()

            # ── Process parcels ──────────────────────────────────────
            for p in parcels:
                if p.track_id is None:
                    continue

                seen_track_ids.add(p.track_id)

                # --- NEW track? → simulate external scanner event ---
                if p.track_id not in active_records:
                    mock_id = generate_mock_package_id()
                    record = PackageRecord(
                        package_id=mock_id,
                        track_id=p.track_id,
                        zone=zone,
                        first_seen=datetime.now(timezone.utc).isoformat(),
                    )
                    active_records[p.track_id] = record
                    print(
                        f"[SCANNER] New package detected → "
                        f"Track #{p.track_id} ↔ {mock_id}  (Zone: {zone.upper()})"
                    )

                active_records[p.track_id].mark_seen()

                # Draw the parcel with its pinned Package ID
                draw_tracked_package(frame, p, active_records[p.track_id], "Normal")

            # ── Process damages → find parent parcel → update severity ──
            for dmg in damages:
                parent = find_parent_parcel(dmg, parcels, overlap_threshold=0.3)

                if parent is not None:
                    severity, ratio = classify_severity(
                        dmg.area, parent.area,
                        minor_max=MINOR_MAX,
                        moderate_max=MODERATE_MAX,
                    )
                    # Update the parent parcel's record with the damage severity
                    if parent.track_id is not None and parent.track_id in active_records:
                        active_records[parent.track_id].update_severity(severity)

                        # Re-draw the parcel box with escalated severity colour
                        draw_tracked_package(
                            frame, parent,
                            active_records[parent.track_id],
                            active_records[parent.track_id].worst_severity,
                        )
                else:
                    # No parent parcel found — classify by confidence alone
                    conf = dmg.confidence
                    if conf >= 0.75:
                        severity, ratio = "Severe", conf
                    elif conf >= 0.50:
                        severity, ratio = "Moderate", conf
                    else:
                        severity, ratio = "Minor", conf

                draw_damage_box(frame, dmg, severity, ratio)

            # ── Mark absent tracks & flush lost ones to DB ───────────
            lost_ids: List[int] = []
            for tid, record in active_records.items():
                if tid not in seen_track_ids:
                    record.mark_absent()
                    if record.frames_since_last_seen >= lost_threshold_override:
                        lost_ids.append(tid)

            for tid in lost_ids:
                record = active_records.pop(tid)
                log_package(conn, record)
                total_logged += 1

            # ── HUD overlay ──────────────────────────────────────────
            # FPS counter
            fps_counter += 1
            elapsed = time.perf_counter() - fps_time
            if elapsed >= 1.0:
                display_fps = fps_counter / elapsed
                fps_counter = 0
                fps_time = time.perf_counter()

            draw_zone_hud(frame, zone, display_fps, len(active_records), total_logged)

            # ── Display ──────────────────────────────────────────────
            cv2.imshow(f"Liability Tracker — {zone.upper()}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    # ── 7. Flush remaining active records before exit ────────────────
    #  Any packages still on-screen when the operator stops the tracker
    #  are logged with their current worst severity.
    print(f"\n[INFO] Flushing {len(active_records)} remaining active records …")
    for record in active_records.values():
        log_package(conn, record)
        total_logged += 1
    active_records.clear()

    # ── 8. Cleanup ──────────────────────────────────────────────────
    conn.close()
    stream.stop()
    cv2.destroyAllWindows()

    print(f"\n{'=' * 60}")
    print(f"  SESSION SUMMARY — Zone: {zone.upper()}")
    print(f"{'=' * 60}")
    print(f"  Total packages logged : {total_logged}")
    print(f"  Database              : {Path(args.db).resolve()}")
    print(f"{'=' * 60}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
