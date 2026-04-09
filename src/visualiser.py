"""
visualiser.py — Drawing utilities for bounding boxes, labels, and overlays.

Keeps all OpenCV rendering logic in one place so that ``main.py`` stays
focused on the processing pipeline.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src.severity import Detection


# ── Colour palette (BGR) ────────────────────────────────────────────
DEFAULT_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "parcel":   (0, 255, 0),     # green
    "damage":   (0, 0, 255),     # red
    "text":     (255, 255, 255), # white
    "bg":       (0, 0, 0),       # black  (label background)
}

SEVERITY_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "Minor":    (0, 200, 0),     # green-ish
    "Moderate": (0, 180, 255),   # orange
    "Severe":   (0, 0, 255),     # red
    "Unknown":  (128, 128, 128), # grey
}


def draw_detection(
    frame: np.ndarray,
    det: Detection,
    label: str,
    colour: Tuple[int, int, int],
    thickness: int = 2,
    font_scale: float = 0.6,
) -> None:
    """Draw a bounding box with a filled label background on *frame* (in-place)."""
    x1, y1, x2, y2 = det.bbox

    # ── Bounding box ────────────────────────────────────────────────
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

    # ── Label background ────────────────────────────────────────────
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1,
    )
    cv2.rectangle(
        frame,
        (x1, y1 - th - baseline - 6),
        (x1 + tw + 4, y1),
        colour,
        cv2.FILLED,
    )

    # ── Label text ──────────────────────────────────────────────────
    cv2.putText(
        frame,
        label,
        (x1 + 2, y1 - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        DEFAULT_COLOURS["text"],
        1,
        cv2.LINE_AA,
    )


def draw_parcel(
    frame: np.ndarray,
    det: Detection,
    *,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> None:
    """Draw a parcel detection with its track ID."""
    tid = det.track_id if det.track_id is not None else "?"
    label = f"Parcel #{tid}  {det.confidence:.0%}"
    draw_detection(frame, det, label, DEFAULT_COLOURS["parcel"], thickness, font_scale)


def draw_damage(
    frame: np.ndarray,
    det: Detection,
    severity: str,
    ratio: float,
    *,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> None:
    """Draw a damage detection with severity and ratio annotation."""
    tid = det.track_id if det.track_id is not None else "?"
    colour = SEVERITY_COLOURS.get(severity, SEVERITY_COLOURS["Unknown"])
    label = f"DMG #{tid} | {severity} ({ratio:.1%})"
    draw_detection(frame, det, label, colour, thickness, font_scale)


def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    parcel_count: int,
    damage_count: int,
) -> None:
    """Render a translucent heads-up display in the top-left corner."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (300, 90), (30, 30, 30), cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    lines = [
        f"Frame: {frame_idx}",
        f"Parcels: {parcel_count}   Damages: {damage_count}",
    ]
    y = 32
    for line in lines:
        cv2.putText(
            frame, line, (16, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            DEFAULT_COLOURS["text"], 1, cv2.LINE_AA,
        )
        y += 26
