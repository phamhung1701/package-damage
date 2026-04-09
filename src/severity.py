"""
severity.py — Damage severity classification logic.

Calculates the area ratio between a detected Damage region and its
parent Parcel, then maps that ratio to a human-readable severity label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


# ── Severity thresholds (defaults — overridden by settings.yaml) ─────
MINOR_MAX: float = 0.10
MODERATE_MAX: float = 0.25


@dataclass
class Detection:
    """Lightweight container for a single detected object."""
    class_id: int
    track_id: Optional[int]
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    confidence: float

    @property
    def area(self) -> float:
        """Bounding-box area in pixels²."""
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


def _iou_overlap(box_a: Tuple[int, ...], box_b: Tuple[int, ...]) -> float:
    """Return the fraction of *box_a* that overlaps with *box_b*."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    return inter / area_a


def find_parent_parcel(
    damage: Detection,
    parcels: List[Detection],
    overlap_threshold: float = 0.5,
) -> Optional[Detection]:
    """
    Return the parcel whose bounding box contains the largest overlap
    with the given damage detection (must exceed *overlap_threshold*).
    """
    best: Optional[Detection] = None
    best_overlap: float = 0.0

    for parcel in parcels:
        overlap = _iou_overlap(damage.bbox, parcel.bbox)
        if overlap > best_overlap:
            best_overlap = overlap
            best = parcel

    return best if best_overlap >= overlap_threshold else None


def classify_severity(
    damage_area: float,
    parcel_area: float,
    minor_max: float = MINOR_MAX,
    moderate_max: float = MODERATE_MAX,
) -> Tuple[str, float]:
    """
    Classify damage severity based on the area ratio.

    Returns:
        (label, ratio)  where label is "Minor", "Moderate", or "Severe".
    """
    if parcel_area <= 0:
        return ("Unknown", 0.0)

    ratio = damage_area / parcel_area

    if ratio <= minor_max:
        label = "Minor"
    elif ratio <= moderate_max:
        label = "Moderate"
    else:
        label = "Severe"

    return (label, round(ratio, 4))
