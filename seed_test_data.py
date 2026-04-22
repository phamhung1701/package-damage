"""
seed_test_data.py — Populate liability_log.db with realistic sample data
for dashboard testing / demo purposes.

Usage:
    python seed_test_data.py
    python seed_test_data.py --count 200
"""

import argparse
import random
import sqlite3
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = "liability_log.db"

ZONES = ["inbound", "storage", "outbound"]
STATUSES = ["Normal", "Minor", "Moderate", "Severe"]

# Weighted probabilities per zone — simulates storage having more damage
ZONE_STATUS_WEIGHTS = {
    "inbound":  [0.60, 0.20, 0.12, 0.08],   # 40% damaged
    "storage":  [0.35, 0.25, 0.22, 0.18],   # 65% damaged (worst zone)
    "outbound": [0.70, 0.15, 0.10, 0.05],   # 30% damaged
}


def generate_mock_id() -> str:
    digits = "".join(random.choices(string.digits, k=6))
    return f"PKG-VN-{digits}"


def seed(db_path: str, count: int) -> None:
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

    now = datetime.now(timezone.utc)
    rows = []

    for i in range(count):
        zone = random.choice(ZONES)
        status = random.choices(STATUSES, weights=ZONE_STATUS_WEIGHTS[zone], k=1)[0]
        # Spread timestamps over the last 7 days
        ts = now - timedelta(
            days=random.uniform(0, 7),
            hours=random.uniform(0, 12),
            minutes=random.uniform(0, 60),
        )
        rows.append((generate_mock_id(), zone, ts.isoformat(), status))

    conn.executemany(
        "INSERT INTO PackageHistory (PackageID, Zone, Timestamp, Final_Status) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()

    print(f"[OK] Seeded {count} records into {Path(db_path).resolve()}")

    # Quick summary
    zone_counts = {}
    for _, zone, _, status in rows:
        zone_counts.setdefault(zone, {"total": 0, "damaged": 0})
        zone_counts[zone]["total"] += 1
        if status != "Normal":
            zone_counts[zone]["damaged"] += 1

    print(f"\n{'Zone':<12} {'Total':>6} {'Damaged':>8} {'Rate':>8}")
    print("-" * 36)
    for z in ZONES:
        c = zone_counts.get(z, {"total": 0, "damaged": 0})
        rate = (c["damaged"] / c["total"] * 100) if c["total"] > 0 else 0
        print(f"{z:<12} {c['total']:>6} {c['damaged']:>8} {rate:>7.1f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=150, help="Number of sample records")
    p.add_argument("--db", type=str, default=DB_PATH)
    args = p.parse_args()
    seed(args.db, args.count)
