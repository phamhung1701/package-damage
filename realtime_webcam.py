"""
realtime_webcam.py — Zero-lag real-time damage detection via webcam.

Uses a background thread to continuously grab the latest frame from the
webcam, preventing OpenCV's internal buffer from queuing stale frames
while the GPU processes inference.

Usage:
    python realtime_webcam.py
    python realtime_webcam.py --weights runs/detect/train_merged/weights/best.pt
    python realtime_webcam.py --source 1          # use second camera
    python realtime_webcam.py --conf 0.40

Controls:
    q  — quit
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════════════
#  Threaded Video Capture
# ═══════════════════════════════════════════════════════════════════════

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

        # Minimise the internal OpenCV buffer to 1 frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._thread: threading.Thread | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> "VideoCaptureThread":
        """Spin up the background capture thread."""
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Signal the thread to stop, wait for it, then release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.cap.release()

    # ── internals ────────────────────────────────────────────────────

    def _reader(self) -> None:
        """Background loop: grab → store latest frame (drop old ones)."""
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        """Return the most recent frame (or None if nothing captured yet)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time Webcam Damage Detection (YOLOv8 + ByteTrack)"
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
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  Main Loop
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── 1. Validate weights ─────────────────────────────────────────
    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights.resolve()}")
        print("        Train a model first with:  python train.py")
        return

    # ── 2. Load model — auto-detect GPU vs CPU ────────────────────────
    import torch
    import numpy as np

    use_gpu = torch.cuda.is_available()
    device  = "0" if use_gpu else "cpu"
    use_fp16 = use_gpu  # FP16 only works with CUDA

    print(f"[INFO] Loading model: {weights}")
    print(f"[INFO] Device: {'CUDA GPU (FP16)' if use_gpu else 'CPU (FP32)'}")
    model = YOLO(str(weights))

    # Warm-up: run a dummy inference to initialise kernels
    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    model.predict(dummy, device=device, half=use_fp16, verbose=False)
    print("[INFO] Model loaded & warmed up")

    # ── 3. Start threaded camera ────────────────────────────────────
    print(f"[INFO] Opening camera {args.source} …")
    stream = VideoCaptureThread(source=args.source).start()
    time.sleep(1.0)  # let the camera auto-expose

    print("[INFO] Running — press 'q' to quit\n")

    fps_counter = 0
    fps_time = time.perf_counter()
    display_fps = 0.0

    # ── 4. Inference loop ───────────────────────────────────────────
    try:
        while True:
            frame = stream.read()
            if frame is None:
                continue

            # --- YOLOv8 tracking (GPU FP16 if available, else CPU) ---
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                half=use_fp16,
                verbose=False,
            )

            # --- Render detections onto the frame ---
            annotated = results[0].plot()

            # --- FPS counter ---
            fps_counter += 1
            elapsed = time.perf_counter() - fps_time
            if elapsed >= 1.0:
                display_fps = fps_counter / elapsed
                fps_counter = 0
                fps_time = time.perf_counter()

            cv2.putText(
                annotated, f"FPS: {display_fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2, cv2.LINE_AA,
            )

            # --- Display ---
            cv2.imshow("Damage Detection — Live", annotated)

            # 'q' to quit  (waitKey(1) = ~1ms delay = minimal)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    # ── 5. Graceful cleanup ─────────────────────────────────────────
    print("[INFO] Shutting down …")
    stream.stop()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
