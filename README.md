# 📦 AI-Powered Logistics Cargo Damage Detection & Severity Assessment

<div align="center">

**An automated computer vision system for real-time detection, tracking, and severity classification of damaged parcels in logistics warehouse environments.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=white)](https://docs.ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-Academic-green)](#license)

</div>

---

## 1. Introduction

### 1.1 Problem Statement

In modern logistics and e-commerce fulfillment centers, millions of parcels are processed daily across conveyor belt systems. **Manual visual inspection** of these parcels for physical damage—such as dents, tears, and crushing—is:

- **Slow and labor-intensive**, creating bottlenecks in high-throughput warehouses.
- **Error-prone**, as human inspectors suffer from fatigue, especially during extended shifts.
- **Inconsistent**, with subjective assessments varying between inspectors and shifts.

These limitations lead to damaged goods reaching customers, increased return rates, and significant financial losses for logistics operators.

### 1.2 Proposed Solution

This project presents an **end-to-end automated Computer Vision pipeline** that:

1. **Detects** parcels and visible damage regions in real-time video streams using a state-of-the-art YOLOv8 object detection model.
2. **Tracks** individual parcels across consecutive video frames using the ByteTrack algorithm, ensuring each parcel receives a persistent identity as it moves along the conveyor belt.
3. **Classifies severity** of detected damage by computing the geometric area ratio between the damage region and its parent parcel, categorizing each instance as **Minor**, **Moderate**, or **Severe**.

---

## 2. Methodology & System Architecture

### 2.1 System Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Video Input │────▶│  YOLOv8      │────▶│  ByteTrack   │────▶│  Severity    │
│  (Camera /   │     │  Detection   │     │  Multi-Object│     │  Assessment  │
│   Video File)│     │              │     │  Tracking    │     │  (Area Ratio)│
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                                                                      ▼
                                                               ┌──────────────┐
                                                               │  Annotated   │
                                                               │  Video Output│
                                                               │  + Reports   │
                                                               └──────────────┘
```

### 2.2 Detection Engine — YOLOv8

The detection backbone is **YOLOv8 Nano** (`yolov8n`), a state-of-the-art single-stage object detector from [Ultralytics](https://docs.ultralytics.com). YOLOv8 was selected for its:

- **Real-time inference speed** — suitable for live video processing.
- **High accuracy-to-speed ratio** — Nano variant balances performance with computational efficiency.
- **Native tracking integration** — eliminates the need for custom tracking implementations.

The model is fine-tuned to detect two target classes:

| Class ID | Class Name | Description |
|----------|-----------|-------------|
| `0` | **Damaged** | Visible damage regions (dents, tears, crushing) |
| `1` | **package** | Intact or whole parcel bounding boxes |

### 2.3 Multi-Object Tracking — ByteTrack

To maintain consistent parcel identities across video frames, the system integrates **ByteTrack** directly through the Ultralytics API:

```python
results = model.track(frame, persist=True, tracker="bytetrack.yaml")
```

ByteTrack associates detections across frames using a two-stage matching strategy, assigning each tracked parcel a unique ID. This enables:

- Frame-to-frame consistency for damage reporting.
- Per-parcel damage history across the video timeline.
- Accurate counting of affected parcels.

### 2.4 Severity Classification Logic

When a `Damaged` region is detected, the system identifies its **parent parcel** by computing bounding-box overlap. The severity is then determined by the **area ratio**:

```
                    Area of Damage Bounding Box
Severity Ratio = ─────────────────────────────────
                    Area of Parent Parcel Bounding Box
```

The ratio is mapped to severity levels using configurable thresholds:

| Ratio Range | Severity | Visual Indicator |
|-------------|----------|-----------------|
| `0% – 10%` | 🟢 **Minor** | Green bounding box |
| `10% – 25%` | 🟡 **Moderate** | Orange bounding box |
| `> 25%` | 🔴 **Severe** | Red bounding box |

> Thresholds are fully configurable in `config/settings.yaml`.

---

## 3. Data Preparation

### 3.1 Dataset Sources

The training dataset was curated by combining **two Roboflow-hosted datasets** containing real-world images of damaged and intact parcels from logistics environments:

| Dataset | Images | Source |
|---------|--------|--------|
| Dataset A | 487 | Roboflow (Damaged / package) |
| Dataset B | 384 | Roboflow (damaged package / undamaged) |
| **Merged Total** | **871** | Combined & deduplicated |

### 3.2 Class Mapping & Harmonization

Both datasets were harmonized to a unified 2-class schema:

```
Original Labels              →  Unified Labels
─────────────────────────────────────────────────
"Damaged"                    →  Class 0: Damaged
"damaged package"            →  Class 0: Damaged
"damaged-packages"           →  Class 0: Damaged
"package"                    →  Class 1: package
"undamaged"                  →  Class 1: package
```

### 3.3 Train / Validation Split

The merged dataset was split using an **80/20 stratified random split** (seed=42 for reproducibility):

| Split | Images | Proportion |
|-------|--------|------------|
| **Train** | 697 | 80% |
| **Valid** | 174 | 20% |
| **Total** | 871 | 100% |

The split is performed automatically via `split_data.py` or `merge_datasets.py`.

---

## 4. Training Configuration

### 4.1 Environment

| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA GTX 1660 |
| **Framework** | PyTorch with Ultralytics YOLOv8 |
| **Base Model** | `yolov8n.pt` (pre-trained on COCO) |

### 4.2 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 90 | Model converged; validation loss plateaued |
| **Image Size** | 640×640 | Standard YOLOv8 input resolution |
| **Batch Size** | 16 | Optimized for GPU memory |
| **Optimizer** | AdamW (auto) | Adaptive learning rate with weight decay |
| **Learning Rate** | 0.001667 (auto) | Ultralytics auto-tuned |
| **Early Stopping** | patience=20 | Prevents overfitting |
| **Mosaic Augmentation** | 1.0 | Full mosaic for training diversity |
| **Horizontal Flip** | 0.5 | Standard augmentation |
| **Rotation** | ±5° | Mild rotation for robustness |

### 4.3 Training Results

| Metric | Score |
|--------|-------|
| **mAP@50** | **82.94%** |
| **mAP@50-95** | **63.95%** |
| **Precision** | **87.93%** |
| **Recall** | **79.17%** |

> Training curves, confusion matrices, and per-class metrics are available in `runs/detect/train_merged/`.

---

## 5. Project Structure

```
package-damage/
│
├── config/
│   ├── data.yaml                  # Dataset class definitions & paths
│   └── settings.yaml              # Runtime configuration (thresholds, paths, visuals)
│
├── datasets/
│   ├── train/
│   │   ├── images/                # Training images (YOLO format)
│   │   └── labels/                # Training annotations (.txt)
│   └── valid/
│       ├── images/                # Validation images
│       └── labels/                # Validation annotations
│
├── runs/
│   └── detect/
│       └── train_merged/
│           ├── weights/
│           │   ├── best.pt        # ⭐ Best model weights
│           │   └── last.pt        # Final epoch weights
│           ├── results.png        # Training & validation curves
│           ├── confusion_matrix.png
│           └── results.csv        # Per-epoch metrics
│
├── src/
│   ├── __init__.py
│   ├── severity.py                # Severity classification logic (area ratio)
│   └── visualiser.py              # Bounding box & HUD rendering utilities
│
├── main.py                        # 🎬 Video inference & tracking pipeline
├── train.py                       # 🚀 Model training script
├── split_data.py                  # ✂️  Dataset train/valid splitter
├── merge_datasets.py              # 🔗 Multi-dataset merger
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

---

## 6. Installation & Usage

### 6.1 Prerequisites

- Python 3.10 or later
- NVIDIA GPU with CUDA support (recommended)
- pip package manager

### 6.2 Step 1 — Environment Setup

```bash
# Clone the repository
git clone https://github.com/phamhung1701/package-damage.git
cd package-damage

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 6.3 Step 2 — Run Inference on Video

The trained model (`best.pt`) is ready to process warehouse video footage:

```bash
# Basic usage — processes video and saves annotated output
python main.py --input data/videos/warehouse_feed.mp4 --show

# Custom configuration
python main.py \
    --input data/videos/conveyor.mp4 \
    --output output/conveyor_analyzed.mp4 \
    --weights runs/detect/train_merged/weights/best.pt \
    --conf 0.35 \
    --show
```

**Available CLI Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `data/videos/sample.mp4` | Path to input video file |
| `--output` | `output/result.mp4` | Path for annotated output video |
| `--weights` | `runs/detect/train_merged/weights/best.pt` | Trained model weights |
| `--conf` | `0.35` | Minimum detection confidence |
| `--iou` | `0.45` | NMS IoU threshold |
| `--tracker` | `bytetrack.yaml` | Tracking algorithm config |
| `--show` | `false` | Display live preview window |
| `--no-save` | `false` | Skip saving output to disk |

> Press **`q`** during live preview to stop processing early.

### 6.4 Step 3 — Run Inference on Images

To quickly test the model on a **single image or folder of images** without video:

```bash
# Single image
python -c "from ultralytics import YOLO; model = YOLO('runs/detect/train_merged/weights/best.pt'); model.predict('path/to/image.jpg', save=True, conf=0.35)"

# Folder of images
python -c "from ultralytics import YOLO; model = YOLO('runs/detect/train_merged/weights/best.pt'); model.predict('path/to/images/', save=True, conf=0.35)"

# Test on validation set
python -c "from ultralytics import YOLO; model = YOLO('runs/detect/train_merged/weights/best.pt'); model.predict('datasets/valid/images', save=True, conf=0.35)"
```

Annotated results are saved to `runs/detect/predict/`. Each output image will show bounding boxes with class labels (`Damaged` / `package`) and confidence scores.

### 6.5 Step 4 — View Results

The output video contains:

- **Green bounding boxes** around detected parcels with track IDs.
- **Color-coded damage boxes** with severity labels and confidence scores.
- **Heads-up display (HUD)** showing frame count, parcel count, and damage count in real-time.

After processing completes, a **Damage Assessment Report** is printed to the terminal:

```
======================================================================
  DAMAGE ASSESSMENT REPORT
======================================================================
  Frames analyzed       : 239
  Frames with damage    : 300
  Frames intact         : 7
  Peak confidence       : 90.0%
----------------------------------------------------------------------
  VERDICT : DAMAGED
  Severity: Severe
  Action  : FLAG FOR INSPECTION
======================================================================
```

A CSV summary is also saved to `output/report.csv`.

### 6.6 (Optional) Retrain the Model

To retrain on a new or updated dataset:

```bash
# Prepare dataset in YOLO format under datasets/
python split_data.py                  # Split 80/20

# Train
python train.py --epochs 90 --batch 16 --imgsz 640
```

---

## 7. Configuration Reference

All runtime parameters are centralized in **`config/settings.yaml`**:

```yaml
model:
  weights: "runs/detect/train_merged/weights/best.pt"
  confidence: 0.35
  iou_threshold: 0.45

severity:
  minor_max: 0.10      # 0%–10%  → Minor
  moderate_max: 0.25   # 10%–25% → Moderate
                       # >25%    → Severe
```

CLI flags always override YAML settings at runtime.

---

## 8. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Detection** | YOLOv8 Nano (Ultralytics) | Real-time object detection |
| **Tracking** | ByteTrack | Multi-object tracking with persistent IDs |
| **Video I/O** | OpenCV (`cv2`) | Frame capture, rendering, and encoding |
| **Severity Engine** | Custom (Python) | Area-ratio-based damage classification |
| **Training Framework** | PyTorch | Deep learning backend |
| **Configuration** | PyYAML | Human-readable settings management |

---

## 9. License

This project was developed for academic and educational purposes as part of a university capstone project. All rights reserved.

---

<div align="center">

**Built with [Ultralytics YOLOv8](https://docs.ultralytics.com) • [OpenCV](https://opencv.org) • [ByteTrack](https://github.com/ifzhang/ByteTrack)**

</div>
