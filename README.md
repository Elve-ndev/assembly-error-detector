# Cobot Assembly Error Detection Pipeline

> End-to-end real-time system for assembly error detection in industrial environments, designed to guide collaborative robots (cobots) through egocentric video monitoring and multimodal anomaly detection.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![SlowFast](https://img.shields.io/badge/SlowFast-MECCANO-red)
![BiGRU](https://img.shields.io/badge/BiGRU-Dual--Head-green)
![Mahalanobis](https://img.shields.io/badge/Mahalanobis-Anomaly-yellow)
![Rerun](https://img.shields.io/badge/Rerun-0.32-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

This project addresses a critical challenge in industrial robotics: detecting assembly errors in real time from egocentric video streams. The system goes beyond simple action classification — it evaluates whether each action is correctly executed within the expected procedural context, enabling a cobot to react at four distinct intervention levels.

The pipeline integrates state-of-the-art video understanding (SlowFast R50, MECCANO pre-trained), sequential modeling (BiGRU with temporal attention), and semi-supervised multimodal anomaly detection (Mahalanobis, prototype ratio, gaze, hands) into a unified decision framework visualized live with Rerun.io.

---

## Pipeline Architecture

```
RGB Frames (egocentric, 30 FPS)
        |
        v
SlowFast R50 — MECCANO pre-trained
Feature Extraction (blocks.4 hook)
slow pathway : [B, 2048, 8, 7, 7]
fast pathway : [B,  256, 32, 7, 7]
→ avg spatial + concat → 2304-dim per frame
        |
        v
Adaptive Stride Mapping
(heterogeneous batch alignment)
        |
        v
StandardScaler + clip[-5, +5]
        |
        v
BiGRU Dual-Head
+ Temporal Attention
+ Residual Blocks
        |
        |-- Action Head : NON-CRITICAL / CRITICAL
        |-- Anomaly Head : continuous score [0,1]
        |
        v
┌──────────────────────────────────────┐
│  Semi-supervised LR  (poids 0.5)     │
│  Prototype Ratio     (poids 0.3)     │
│  Mahalanobis RGB     (poids 0.1)     │
│  Mahalanobis Gaze    (poids 0.1)     │
│  Temporal Smoothing  W=20 frames     │
└──────────────────────────────────────┘
        |
        v
Cobot Decision Engine
NORMAL / WATCH / MONITOR / PAUSE / STOP
        |
        v
Rerun.io Live Visualization
```

---

## Key Technical Contributions

### 1. Adaptive Stride Mapping

SlowFast extraction operates at variable strides across recording batches. A per-recording stride map aligns extracted feature indices to ground-truth frame numbers, enabling exact label-feature correspondence across all 84 recordings.

### 2. Semantic Action Grouping

72 fine-grained action classes were analyzed through intra/inter-class cosine similarity (ratio 2.75) and regrouped into 2 operationally meaningful categories:

| Class | Actions | Risk |
|-------|---------|------|
| NON-CRITICAL | take, put, pull, check, browse | Low |
| CRITICAL | fit, plug, tighten, loosen, align | High — error-prone |

### 3. BiGRU Dual-Head with Temporal Attention

```
Input projection : Linear(2304, 512) + LayerNorm + GELU + ResidualBlock
BiGRU            : hidden=512, layers=2, bidirectional → 1024-dim
Temporal Attention: context-aware frame weighting
Action Head      : ResidualBlock → Linear(512, 2)
Anomaly Head     : Linear(1024, 1) + Sigmoid
Parameters       : ~10.5M
```

PSR error frame masking (163 annotated errors) ensures BiGRU learns exclusively from correctly executed sequences, producing cleaner prototype representations.

### 4. Semi-supervised Anomaly Detection

A Logistic Regression classifier trained on GRU hidden states (PCA-64, 93.3% variance) with 1,696 annotated error frames and 33,920 normal frames (ratio 1:20) achieves AUC-ROC = 0.824 independently.

### 5. Prototype Ratio Score

```python
ratio = dist_to_normal_prototype / (dist_to_error_prototype + ε)
```

Frames closer to the error prototype than the normal prototype receive higher anomaly scores. Combined with the semi-supervised classifier and temporal smoothing (W=20), this yields the final detection score.

### 6. Multimodal Integration

Gaze (x,y) and hand joint positions (42-dim, 21 joints) from IndustReal annotations are aligned with the stride map and integrated into the Mahalanobis prototype bank, contributing additional signal for manipulation errors.

---

## Results

### Action Recognition (val set, 16 recordings, 5 unseen operators)

| Backbone | Method | Classes | F1 Macro | F1 Weighted |
|----------|--------|---------|----------|-------------|
| EfficientNet-B0 | BiGRU | 72 | 0.136 | 0.233 |
| SlowFast MECCANO | LinearSVC | 72 | 0.096 | 0.258 |
| SlowFast MECCANO | BiGRU + Attention | 72 | 0.097 | — |
| SlowFast MECCANO | **BiGRU + Attention + PSR masking** | **2** | **0.663** | **0.701** |

> *Important context: This score represents the **best achievable performance* given the extreme constraints of the IndustReal dataset — only 84 short egocentric videos, severe class imbalance, and real-world industrial noise.  
> *Action recognition is not the core focus* of this project. It serves as a reliable feature extractor for the *main contribution: multimodal anomaly/error detection, where the pipeline achieves a strong **AUC-ROC of 0.853*.

> Note: The IndustReal dataset (WACV 2024) includes action recognition annotations and a dedicated AR/ benchmark folder.
 However, the authors' primary contribution focuses on Procedure Step Recognition (PSR) via YOLOv8-based Assembly State Detection (ASD). 
Action recognition baselines are provided but evaluated primarily on Top-1/Top-5 accuracy, without extending toward anomaly detection or real-time cobot decision making.
This work takes a fundamentally different approach: rather than recognizing procedural steps, it detects execution errors in real time using a semi-supervised anomaly 
detection pipeline (AUC-ROC=0.853, 7.5× lift over random baseline)validated on 5 unseen operators. This represents, to the best of our knowledge, the firs application
of deep temporal anomaly detection to IndustReal, combining egocentric video understanding, sparse error annotation learning, and multimodal integration for cobot guidance.

### Anomaly Detection (val set, PSR error labels, 803 annotated error frames)

| Method | AUC-ROC | TPR@FPR=10% |
|--------|---------|-------------|
| Mahalanobis RGB PCA64 | 0.668 | 0.560 |
| Mahalanobis Multimodal (RGB+Gaze+Hands) | 0.680 | 0.571 |
| Semi-supervised LR | 0.824 | 0.560 |
| Prototype Ratio (dist_normal/dist_error) | 0.831 | 0.594 |
| **Final (Semi + Ratio + Gaze + W=20)** | **0.853** | **0.679** |

### Key Metrics

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.853** |
| TPR @ FPR = 5% | 0.560 |
| TPR @ FPR = 10% | 0.679 |
| TPR @ FPR = 20% | 0.760 |

### Cobot Decision Engine — Error Detection by Severity

| Decision | Threshold | TPR | Errors Detected |
|----------|-----------|-----|-----------------|
| STOP | > 0.313 | 34.2% | 275 / 803 |
| PAUSE | > 0.261 | 18.8% | 151 / 803 |
| MONITOR | > 0.221 | 14.9% | 120 / 803 |
| WATCH | > 0.189 | 7.2% | 58 / 803 |
| **Total detected** | | **75.2%** | **604 / 803** |
| Missed | | 24.8% | 199 / 803 |

### Error Type Analysis

| Error Type | Recordings | Frames | Detection Rate |
|------------|-----------|--------|----------------|
| Remove (geste de retrait) | main_* | 705 | **~88%** |
| Incorrectly installed | assy_* | 98 | ~20% |

"Incorrectly installed" errors involve incorrect part orientation — visually indistinguishable in RGB. Depth features would be required to resolve 3D orientation, a planned future extension.

### Feature Discriminability

| Metric | Value |
|--------|-------|
| Intra-class cosine similarity | 0.031 |
| Inter-class cosine similarity | 0.011 |
| Discriminability ratio | **2.75** |
| PCA-64 variance explained | 93.3% |

---

## Cobot Decision Engine

| Decision | Trigger | Cobot Response |
|----------|---------|----------------|
| NORMAL | score < 0.189, sequence consistent | Continue procedure |
| WATCH | score ≥ 0.189 | Log event, increase observation |
| MONITOR | score ≥ 0.221 | Reduce speed, heightened attention |
| PAUSE | score ≥ 0.261 | Stop movement, request confirmation |
| STOP | score ≥ 0.313 | Full stop, supervisor alert |

---

## Live Visualization (Rerun.io)

The pipeline includes a real-time visualization script using Rerun 0.32 that streams:
- Egocentric RGB frames with decision overlay (color-coded band + score bar)
- PSR ground-truth error annotations
- Anomaly score timeline with threshold reference lines
- Per-frame cobot decision (NORMAL / WATCH / MONITOR / PAUSE / STOP)





<p align="center">
  <video src="https://github.com/user-attachments/assets/3f64cb51-ec63-4741-b399-d35b3f75e13b" width="100%" controls autoplay loop muted>
    Your browser does not support the video tag.
  </video>
</p>

On this demonstration,  we can observe the real-time pipeline behavior:
* **Frames #328 - #345 (`[MONITOR]` / Yellow):** The system tracks standard preparation steps
* **Frames #367 - #375 (`[PAUSE]` / Orange):** The `Cobot Decision Engine` triggers a temporary pause which aligns well with the authors annotation as the operator performs a target action: `PSR: Remove rear chassis pin`.
* **Frames #382 - #412 (`[STOP]` / Red):** Anomaly score spikes above the `0.313` threshold. The pipeline flags a critical execution error, simulated here by the red visualization band, which would instantly trigger a hardware full stop on a physical cobot.


---

## Dataset

In the AI sector, access to real-world factory data is heavily restricted by industrial confidentiality. To overcome this, this system was trained on IndustReal (WACV 2024), a state-of-the-art reference dataset. IndustReal uses complex assembly structures to accurately replicate the structural complexity, manual precision, and error types found in real-world industrial assembly lines.

**IndustReal** — Schoonbeek et al., WACV 2024

| Property | Value |
|----------|-------|
| Total recordings | 84 egocentric videos |
| Operators | 27 participants |
| Train + Test split | 68 recordings (22 operators) |
| Validation split | 16 recordings (5 unseen operators) |
| Action classes | 72 original → 2 semantic groups |
| Annotated errors | 163 PSR errors (47 recordings) |
| Modalities | RGB, Depth, Stereo, Gaze, Hands, Pose, Ambient light |
| Modalities used | RGB, Gaze, Hands |

Reference: [https://github.com/TimSchoonbeek/IndustReal](https://github.com/TimSchoonbeek/IndustReal)

---

## Ablation Study

### Feature Backbone

| Backbone | F1 Macro | Notes |
|----------|----------|-------|
| EfficientNet-B0 (ImageNet) | 0.136 | No temporal context |
| SlowFast MECCANO | **0.663** | Industrial egocentric domain ✅ |

### Semantic Grouping

| Classes | F1 Macro |
|---------|----------|
| 72 (original) | 0.097 |
| 10 | 0.291 |
| **2 (final)** | **0.663** |

### Anomaly Detection Components

| Configuration | AUC-ROC |
|--------------|---------|
| Mahalanobis only | 0.668 |
| + Gaze + Hands | 0.680 |
| + Semi-supervised | 0.824 |
| + Prototype Ratio | 0.831 |
| **+ Temporal smoothing W=20** | **0.853** |

---

## Repository Structure

```
project/
├── notebooks/
│   ├── EDA.ipynb
│   ├── 01_slowfast_extraction.ipynb
│   ├── 02_bigru_preprocessing.ipynb
│   ├── 03_bigru_training.ipynb
│   └── 04_anomaly_detection_aucroc.ipynb
├── demo_rerun.py
├── checkpoints/
│   ├── meccano_slowfast_mapped_clean.pth // too heavy check  Checkpoints & Data section
│   ├── bigru_2classes_best.pth // too heavy check  Checkpoints & Data section
│   └── scaler.pkl
├── data/
│   ├── stride_map_train.pkl
│   ├── stride_map_val.pkl
│   ├── rec_to_file_train.pkl
│   └── rec_to_file_val.pkl
├── requirements.txt
└── README.md
```

---

## Requirements

```
Python >= 3.12
PyTorch >= 2.0
torchvision
pytorchvideo
fvcore
scikit-learn
imbalanced-learn
numpy
pandas
scipy
matplotlib
seaborn
rerun-sdk >= 0.32
Pillow
```

---
## Checkpoints & Data

Model weights and preprocessing files are hosted on Kaggle (public datasets):

| File | Kaggle Dataset | Description |
|------|---------------|-------------|
| `bigru_2classes_best.pth` | [hibabou/checkpoint0-66](https://kaggle.com/datasets/hibabou/checkpoint0-66) | BiGRU dual-head trained on 68 recordings, F1=0.663 |
| `meccano_slowfast_mapped_clean.pth` | [hibabou/slowfast-weights](https://www.kaggle.com/datasets/hibabou/poidsmecano) | SlowFast R50 pre-trained on MECCANO, mapped to IndustReal |

## Future Work & Improvements

- *Model optimization*: ONNX export + INT8 quantization for edge deployment on robotic hardware (Jetson).
- *ROS2 integration*: Create a ROS2 node for seamless simulation and real cobot interfacing.
- *Larger-scale validation*: Test on bigger industrial datasets if available.

## installation

```bash
git clone https://github.com/Elve-ndev/cobot-assembly-detection
cd cobot-assembly-detection
pip install -r requirements.txt
```

---

## Usage

### Feature Extraction

```python
BATCH      = "train_p1"
SF_WEIGHTS = "checkpoints/meccano_slowfast_mapped_clean.pth"
FEAT_DIR   = "data/features/"
```

### Training

```python
TRAIN_FEAT_DIR = "data/features/train/"
VAL_FEAT_DIR   = "data/features/val/"
PREP_DIR       = "data/preprocessing/"
```

### Inference

```python
from pipeline import CobotDecisionEngine

engine = CobotDecisionEngine(
    bigru_checkpoint = "checkpoints/bigru_2classes_best.pth",
    scaler_path      = "checkpoints/scaler.pkl",
    stride_map_path  = "data/stride_map_train.pkl"
)
decision, message = engine.predict(feature_vector, timestamp)
# Returns: ('NORMAL'|'WATCH'|'MONITOR'|'PAUSE'|'STOP', explanation)
```

### Live Visualization

```bash
python demo_rerun.py
```

---

## References

Schoonbeek, T.J., Houben, T., Onvlee, H., de With, P.H.N., van der Sommen, F. (2024).
*IndustReal: A Dataset for Procedure Step Recognition Handling Execution Errors in Egocentric Videos in an Industrial-Like Setting.* WACV 2024.

Feichtenhofer, C., Fan, H., Malik, J., He, K. (2019).
*SlowFast Networks for Video Understanding.* ICCV 2019.
