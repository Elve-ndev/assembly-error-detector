# Cobot Assembly Error Detection Pipeline

> End-to-end real-time system for assembly error detection in industrial environments, designed to guide collaborative robots through procedure monitoring and anomaly detection.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![SlowFast](https://img.shields.io/badge/SlowFast-MECCANO-red)
![BiGRU](https://img.shields.io/badge/BiGRU-Dual--Head-green)
![Viterbi](https://img.shields.io/badge/Viterbi-Decoder-purple)
![Mahalanobis](https://img.shields.io/badge/Mahalanobis-Anomaly-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

This project addresses a critical challenge in industrial robotics: detecting assembly errors in real time from egocentric video streams. Rather than simply classifying actions, the system evaluates whether each action is correctly performed and positioned within the expected procedure, enabling a cobot to react intelligently at four distinct intervention levels.

The full pipeline integrates state-of-the-art video understanding, sequential modeling, probabilistic decoding, and unsupervised anomaly detection into a unified decision framework.

---

## Pipeline Architecture

```
RGB Frames (egocentric)
        |
        v
SlowFast R50 — MECCANO pre-trained
Feature Extraction (blocks.4, 2304-dim)
        |
        v
Normalization + Outlier Clipping
        |
        v
BiGRU Dual-Head + Temporal Attention
        |-- Head 1 : Action Classification (NON-CRITICAL / CRITICAL)
        |-- Head 2 : Anomaly Score
        |
        v
Viterbi Decoder
Learned transition constraints
        |
        v
Mahalanobis Anomaly Detection
Prototype Bank from GRU hidden states (PCA-64)
        |
        v
Cobot Decision Engine
NORMAL / MONITOR / PAUSE / STOP
```

---

## Key Technical Contributions

### Adaptive Feature Extraction

SlowFast R50 pre-trained on the MECCANO industrial egocentric dataset is used as a feature backbone. Hooks are placed at blocks.4 (slow + fast pathways), producing 2304-dim temporal feature vectors. An adaptive stride mapping was engineered to handle heterogeneous extraction conditions across recording batches, achieving complete label-feature alignment across all training sequences.

### Semantic Action Grouping

72 fine-grained action classes were systematically analyzed and regrouped into 2 semantically meaningful categories based on industrial risk and visual discriminability, validated through intra/inter-class cosine similarity analysis (ratio 2.75).

| Class | Actions | Industrial Role |
|-------|---------|----------------|
| NON-CRITICAL | take, put, pull, check, browse | Preparation and verification |
| CRITICAL | fit, plug, tighten, loosen, align | Precision assembly — high error risk |

### BiGRU Dual-Head with Temporal Attention and Residual Blocks

A bidirectional GRU with temporal attention mechanism and residual connections processes variable-length action sequences. The dual-head architecture simultaneously outputs frame-level action predictions and a continuous anomaly score.

```
Input projection : Linear(2304, 512) + LayerNorm + GELU + ResidualBlock
BiGRU            : hidden=512, layers=2, bidirectional → 1024-dim
Temporal Attention: context-aware frame weighting
Action Head      : ResidualBlock → Linear(512, 2)
Anomaly Head     : Linear(1024, 1) + Sigmoid
```

### PSR Error Frame Masking

PSR labels with errors (163 annotated errors across 47 recordings) were used to mask erroneous frames during training, ensuring the BiGRU learns exclusively from correctly executed procedures. This produces cleaner Mahalanobis prototypes and improves anomaly detection.

### Viterbi Temporal Decoding

A Viterbi decoder with a transition matrix learned from training sequences enforces procedural consistency. The highly stable transition matrix (95% self-transition probability) effectively flags sequence deviations as anomalies.

### Mahalanobis Prototype Bank

Class prototypes are built from GRU hidden state representations (1024-dim) of normal training sequences, reduced via PCA-64 (93.3% variance explained). Per-frame Mahalanobis distances provide calibrated anomaly scores without requiring error labels at training time.

---

## Results

### Action Recognition (val set, 2 semantic classes)

| Method | F1 Macro | F1 Weighted |
|--------|----------|-------------|
| LinearSVC frame-level | 0.096 | 0.258 |
| BiGRU 72 classes | 0.097 | — |
| BiGRU 10 classes | 0.291 | — |
| BiGRU 3 classes | 0.463 | — |
| BiGRU 2 classes (final) | 0.663 | 0.701 |

### Anomaly Detection (val set, PSR error labels)

| Method | AUC-ROC |
|--------|---------|
| Isolation Forest PCA64 | 0.658 |
| Mahalanobis PCA64 | 0.668 |
| Mahalanobis per operator | 0.673 |
| Viterbi log-likelihood | 0.751 |
| Combined Mahalanobis + Viterbi | 0.768 |

### Error Type Analysis

| Error Type | Frames | AUC-ROC |
|------------|--------|---------|
| Incorrectly installed | 124 | 0.604 |
| Remove (correction) | 705 | 0.676 |

### Feature Discriminability

| Metric | Value |
|--------|-------|
| Intra-class cosine similarity | 0.031 |
| Inter-class cosine similarity | 0.011 |
| Ratio intra/inter | 2.75 |
| PCA-64 variance explained | 93.3% |

---

## Cobot Decision Engine

| Decision | Trigger | Cobot Response |
|----------|---------|----------------|
| NORMAL | Low anomaly score, sequence consistent | Continue procedure |
| MONITOR | Moderate anomaly or BiGRU/Viterbi disagreement | Increase monitoring |
| PAUSE | Action outside expected sequence | Stop, request confirmation |
| STOP | Critical anomaly score | Full stop, supervisor alert |

---

## Dataset

**IndustReal** — Schoonbeek et al., WACV 2024

| Property | Value |
|----------|-------|
| Total recordings | 84 egocentric videos |
| Operators | 27 participants |
| Train split | 36 recordings + 32 test recordings |
| Val split | 16 recordings (5 operators, never seen) |
| Original action classes | 72 → regrouped to 2 |
| Annotated errors | 163 across 47 recordings |

Reference: [https://github.com/TimSchoonbeek/IndustReal](https://github.com/TimSchoonbeek/IndustReal)

---



---

## Requirements

```
Python >= 3.12
PyTorch >= 2.0
torchvision
pytorchvideo
fvcore
scikit-learn
numpy
pandas
tqdm
scipy
matplotlib
seaborn
```

---

## Installation

```bash
git clone https://github.com/Elve-ndev/cobot-assembly-detection
cd cobot-assembly-detection
pip install -r requirements.txt
```

---

## Usage

### Feature Extraction

```python
BATCH      = 'train_p1'
SF_WEIGHTS = 'checkpoints/meccano_slowfast_mapped_clean.pth'
FEAT_DIR   = 'data/features/'
```

### Training

```python
TRAIN_FEAT_DIR = 'data/features/train/'
VAL_FEAT_DIR   = 'data/features/val/'
PREP_DIR       = 'data/preprocessing/'
```

### Inference

```python
from pipeline import CobotDecisionEngine

engine = CobotDecisionEngine(
    bigru_checkpoint = 'checkpoints/bigru_2classes_best.pth',
    scaler_path      = 'checkpoints/scaler.pkl',
    stride_map_path  = 'data/stride_map_train.pkl'
)

decision, message = engine.predict(feature_vector, timestamp)
# Returns: ('NORMAL' | 'MONITOR' | 'PAUSE' | 'STOP', explanation_string)
```

---

## References

Schoonbeek, T.J., Houben, T., Onvlee, H., de With, P.H.N., van der Sommen, F. (2024).
IndustReal: A Dataset for Procedure Step Recognition Handling Execution Errors in Egocentric Videos in an Industrial-Like Setting. WACV 2024.

Feichtenhofer, C., Fan, H., Malik, J., He, K. (2019).
SlowFast Networks for Video Understanding. ICCV 2019.
