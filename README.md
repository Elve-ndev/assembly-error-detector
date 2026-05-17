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
        |-- Head 1 : Action Classification
        |-- Head 2 : Anomaly Score
        |
        v
Viterbi Decoder
Learned transition constraints
        |
        v
Mahalanobis Anomaly Detection
Prototype Bank from GRU hidden states
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

72 fine-grained action classes were analyzed and regrouped into 10 semantically coherent categories based on industrial relevance and visual discriminability, validated through intra/inter-class cosine similarity analysis. The grouping demonstrates a discriminability ratio of 2.75, confirming the quality of the learned representations.

### BiGRU Dual-Head with Temporal Attention

A bidirectional GRU with temporal attention mechanism processes variable-length action sequences. The dual-head architecture simultaneously outputs frame-level action predictions and a continuous anomaly score, enabling both classification and detection in a single forward pass.

```
Input projection : Linear(2304, 256) + LayerNorm + ReLU + Dropout
BiGRU            : hidden=256, layers=2, bidirectional
Temporal Attention: context-aware frame weighting
Action Head      : frame-level classification
Anomaly Head     : continuous anomaly scoring [0, 1]
```

### Viterbi Temporal Decoding

A Viterbi decoder with a transition matrix learned from training sequences enforces procedural consistency. Laplace smoothing and minimum segment duration constraints prevent over-segmentation and improve temporal coherence of predictions.

### Mahalanobis Prototype Bank

Class prototypes are built from GRU hidden state representations of training sequences. Per-frame Mahalanobis distances to the nearest prototype provide a calibrated anomaly score, forming the basis of the cobot decision engine without requiring any error labels at training time.

---

## Cobot Decision Engine

The decision engine combines action predictions, Viterbi consistency, and Mahalanobis anomaly scores into four actionable intervention levels:

| Decision | Trigger | Cobot Response |
|----------|---------|----------------|
| NORMAL | Action recognized, sequence consistent, low anomaly score | Continue procedure |
| MONITOR | Moderate anomaly or BiGRU/Viterbi disagreement | Increase observation frequency |
| PAUSE | Action outside expected sequence or elevated anomaly | Stop and request operator confirmation |
| STOP | Critical anomaly score, potential execution error | Full stop, supervisor alert |

---

## Dataset

**IndustReal** — Schoonbeek et al., WACV 2024

A multimodal egocentric dataset for procedure step recognition in industrial-like settings, containing video, depth, stereo, ambient light, gaze, hand and pose data.

| Property | Value |
|----------|-------|
| Total recordings | 84 egocentric videos |
| Operators | 27 participants |
| Procedures | Assembly and maintenance |
| Original action classes | 72 |
| Semantic classes (this work) | 10 |
| Annotations | Action recognition, procedure step recognition, procedural errors |

Reference: [https://github.com/TimSchoonbeek/IndustReal](https://github.com/TimSchoonbeek/IndustReal)

---

## Action Classes

| ID | Class | Semantic Group |
|----|-------|---------------|
| 0 | TAKE | Grasping any component |
| 1 | PUT | Placing any component |
| 2 | FIT_PIN | Pin insertion and fitting |
| 3 | FIT_NUT | Nut and washer assembly |
| 4 | FIT_WHEEL | Wheel, brace, wing assembly |
| 5 | PULL | Component removal |
| 6 | LOOSEN | Fastener loosening |
| 7 | TIGHTEN | Fastener tightening |
| 8 | ALIGN | Object alignment |
| 9 | CHECK | Instruction and verification |

---

## Repository Structure

```
project/
├── notebooks/
│   ├── 01_slowfast_extraction.ipynb
│   ├── 02_bigru_preprocessing.ipynb
│   ├── 03_bigru_training.ipynb
│   └── 04_viterbi_mahalanobis_eval.ipynb
├── checkpoints/
│   ├── meccano_slowfast_mapped_clean.pth
│   ├── bigru_best_10classes.pth
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
numpy
pandas
tqdm
scipy
matplotlib
```

---

## Installation

```bash
git clone https://github.com/username/cobot-assembly-detection
cd cobot-assembly-detection
pip install -r requirements.txt
```

---

## Usage

### Feature Extraction

```python
BATCH      = 'train_p1'  # train_p1, train_p2, train_p3, train_p4, val_p1, val_p2
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
    bigru_checkpoint = 'checkpoints/bigru_best_10classes.pth',
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
