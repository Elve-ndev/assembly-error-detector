# Architecture

## Pipeline overview

```
RGB Frames (egocentric, 30 FPS)
        │
        ▼
SlowFast R50 — MECCANO pre-trained
Feature Extraction (blocks.4 hook)
  slow pathway : [B, 2048,  8, 7, 7]
  fast pathway : [B,  256, 32, 7, 7]
  → avg spatial + concat → 2304-dim per frame
        │
        ▼
Adaptive Stride Mapping
(heterogeneous batch alignment)
        │
        ▼
StandardScaler + clip[-5, +5]
        │
        ▼
BiGRU Dual-Head
+ Temporal Attention
+ Residual Blocks
        │
        ├── Action Head  : NON-CRITICAL / CRITICAL
        └── Anomaly Head : continuous score [0, 1]
                │
                ▼
┌──────────────────────────────────────────┐
│  Semi-supervised LR   (weight 0.5)       │
│  Prototype Ratio      (weight 0.3)       │
│  Mahalanobis RGB      (weight 0.1)       │
│  Mahalanobis Gaze     (weight 0.1)       │
│  Temporal Smoothing   W = 20 frames      │
└──────────────────────────────────────────┘
        │
        ▼
Cobot Decision Engine
NORMAL / WATCH / MONITOR / PAUSE / STOP
        │
        ▼
Rerun.io Live Visualization
```

---

## Components

### 1. SlowFast R50 feature extractor

SlowFast processes egocentric RGB frames at two temporal resolutions simultaneously:

- **Slow pathway** — captures spatial semantics at low frame rate (8 frames)
- **Fast pathway** — captures motion dynamics at high frame rate (32 frames)

Features are extracted at `blocks.4` via a forward hook, spatially averaged, and concatenated to produce a **2304-dim feature vector** per frame.

The backbone uses weights pre-trained on the **MECCANO dataset** (industrial egocentric actions), providing strong domain adaptation compared to Kinetics-400 weights.

### 2. Adaptive Stride Mapping

SlowFast extraction operates at variable strides across recording batches. A per-recording stride map aligns extracted feature indices to ground-truth frame numbers, enabling exact label-feature correspondence across all 84 recordings.

### 3. Semantic Action Grouping

72 fine-grained action classes were analyzed through intra/inter-class cosine similarity (ratio 2.75) and regrouped into 2 operationally meaningful categories:

| Class | Actions | Risk |
|---|---|---|
| NON-CRITICAL | take, put, pull, check, browse | Low |
| CRITICAL | fit, plug, tighten, loosen, align | High — error-prone |

### 4. BiGRU Dual-Head

```
Input projection : Linear(2304, 512) + LayerNorm + GELU + ResidualBlock
BiGRU            : hidden=512, layers=2, bidirectional → 1024-dim
Temporal Attention: context-aware frame weighting
Action Head      : ResidualBlock → Linear(512, 2)
Anomaly Head     : Linear(1024, 1) + Sigmoid
Parameters       : ~10.5M
```

PSR error frame masking (163 annotated errors) ensures BiGRU learns exclusively from correctly executed sequences, producing cleaner prototype representations.

### 5. Semi-supervised Anomaly Detection

A Logistic Regression classifier trained on GRU hidden states (PCA-64, 93.3% variance explained) with 1,696 annotated error frames and 33,920 normal frames (ratio 1:20). Achieves AUC-ROC = 0.824 independently.

### 6. Prototype Ratio Score

```python
ratio = dist_to_normal_prototype / (dist_to_error_prototype + ε)
```

Frames closer to the error prototype than the normal prototype receive higher anomaly scores.

### 7. Multimodal Integration

Gaze (x, y) and hand joint positions (42-dim, 21 joints) from IndustReal annotations are aligned with the stride map and integrated into the Mahalanobis prototype bank.

---

## Cobot Decision Engine

| Decision | Threshold | Cobot Response |
|---|---|---|
| NORMAL | score < 0.189 | Continue procedure |
| WATCH | ≥ 0.189 | Log event, increase observation |
| MONITOR | ≥ 0.221 | Reduce speed, heightened attention |
| PAUSE | ≥ 0.261 | Stop movement, request confirmation |
| STOP | ≥ 0.313 | Full stop, supervisor alert |

---

## Results

### Action recognition

| Backbone | Method | Classes | F1 Macro |
|---|---|---|---|
| EfficientNet-B0 | BiGRU | 72 | 0.136 |
| SlowFast MECCANO | LinearSVC | 72 | 0.096 |
| SlowFast MECCANO | **BiGRU + Attention + PSR masking** | **2** | **0.663** |

### Anomaly detection

| Method | AUC-ROC | TPR@FPR=10% |
|---|---|---|
| Mahalanobis RGB PCA64 | 0.668 | 0.560 |
| + Gaze + Hands | 0.680 | 0.571 |
| + Semi-supervised LR | 0.824 | 0.560 |
| + Prototype Ratio | 0.831 | 0.594 |
| **Final fusion + W=20** | **0.853** | **0.679** |

### Error detection by severity

| Decision | Threshold | TPR | Errors detected |
|---|---|---|---|
| STOP | > 0.313 | 34.2% | 275 / 803 |
| PAUSE | > 0.261 | 18.8% | 151 / 803 |
| MONITOR | > 0.221 | 14.9% | 120 / 803 |
| WATCH | > 0.189 | 7.2% | 58 / 803 |
| **Total** | | **75.2%** | **604 / 803** |
