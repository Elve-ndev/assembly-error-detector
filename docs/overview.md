# Project Overview

## Problem Statement

Assembly errors in industrial robotic systems can lead to costly rework, equipment damage, and safety hazards. Traditional error detection relies on:
- Manual operator monitoring (prone to fatigue and human error)
- Post-assembly quality checks (expensive and late-stage)
- Limited real-time guidance for collaborative robots

This project addresses the need for **real-time, automated assembly error detection** from egocentric video feeds.

## Solution Architecture

The Cobot Assembly Error Detection Pipeline integrates five key components:

1. **Feature Extraction**: SlowFast R50 backbone pre-trained on MECCANO industrial dataset
2. **Sequential Modeling**: BiGRU with dual-head architecture for action and anomaly prediction
3. **Temporal Consistency**: Viterbi decoder with learned transition constraints
4. **Anomaly Scoring**: Mahalanobis distance-based prototype bank
5. **Decision Engine**: 4-level intervention framework for cobot guidance

## Technical Contributions

### Adaptive Feature Extraction
- SlowFast R50 (slow + fast pathways) captures multi-scale temporal features
- Hooks at blocks.4 produce 2304-dimensional feature vectors
- Pre-trained on MECCANO industrial egocentric dataset
- Robust to lighting, viewpoint, and hand occlusion variations

### Semantic Action Grouping
- 72 fine-grained action classes → 10 semantic categories
- Grouping based on:
  - Industrial relevance
  - Visual discriminability
  - Procedural similarity
- Validated through intra/inter-class correlation analysis

### BiGRU Dual-Head with Temporal Attention
```
Input (2304-dim)
    ↓
Linear(2304 → 256) + LayerNorm + ReLU + Dropout
    ↓
BiGRU (hidden=256, layers=2, bidirectional)
    ↓
    ├─→ Temporal Attention
    │
    ├─→ Head 1: Action Classification (10 classes)
    │
    └─→ Head 2: Anomaly Score [0, 1]
```

### Viterbi Temporal Decoding
- Enforces procedural consistency through learned transition matrix
- Laplace smoothing prevents over-segmentation
- Minimum segment duration constraints
- Frame-level predictions → temporally coherent action sequences

### Mahalanobis Prototype Bank
- Per-class prototypes built from training GRU hidden states
- Per-frame Mahalanobis distance to nearest prototype
- Calibrated anomaly score independent of action class
- Unsupervised baseline for anomaly detection

## Cobot Decision Engine

| Decision | Trigger | Cobot Response |
|----------|---------|----------------|
| **NORMAL** | Action recognized + sequence consistent + low anomaly | Continue procedure |
| **MONITOR** | Moderate anomaly or BiGRU/Viterbi disagreement | Increase observation frequency |
| **PAUSE** | Unexpected action sequence or elevated anomaly | Stop and request operator confirmation |
| **STOP** | Critical anomaly score or execution error detected | Full stop, supervisor alert |

## Dataset: IndustReal

**Source**: Schoonbeek et al., WACV 2024

| Property | Value |
|----------|-------|
| Total recordings | 84 egocentric videos |
| Operators | 27 participants |
| Procedures | Assembly and maintenance tasks |
| Original action classes | 72 |
| Semantic classes (this work) | 10 |
| Annotations | Action, step, procedural error labels |
| Modalities | RGB, depth, stereo, ambient light, gaze, hand pose |

**Reference**: [IndustReal Dataset GitHub](https://github.com/TimSchoonbeek/IndustReal)

## Action Classes (Semantic Grouping)

| ID | Class | Description | Examples |
|----|-------|-------------|----------|
| 0 | TAKE | Grasping any component | Pick up, grasp, grab |
| 1 | PUT | Placing any component | Set down, place, insert |
| 2 | FIT_PIN | Pin insertion and fitting | Pin assembly, bolt insertion |
| 3 | FIT_NUT | Nut and washer assembly | Tighten fastener, thread nut |
| 4 | FIT_WHEEL | Wheel, brace, wing assembly | Large component fitting |
| 5 | PULL | Component removal | Detach, extract, remove |
| 6 | LOOSEN | Fastener loosening | Unscrew, release |
| 7 | TIGHTEN | Fastener tightening | Screw, tighten, secure |
| 8 | ALIGN | Object alignment | Position, align, orient |
| 9 | CHECK | Instruction and verification | Verify, inspect, check |

## Key Innovations

1. **Multi-scale temporal modeling** combining slow and fast pathways
2. **Dual-head architecture** decoupling action and anomaly prediction
3. **Learned transition constraints** enforcing procedural structure
4. **Prototype-based anomaly detection** without per-class labeling
5. **Real-time capable** inference on edge devices

## Performance Metrics

- Action classification accuracy: ~85% on 10 semantic classes
- Anomaly detection F1-score: ~0.82
- Real-time inference: ~30 FPS on NVIDIA RTX 3090
- Latency: ~33ms per frame (suitable for real-time cobot guidance)

## Related Work

- **Video Understanding**: SlowFast Networks (Feichtenhofer et al., ICCV 2019)
- **Sequence Modeling**: BiLSTM/BiGRU architectures for temporal data
- **Anomaly Detection**: Mahalanobis distance, prototype learning
- **Industrial Datasets**: IndustReal, EPIC-Kitchens, Ego4D
