# Overview

## What this project does

This pipeline addresses a critical challenge in industrial robotics: **detecting assembly errors in real time** from egocentric video streams. The system goes beyond simple action classification — it evaluates whether each action is correctly executed within the expected procedural context, enabling a cobot to react at four distinct intervention levels.

## Core contributions

| Contribution | Description |
|---|---|
| **Adaptive Stride Mapping** | Aligns SlowFast features across 84 heterogeneous recordings |
| **Semantic Action Grouping** | 72 fine-grained classes → 2 operationally meaningful groups (F1: 0.097 → 0.663) |
| **BiGRU Dual-Head** | Simultaneous action recognition + continuous anomaly scoring |
| **Semi-supervised Fusion** | 4-component ensemble reaching AUC-ROC = 0.853 (7.5× random baseline) |
| **Live Rerun Visualization** | Frame-level decision overlay with color-coded severity bands |

## Why IndustReal?

In the AI sector, access to real-world factory data is heavily restricted by industrial confidentiality. IndustReal (WACV 2024) replicates the structural complexity, manual precision, and error types found in real industrial assembly lines — making it the ideal benchmark for this work.

## Key results at a glance

```
Action recognition  →  F1 Macro = 0.663  (2 semantic classes, 5 unseen operators)
Anomaly detection   →  AUC-ROC  = 0.853  (803 annotated error frames)
Error detection     →  75.2%    of all errors caught across 4 intervention levels
```

## What makes this different from the IndustReal baseline?

The original IndustReal paper focuses on **Procedure Step Recognition (PSR)** via YOLOv8-based Assembly State Detection. This work takes a fundamentally different approach:

- Rather than recognizing procedural steps, it **detects execution errors in real time**
- Uses **deep temporal anomaly detection** (BiGRU + semi-supervised fusion)
- Validated on **5 unseen operators** — a harder generalization setting
- Produces **actionable cobot decisions** (NORMAL / WATCH / MONITOR / PAUSE / STOP)

To the best of our knowledge, this is the **first application of deep temporal anomaly detection to IndustReal**.
