# Overview

## Problem statement

Detecting assembly errors in real time from egocentric video is an open challenge in industrial robotics. Existing approaches either classify procedural steps (PSR) without checking execution correctness, or rely on controlled lab conditions that don't generalize to unseen operators.

This project builds a pipeline that answers: **"Is this action being executed correctly, right now?"** — and triggers a graded cobot response accordingly.

## What we built

An end-to-end pipeline combining:

- **SlowFast R50** (MECCANO pre-trained) for egocentric feature extraction
- **BiGRU dual-head** for simultaneous action recognition and anomaly scoring
- **Semi-supervised fusion** (LR + prototype ratio + Mahalanobis) for error detection
- **Rerun.io** live visualization with frame-level cobot decisions

## Key results

| Task | Metric | Value |
|---|---|---|
| Action recognition | F1 Macro | **0.663** (2 classes, 5 unseen operators) |
| Anomaly detection | AUC-ROC | **0.853** (7.5× random baseline) |
| Error coverage | Detection rate | **75.2%** of 803 annotated error frames |

## Dataset

Trained and validated on **IndustReal** (WACV 2024) — 84 egocentric recordings of industrial assembly with PSR error annotations, gaze, and hand joint data.

Reference: [TimSchoonbeek/IndustReal](https://github.com/TimSchoonbeek/IndustReal)
