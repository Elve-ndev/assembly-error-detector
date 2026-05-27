# Usage

The pipeline is split across 4 notebooks covering each stage, plus a live demo script.
Run them in order for a full reproduction.

---

## Step 1 — Feature extraction

**`notebooks/01_slowfast_extraction.ipynb`**

Loads SlowFast R50 with MECCANO weights, hooks `blocks.4`, processes all 84 recordings and saves 2304-dim feature tensors to `data/features/`.

```python
SF_WEIGHTS = "checkpoints/meccano_slowfast_mapped_clean.pth"
FEAT_DIR   = "data/features/"
BATCH      = "train_p1"
```

Outputs: `data/features/train/`, `data/features/val/`, `data/stride_map_train.pkl`, `data/stride_map_val.pkl`

---

## Step 2 — Preprocessing

**`notebooks/02_bigru_preprocessing.ipynb`**

Applies adaptive stride mapping, fits the `StandardScaler`, masks PSR error frames, and prepares train/val tensors.

Outputs: `checkpoints/scaler.pkl`, aligned tensors ready for BiGRU training.

---

## Step 3 — BiGRU training

**`notebooks/03_bigru_training.ipynb`**

Trains the dual-head BiGRU (action head + anomaly head) with temporal attention and residual blocks.

```python
TRAIN_FEAT_DIR = "data/features/train/"
VAL_FEAT_DIR   = "data/features/val/"
PREP_DIR       = "data/preprocessing/"
```

Outputs: `checkpoints/bigru_2classes_best.pth`

---

## Step 4 — Anomaly detection & evaluation

**`notebooks/04_anomaly_detection_aucroc.ipynb`**

Runs the full 4-component fusion (semi-supervised LR + prototype ratio + Mahalanobis RGB + Mahalanobis gaze), computes AUC-ROC curves, ablation study, and calibrates the 5 cobot decision thresholds.

Key result: **AUC-ROC = 0.853**, TPR@FPR=10% = 67.9%

---

## Live visualization

**`demo_rerun.py`**

Streams a validation recording through the full pipeline and visualizes frame-level decisions in Rerun.

```bash
python demo_rerun.py
```

Requires Rerun ≥ 0.32 and both checkpoints downloaded. See [Installation](installation.md).
