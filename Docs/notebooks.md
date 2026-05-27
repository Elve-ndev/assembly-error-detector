# Notebooks

The pipeline is split across 5 notebooks, each covering a distinct stage.
All notebooks are available on Kaggle.

---

## 📊 EDA — Exploratory Data Analysis

`notebooks/EDA.ipynb`

Dataset exploration: recording statistics, action class distribution, error annotation density, gaze and hand joint coverage, stride map validation.

[![Open on Kaggle](https://img.shields.io/badge/Kaggle-Open%20notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/hibabou/checkpoint0-66)

---

## 01 — SlowFast Feature Extraction

`notebooks/01_slowfast_extraction.ipynb`

Loads SlowFast R50 with MECCANO weights, registers a forward hook on `blocks.4`, processes all 84 recordings, and saves 2304-dim feature tensors per frame to `data/features/`.

[![Open on Kaggle](https://img.shields.io/badge/Kaggle-Open%20notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/code/hibabou/extraction))

---

## 02 — BiGRU Preprocessing

`notebooks/02_bigru_preprocessing.ipynb`

Applies adaptive stride mapping, StandardScaler fitting, PSR error frame masking, and prepares train/val tensors for BiGRU training.

[![Open on Kaggle](https://img.shields.io/badge/Kaggle-Open%20notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/code/hibabou/entrainement))

---

## 03 — BiGRU Training

`notebooks/03_bigru_training.ipynb`

Trains the dual-head BiGRU (action + anomaly heads) with temporal attention and residual blocks. Logs F1, loss curves, and saves `bigru_2classes_best.pth`.

[![Open on Kaggle](https://img.shields.io/badge/Kaggle-Open%20notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/code/hibabou/bigru)

---

## 04 — Anomaly Detection & AUC-ROC

`notebooks/04_anomaly_detection_aucroc.ipynb`

Runs the full 4-component fusion pipeline (semi-supervised LR + prototype ratio + Mahalanobis RGB + Mahalanobis gaze), computes AUC-ROC curves, ablation study, and cobot decision thresholds.

[![Open on Kaggle](https://img.shields.io/badge/Kaggle-Open%20notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/code/hibabou/auc-roc)

[![Open on Kaggle](https://img.shields.io/badge/Kaggle-Open%20notebook-20BEFF?logo=kaggle)]((https://www.kaggle.com/code/hibabou/auc-roc))

---

> **Note**: update the Kaggle badge URLs above to point to the specific notebook URLs once published.
