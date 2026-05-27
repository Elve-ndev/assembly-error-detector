# Installation

## Prerequisites

- Python ≥ 3.12
- PyTorch ≥ 2.0 (CUDA recommended for feature extraction)
- Git

## Clone the repository

```bash
git clone https://github.com/Elve-ndev/cobot-assembly-detection
cd cobot-assembly-detection
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Download model checkpoints

Weights are hosted on Kaggle. Download them and place in the `checkpoints/` folder:

| File | Source | Description |
|---|---|---|
| `bigru_2classes_best.pth` | [hibabou/checkpoint0-66](https://www.kaggle.com/datasets/hibabou/checkpoint0-66) | BiGRU dual-head — F1=0.663 |
| `meccano_slowfast_mapped_clean.pth` | [hibabou/poidsmecano](https://www.kaggle.com/datasets/hibabou/poidsmecano) | SlowFast R50 MECCANO weights |

```
checkpoints/
├── bigru_2classes_best.pth
├── meccano_slowfast_mapped_clean.pth
└── scaler.pkl
```

> `scaler.pkl` is included in the repository.

## Dataset

Download IndustReal from the [official repository](https://github.com/TimSchoonbeek/IndustReal) and follow their setup instructions. Place recordings under `data/`.

## Verify installation

```bash
python -c "import torch; import rerun; print('OK')"
```

## Quick start — live demo

```bash
python demo_rerun.py
```

This launches the Rerun visualization with a sample recording.
