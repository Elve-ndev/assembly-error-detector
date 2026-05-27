# Requirements

## Python version

Python ≥ 3.12 required.

## Core dependencies

```
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

## Install

```bash
pip install -r requirements.txt
```

## Hardware

- GPU strongly recommended for SlowFast feature extraction (CUDA or MPS)
- CPU-only inference is supported for the BiGRU and decision engine
- Rerun visualization runs on any machine with a display
