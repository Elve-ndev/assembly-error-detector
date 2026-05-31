# Data & Checkpoints

## Model weights (Kaggle)

Weights are too large for the repository and are hosted as public Kaggle datasets.

| File | Kaggle dataset | Description |
|---|---|---|
| `bigru_2classes_best.pth` | [hibabou/checkpoint0-66](https://www.kaggle.com/datasets/hibabou/checkpoint0-66) | BiGRU dual-head, trained on 68 recordings, F1=0.663 |
| `meccano_slowfast_mapped_clean.pth` | [hibabou/poidsmecano](https://www.kaggle.com/datasets/hibabou/poidsmecano) | SlowFast R50 pre-trained on MECCANO, mapped to IndustReal |

Download and place both files in `checkpoints/`.

---

## IndustReal dataset

| Property | Value |
|---|---|
| Total recordings | 84 egocentric videos |
| Operators | 27 participants |
| Train + Test split | 68 recordings (22 operators) |
| Validation split | 16 recordings  |
| Action classes | 72 original → 2 semantic groups |
| Annotated errors | 163 PSR errors (47 recordings) |
| Modalities available | RGB, Depth, Stereo, Gaze, Hands, Pose, Ambient light |
| Modalities used | RGB, Gaze, Hands |

Reference: [TimSchoonbeek/IndustReal](https://github.com/TimSchoonbeek/IndustReal)

---

## Preprocessing files

These files are included in the repository under `data/`:

| File | Description |
|---|---|
| `stride_map_train.pkl` | Frame-to-feature alignment map for train recordings |
| `stride_map_val.pkl` | Frame-to-feature alignment map for val recordings |
| `rec_to_file_train.pkl` | Recording ID → file path mapping (train) |
| `rec_to_file_val.pkl` | Recording ID → file path mapping (val) |

---

## Repository structure

```
project/
├── notebooks/
│   ├── EDA.ipynb
│   ├── 01_slowfast_extraction.ipynb
│   ├── 02_bigru_preprocessing.ipynb
│   ├── 03_bigru_training.ipynb
│   └── 04_anomaly_detection_aucroc.ipynb
├── demo_rerun.py
├── checkpoints/
│   ├── meccano_slowfast_mapped_clean.pth   ← download from Kaggle
│   ├── bigru_2classes_best.pth             ← download from Kaggle
│   └── scaler.pkl
├── data/
│   ├── stride_map_train.pkl
│   ├── stride_map_val.pkl
│   ├── rec_to_file_train.pkl
│   └── rec_to_file_val.pkl
├── requirements.txt
└── README.md
```
