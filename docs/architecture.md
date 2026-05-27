# System Architecture

## High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Egocentric Video Stream                       │
│                   (RGB Frames @ 30 FPS)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Feature Extraction (SlowFast R50)                   │
│  - MECCANO pre-trained weights                                   │
│  - Hook at blocks.4: slow + fast pathways                        │
│  - slow: [B, 2048, 8, 7, 7] + fast: [B, 256, 32, 7, 7]        │
│  - avg spatial + concat → 2304-dim per frame                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Adaptive Stride Mapping + Normalization             │
│  - Per-recording stride/f_min/n_feats alignment                  │
│  - StandardScaler (fit on train+test, 68 recordings)             │
│  - Outlier clipping [-5, +5]                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         BiGRU Dual-Head with Temporal Attention                  │
│                                                                   │
│  Linear(2304→512) + LayerNorm + GELU + ResidualBlock            │
│  BiGRU: hidden=512, layers=2, bidirectional → 1024-dim          │
│  Temporal Attention: context-aware frame weighting               │
│  PSR error masking: 163 errors masked during training            │
│                                                                   │
│  ├─→ Action Head: ResidualBlock → Linear(512, 2)                │
│  │   NON-CRITICAL / CRITICAL  |  F1 Macro = 0.663               │
│  │                                                                │
│  └─→ Anomaly Head: Linear(1024, 1) + Sigmoid                   │
│      continuous score [0, 1]                                     │
└──┬──────────────────────────────────────────────┬───────────────┘
   │                                              │
   ▼                                              ▼
┌──────────────────────────┐  ┌──────────────────────────────────────┐
│  GRU Hidden States       │  │  Multimodal Features                 │
│  PCA-64 (93.3% variance) │  │  Gaze (x,y) + Hands (42-dim)        │
└──────────┬───────────────┘  └─────────────────┬────────────────────┘
           │                                     │
           ▼                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│              Anomaly Detection Stack                              │
│                                                                   │
│  Semi-supervised LR    (w=0.5) → AUC = 0.824                    │
│  Prototype Ratio       (w=0.3) → AUC = 0.831                    │
│  Mahalanobis RGB PCA64 (w=0.1) → AUC = 0.668                    │
│  Mahalanobis Gaze      (w=0.1) → AUC = 0.592                    │
│  Temporal Smoothing    W=20 frames                                │
│                                                                   │
│  Weights: grid search over val set (AUC-ROC maximization)        │
│  Final AUC-ROC = 0.853 | TPR@FPR=10% = 0.679                    │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────┐
        │   Cobot Decision Engine             │
        │                                     │
        │  STOP    : score ≥ 0.313 (FPR=2%)  │
        │  PAUSE   : score ≥ 0.261 (FPR=5%)  │
        │  MONITOR : score ≥ 0.221 (FPR=10%) │
        │  WATCH   : score ≥ 0.189 (FPR=20%) │
        │  NORMAL  : score < 0.189            │
        │                                     │
        │  75.2% errors detected (604/803)    │
        └─────────────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────┐
        │   Rerun.io Live Visualization       │
        │   + Cobot Intervention Command      │
        └─────────────────────────────────────┘
```

## Component Details

### 1. Feature Extraction: SlowFast R50

**Architecture:**
- Two pathways with different temporal strides:
  - **Slow pathway**: low frame rate, captures spatial semantics
  - **Fast pathway**: high frame rate, captures motion details
- Hook at blocks.4 before final pooling head

**Configuration:**
```python
# Hook placement
hooks = {
    "s4.pathway0_res2": slow_pathway,   # [B, 2048, 8, 7, 7]
    "s4.pathway1_res2": fast_pathway,   # [B,  256, 32, 7, 7]
}
# avg pool spatial + temporal → concat → 2304-dim
```

**Pre-training:**
- MECCANO dataset (Ragusa et al., 2021) — industrial egocentric assembly actions
- Weights remapped to IndustReal feature extraction pipeline

### 2. Adaptive Stride Mapping

Each recording has a different number of extracted features depending on frame count and extraction stride.

```python
# Per-recording mapping (example)
stride_map = {
    "14_main_2_3": {"stride": 2, "f_min": 33, "n_feats": 824,  "offset": 2624},
    "05_main_0_1": {"stride": 2, "f_min": 9,  "n_feats": 680,  "offset": 11192},
    ...
}

# Frame → feature index
feat_idx = round((frame_num - f_min) / stride)
feat_idx = max(0, min(feat_idx, n_feats - 1))
```

### 3. BiGRU Dual-Head Architecture

```python
class BiGRUDualHead(nn.Module):
    def __init__(self, input_dim=2304, hidden_dim=512,
                 num_classes=2, dropout=0.4):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout)
        )

        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        self.norm      = nn.LayerNorm(hidden_dim * 2)
        self.attention = TemporalAttention(hidden_dim)

        self.action_head = nn.Sequential(
            ResidualBlock(hidden_dim * 2, dropout),
            nn.Linear(hidden_dim * 2, num_classes)
        )

        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, lengths=None):
        x = self.input_proj(x)
        gru_out, _ = self.bigru(x)
        gru_out = self.norm(gru_out)
        attn = self.attention(gru_out)
        action_logits = self.action_head(gru_out)
        anomaly_score = self.anomaly_head(gru_out)
        return action_logits, anomaly_score, attn, gru_out
```

**Training details:**
- PSR error frames masked (label=-1) during action loss computation
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Early stopping: patience=5 on val F1 macro
- Parameters: ~10.5M

### 4. Semi-supervised Anomaly Detection

**Training data:**
- 33,920 normal frames (train set, no errors)
- 1,696 annotated error frames (PSR labels, train set)
- Ratio 1:20 (errors:normal) — selected by grid search

**Feature space:**
- GRU hidden states (1024-dim) → PCA-64 (93.3% variance explained)
- StandardScaler applied before PCA

**Classifier:**
```python
clf = LogisticRegression(max_iter=1000, C=0.01)
clf.fit(X_train_pca64, y_train)  # y=0 normal, y=1 error
scores = clf.predict_proba(X_val_pca64)[:, 1]
# AUC-ROC = 0.824
```

### 5. Prototype Ratio Score

```python
# Prototypes built from training data
proto_normal = X_train_normal_pca64.mean(axis=0)
proto_error  = X_train_error_pca64.mean(axis=0)

# Per-frame score
dist_normal = mahalanobis(feat, proto_normal, inv_cov_normal)
dist_error  = mahalanobis(feat, proto_error,  inv_cov_error)
ratio_score = dist_normal / (dist_error + 1e-8)
# AUC-ROC = 0.831
```

### 6. Final Score Combination

```python
from scipy.ndimage import uniform_filter1d

final_score = uniform_filter1d(
    0.5 * norm(semi_supervised_score) +
    0.3 * norm(ratio_score)           +
    0.1 * norm(mahalanobis_rgb_score) +
    0.1 * norm(mahalanobis_gaze_score),
    size=20  # temporal smoothing W=20
)
# AUC-ROC = 0.853
```

Weights selected by grid search over val set maximizing AUC-ROC.

### 7. Cobot Decision Engine

```python
THRESHOLDS = {
    "STOP"   : 0.313,  # FPR=2%
    "PAUSE"  : 0.261,  # FPR=5%
    "MONITOR": 0.221,  # FPR=10%
    "WATCH"  : 0.189,  # FPR=20%
}

def decide(score):
    if score >= THRESHOLDS["STOP"]:    return "STOP"
    if score >= THRESHOLDS["PAUSE"]:   return "PAUSE"
    if score >= THRESHOLDS["MONITOR"]: return "MONITOR"
    if score >= THRESHOLDS["WATCH"]:   return "WATCH"
    return "NORMAL"
```

Thresholds derived from ROC curve at fixed FPR operating points.

## Data Flow: Training vs. Inference

### Training Pipeline

```
Raw Video → SlowFast Feature Extraction (frozen)
    ↓
Adaptive stride mapping + StandardScaler
    ↓
BiGRU Dual-Head training
  - PSR error frames masked (label=-1)
  - Action loss + Anomaly loss
    ↓
Collect GRU hidden states → PCA-64
    ↓
Semi-supervised LR training (1696 errors + 33920 normal)
Prototype Ratio computation (normal + error prototypes)
Mahalanobis bank (RGB + Gaze + Hands)
    ↓
Save: BiGRU weights + scaler + stride_map +
      LR classifier + PCA + prototypes
```

### Inference Pipeline

```
Live egocentric video (30 FPS)
    ↓
SlowFast feature extraction (2304-dim)
    ↓
Stride-aligned normalization
    ↓
BiGRU forward → hidden states
    ↓
Semi-supervised LR + Prototype Ratio + Mahalanobis
    ↓
Weighted combination + temporal smoothing W=20
    ↓
Threshold → NORMAL / WATCH / MONITOR / PAUSE / STOP
    ↓
Rerun.io overlay + Cobot intervention
```

## Performance Summary

| Component | Metric | Value |
|-----------|--------|-------|
| Feature discriminability | Intra/inter ratio | 2.75 |
| Action recognition | F1 Macro | 0.663 |
| Action recognition | F1 Weighted | 0.701 |
| Anomaly detection | AUC-ROC | **0.853** |
| Anomaly detection | TPR@FPR=10% | 0.679 |
| Anomaly detection | Average Precision | 0.321 |
| Anomaly detection | Lift over random | **7.53×** |
| Error detection | Total detected | 75.2% (604/803) |
| Error detection | STOP+PAUSE | 53.1% (426/803) |
| "Remove" errors | Detection rate | ~88% |
| "Incorrectly installed" | Detection rate | ~20% (RGB limit) |
