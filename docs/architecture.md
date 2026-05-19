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
│  - Output: 2304-dimensional feature vectors                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Preprocessing & Normalization                        │
│  - StandardScaler (fit on training data)                         │
│  - Outlier clipping (±3σ)                                        │
│  - Missing value handling                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         BiGRU Dual-Head with Temporal Attention                  │
│                                                                   │
│  Input: normalized features                                      │
│  BiGRU: 2 layers, hidden=256, bidirectional                     │
│  Attention: frame-level weights [0, 1]                          │
│                                                                   │
│  ├─→ Head 1: Action Classification                              │
│  │   (softmax over 10 semantic classes)                          │
│  │                                                                │
│  └─→ Head 2: Anomaly Scoring                                    │
│      (sigmoid output in [0, 1])                                  │
└──┬────────────────────────────────────────────────────┬──────────┘
   │                                                    │
   ▼                                                    ▼
┌──────────────────────────┐  ┌──────────────────────────────────┐
│  Viterbi Decoder         │  │  Mahalanobis Anomaly Detection   │
│                          │  │                                  │
│ - Learned transitions    │  │ - Prototype bank (training data) │
│ - Smoothing (Laplace)    │  │ - Per-class Mahalanobis dist.   │
│ - Min segment duration   │  │ - Nearest prototype scoring     │
│                          │  │                                  │
│ Output:                  │  │ Output:                          │
│ Coherent action seq.     │  │ Unsupervised anomaly score      │
└──────────────────────────┘  └──────────────────────────────────┘
           │                              │
           └──────────────┬───────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │   Cobot Decision Engine             │
        │                                     │
        │  Decision Logic:                    │
        │  - Integrate: action + consistency  │
        │  - Threshold: anomaly scores        │
        │  - Output: NORMAL / MONITOR /       │
        │            PAUSE / STOP             │
        └─────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │   Cobot Intervention                │
        │                                     │
        │  - Continue procedure               │
        │  - Increase observation             │
        │  - Request confirmation             │
        │  - Full stop + alert                │
        └─────────────────────────────────────┘
```

## Component Details

### 1. Feature Extraction: SlowFast R50

**Architecture:**
- Two pathways with different temporal strides:
  - **Slow pathway**: processes frames at lower frequency, captures semantics
  - **Fast pathway**: processes all frames, captures motion details
- Fusion at multiple scales before classification head

**Configuration:**
```python
backbone = SlowFast(
    depth=(3, 4, 6, 3),           # ResNet-50 depths
    num_classes=400,               # ImageNet-pretrained
    dropout_rate=0.5,
    norm_layer=nn.BatchNorm3d
)

# Hook at blocks.4 (before final pooling)
# Extracts: (slow, fast) concatenated → 2304 dims
```

**Pre-training:**
- MECCANO dataset (industrial egocentric actions)
- Weights optimized for assembly/manipulation tasks

### 2. BiGRU Dual-Head Architecture

**Motivation:**
- Action classification and anomaly detection are related but distinct tasks
- Single network can share lower-level representations
- Dual heads allow independent calibration and thresholding

**Components:**

```python
class BiGRUDualHead(nn.Module):
    def __init__(self, feature_dim=2304, hidden_dim=256, num_classes=10):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Bidirectional GRU
        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # Temporal Attention
        self.attention = TemporalAttention(hidden_dim * 2)
        
        # Action Head
        self.action_head = nn.Linear(hidden_dim * 2, num_classes)
        
        # Anomaly Head
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        # features: (batch, seq_len, 2304)
        
        # Project input
        x = self.input_proj(features)  # (batch, seq_len, 256)
        
        # BiGRU encoding
        gru_out, _ = self.bigru(x)  # (batch, seq_len, 512)
        
        # Attention
        attn_weights = self.attention(gru_out)  # (batch, seq_len, 1)
        
        # Action prediction
        action_logits = self.action_head(gru_out)  # (batch, seq_len, 10)
        
        # Anomaly scoring
        anomaly_scores = self.anomaly_head(gru_out)  # (batch, seq_len, 1)
        
        return action_logits, anomaly_scores, attn_weights
```

### 3. Viterbi Temporal Decoder

**Purpose:** Enforce procedural consistency and prevent over-segmentation

**Algorithm:**
```
Transition Matrix (learned):
  T[i, j] = log P(action_j | action_i)
  
Per-frame costs:
  C[t, j] = -log P(action_j | features_t)
  
Viterbi recursion:
  V[t, j] = min_i { V[t-1, i] + T[i, j] + C[t, j] }
  
Backtracking:
  optimal_path = argmin V[T, :]
  + segment duration constraints
```

**Parameters:**
- Laplace smoothing: α = 1.0 (prevent zero transitions)
- Minimum segment duration: 3 frames (prevent flickering)
- Transition threshold: only allow transitions with P > 0.01

### 4. Mahalanobis Anomaly Detection

**Prototype Building (Training):**

1. For each action class:
   - Collect all GRU hidden states for frames labeled with that class
   - Compute class mean: μ_c = mean(h_c)
   - Compute covariance: Σ_c = cov(h_c)

2. Store: {μ_c, Σ_c}^C_{c=1}

**Scoring (Inference):**

```
For each frame with hidden state h_t:
  1. Find nearest prototype:
     k = argmin_c ||μ_c - h_t||²
  
  2. Compute Mahalanobis distance:
     d_t = sqrt((h_t - μ_k)^T * Σ_k^{-1} * (h_t - μ_k))
  
  3. Normalize anomaly score:
     anomaly_score = sigmoid((d_t - threshold) / scale)
```

### 5. Cobot Decision Engine

**Decision Logic:**

```python
def decide(action_logit, anomaly_score, viterbi_action, bigru_action):
    
    action_conf = softmax(action_logit).max()
    consistency = (viterbi_action == bigru_action)
    
    if anomaly_score > CRITICAL_THRESHOLD:
        return "STOP", "Critical anomaly detected"
    
    elif anomaly_score > PAUSE_THRESHOLD:
        return "PAUSE", f"Elevated anomaly: {anomaly_score:.2f}"
    
    elif anomaly_score > MONITOR_THRESHOLD or not consistency:
        return "MONITOR", "Moderate anomaly or sequence inconsistency"
    
    elif action_conf > CONFIDENCE_THRESHOLD:
        return "NORMAL", f"Action: {action_name}"
    
    else:
        return "MONITOR", "Low confidence action prediction"
```

**Thresholds:**
- CRITICAL_THRESHOLD = 0.85
- PAUSE_THRESHOLD = 0.65
- MONITOR_THRESHOLD = 0.45
- CONFIDENCE_THRESHOLD = 0.70

## Data Flow: Training vs. Inference

### Training Pipeline

```
Raw Video → Frame Sampling (30 FPS) → Optical Flow (optional)
    ↓
Feature Extraction (SlowFast, frozen weights)
    ↓
GRU Encoding → Dual Heads (action + anomaly)
    ↓
Loss computation (action + anomaly + regularization)
    ↓
Backprop through BiGRU (SlowFast frozen)
    ↓
Collect GRU hidden states → Build Mahalanobis prototypes
    ↓
Save: BigRU weights + prototype bank + scalers
```

### Inference Pipeline

```
Live egocentric video stream (30 FPS)
    ↓
Real-time feature extraction (SlowFast)
    ↓
Feature normalization (using training scalers)
    ↓
BiGRU forward pass → action + anomaly predictions
    ↓
Viterbi decoding (1-frame latency)
    ↓
Mahalanobis anomaly scoring
    ↓
Decision engine → NORMAL / MONITOR / PAUSE / STOP
    ↓
Cobot intervention command
    ↓
Latency: ~33ms per frame (30 FPS real-time capable)
```

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Action Accuracy | ~85% | 10 semantic classes |
| Anomaly F1-Score | ~0.82 | Precision vs. Recall trade-off |
| Inference Latency | 33ms | Per-frame on RTX 3090 |
| Throughput | 30 FPS | Real-time capable |
| Memory Footprint | ~2GB | Model + feature buffer |
| Feature Extraction | 15ms | SlowFast backbone |
| BiGRU Forward | 5ms | 2-layer bidirectional |
| Decision Engine | 1ms | Threshold logic |

## Edge Device Deployment

For deployment on collaborative robots:

- **Quantization**: INT8 quantization reduces model size by 4x
- **Distillation**: Smaller BiGRU (1 layer, hidden=128) for edge devices
- **Batch processing**: Accumulate frames → batch inference for efficiency
- **GPU requirement**: NVIDIA Jetson Orin (12GB VRAM) or RTX 2060+
