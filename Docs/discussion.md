# Discussion

## What we solved

### The core problem

Detecting **execution errors** in real-time egocentric video for industrial assembly — not just recognizing what action is being done, but whether it is done correctly.

### What works

**Semi-supervised anomaly detection** is the main contribution. With only 163 annotated error sequences across 84 recordings, reaching AUC-ROC = 0.853 required:

- PSR error masking so BiGRU learns exclusively from clean sequences
- Prototype ratio scoring: frames closer to the error prototype than the normal prototype score higher
- Logistic regression on PCA-64 GRU hidden states (93.3% variance, AUC = 0.824 standalone)
- Temporal smoothing (W=20) that reduces frame-level noise without adding latency

**Domain-adapted features** matter significantly. Switching from EfficientNet-B0 (ImageNet) to SlowFast R50 (MECCANO) pushed F1 from 0.136 to 0.663 — a 5× improvement driven purely by pre-training domain match.

**Semantic grouping** unlocks generalization. 72 fine-grained classes → 2 operational categories raised F1 from 0.097 to 0.663 while making the decision boundary industrially meaningful (NON-CRITICAL vs CRITICAL).

---

## Limitations

### Dataset size

84 short egocentric recordings is a hard constraint. The validation set (16 recordings) is too small to draw statistically robust conclusions. Performance variance across operators is significant.

### "Incorrectly installed" errors (~20% detection rate)

98 error frames annotated as incorrect part orientation are nearly invisible in RGB. The part looks placed but is rotated — depth features or stereo vision would be required to resolve 3D orientation. This is the main gap in the current detection coverage.

### ONNX

The pipeline runs in real time on GPU but has not been benchmarked on edge hardware (Jetson).(ONNX export and INT8 quantization )

### Class imbalance

1,696 annotated error frames against 33,920 normal frames (ratio 1:20). Even with SMOTE and weighted loss, the model is biased toward normal sequences — rare error patterns are underrepresented in the decision boundary.

### Dataset size

84 short recordings is insufficient to draw statistically robust conclusions. The validation set (16 recordings) is too small — performance variance across operators is high and results may not generalize beyond IndustReal.

---

## What we did not do (and why)

| Out of scope | Reason |
|---|---|
| ROS2 integration | Requires access to physical cobot hardware and a real time OS which is unavailable during development |
| Per-operator calibration | Insufficient per-operator samples |
| Real cobot hardware testing | No access to physical cobot during development |

---

## Future work

- **ONNX + INT8 quantization** for Jetson edge deployment
- **Depth modality and bigger dataset ** to resolve 3D orientation errors 
- **ROS2 node** for simulation and real cobot interfacing
- **Larger dataset validation** — if access to factory data becomes available

