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

84 short egocentric recordings is a hard constraint. The validation set (16 recordings, 5 unseen operators) is too small to draw statistically robust conclusions. Performance variance across operators is significant.

### "Incorrectly installed" errors (~20% detection rate)

98 error frames annotated as incorrect part orientation are nearly invisible in RGB. The part looks placed but is rotated — depth features or stereo vision would be required to resolve 3D orientation. This is the main gap in the current detection coverage.

### No online learning

The prototype bank and LR classifier are fixed at inference time. The system cannot adapt to new operators or assembly variants without retraining.

### Latency (not yet measured on edge hardware)

The pipeline runs in real time on GPU but has not been benchmarked on edge hardware (Jetson). ONNX export and INT8 quantization are planned but not implemented.

### Annotation sparsity

163 PSR error sequences across 47 recordings means many recording types have zero error labels. The semi-supervised approach mitigates this but does not eliminate the bias.

---

## What we did not do (and why)

| Out of scope | Reason |
|---|---|
| Depth-based orientation detection | Requires stereo/depth hardware not available in IndustReal splits used |
| ROS2 integration | Out of scope for a research prototype |
| Per-operator calibration | Insufficient per-operator samples |
| Real cobot hardware testing | No access to physical cobot during development |

---

## Future work

- **ONNX + INT8 quantization** for Jetson edge deployment
- **Depth modality** to resolve 3D orientation errors (currently ~20% detection)
- **ROS2 node** for simulation and real cobot interfacing
- **Larger dataset validation** — if access to factory data becomes available
- **Online prototype update** — incremental adaptation to new operators
