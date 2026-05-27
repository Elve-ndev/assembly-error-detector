# Live Demo — Rerun Visualization

## Demo video

The video below shows the real-time pipeline running on a validation recording.

```{raw} html
<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;border-radius:8px;margin:1rem 0;">
  <video
    src="https://github.com/user-attachments/assets/3f64cb51-ec63-4741-b399-d35b3f75e13b"
    style="position:absolute;top:0;left:0;width:100%;height:100%;border-radius:8px;"
    controls autoplay loop muted
    title="Cobot Assembly Error Detection — Rerun live visualization">
  </video>
</div>
```

---

## What you see in the video

| Frames | Decision | Color | Event |
|---|---|---|---|
| #328 – #345 | MONITOR | 🟡 Yellow | Standard preparation steps tracked |
| #367 – #375 | PAUSE | 🟠 Orange | PSR annotation: *Remove rear chassis pin* |
| #382 – #412 | STOP | 🔴 Red | Anomaly score > 0.313 — critical execution error |

Frame #367–#375 aligns precisely with the authors' PSR annotation, validating the pipeline's temporal precision.

---

## What Rerun streams

- **Egocentric RGB frames** with decision overlay (color-coded band + score bar)
- **PSR ground-truth error annotations** per frame
- **Anomaly score timeline** with threshold reference lines (WATCH / MONITOR / PAUSE / STOP)
- **Per-frame cobot decision** text label

---
## other examples

```{raw} html
<video width="100%" controls autoplay loop muted>
  <source src="https://github.com/user-attachments/assets/d319a0ee-b9d1-4e96-9f47-c462616aaba4" type="video/mp4">
</video>
```




---
## Run it yourself

```bash
python demo_rerun.py
```

Requires Rerun ≥ 0.32 and the model checkpoints. See [Installation](installation.md) for setup.

---

## Decision color coding

| Decision | Color | Score threshold |
|---|---|---|
| NORMAL | ⬜ White | < 0.189 |
| WATCH | 🔵 Blue | ≥ 0.189 |
| MONITOR | 🟡 Yellow | ≥ 0.221 |
| PAUSE | 🟠 Orange | ≥ 0.261 |
| STOP | 🔴 Red | ≥ 0.313 |
