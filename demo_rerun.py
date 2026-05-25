import rerun as rr
import rerun.blueprint as rrb
import numpy as np
import pandas as pd
import zipfile
import os
from PIL import Image, ImageDraw

# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────

VAL_P1_ZIP  = r"D:\cobotRerun\val_p1.zip"
SCORES_DIR  = r"D:\cobotRerun\scores"
PSR_VAL_DIR = r"D:\cobotRerun\psr_labels_val"
EXTRACT_DIR = r"D:\val_p1_extracted"

# UN SEUL recording pour tester
TARGET_RECS = ["05_assy_0_1"]

# ──────────────────────────────────────────────────────────
# VRAIES INFOS RECORDINGS
# ──────────────────────────────────────────────────────────

# Remplace REC_LENGTHS et rec_offsets par :
REC_INFO = {
    "20_assy_3_6" : {"stride":2, "f_min":33,  "n_feats":1472, "offset":0},
    "05_assy_2_2" : {"stride":2, "f_min":39,  "n_feats":1152, "offset":1472},
    "14_main_2_3" : {"stride":2, "f_min":33,  "n_feats":824,  "offset":2624},
    "26_assy_1_5" : {"stride":2, "f_min":39,  "n_feats":2280, "offset":3448},
    "24_assy_2_4" : {"stride":2, "f_min":39,  "n_feats":1464, "offset":5728},
    "05_assy_0_1" : {"stride":2, "f_min":39,  "n_feats":1448, "offset":7192},
    "14_assy_0_1" : {"stride":2, "f_min":39,  "n_feats":1488, "offset":8640},
    "24_assy_0_1" : {"stride":2, "f_min":39,  "n_feats":1064, "offset":10128},
    "05_main_0_1" : {"stride":2, "f_min":9,   "n_feats":680,  "offset":11192},
    "20_assy_0_1" : {"stride":2, "f_min":39,  "n_feats":1416, "offset":11872},
    "26_assy_0_1" : {"stride":2, "f_min":39,  "n_feats":1536, "offset":13288},
    "26_main_0_1" : {"stride":2, "f_min":39,  "n_feats":784,  "offset":14824},
    "20_main_0_1" : {"stride":2, "f_min":39,  "n_feats":1024, "offset":15608},
    "14_main_0_1" : {"stride":2, "f_min":33,  "n_feats":832,  "offset":16632},
    "14_main_2_2" : {"stride":2, "f_min":75,  "n_feats":688,  "offset":17464},
    "24_main_0_1" : {"stride":2, "f_min":39,  "n_feats":672,  "offset":18152},
}

# ──────────────────────────────────────────────────────────
# Thresholds
# ──────────────────────────────────────────────────────────

T_STOP    = 0.313
T_PAUSE   = 0.261
T_MONITOR = 0.221
T_WATCH   = 0.189

# ──────────────────────────────────────────────────────────
# Couleurs décisions
# ──────────────────────────────────────────────────────────

DECISION_COLORS = {
    "STOP"   : (255, 0,   0),
    "PAUSE"  : (255, 140, 0),
    "MONITOR": (255, 215, 0),
    "WATCH"  : (0,   200, 0),
    "NORMAL" : (50,  50,  50),
}

# ──────────────────────────────────────────────────────────
# Décision cobot
# ──────────────────────────────────────────────────────────

def get_decision(score):

    if score >= T_STOP:
        return "STOP"

    if score >= T_PAUSE:
        return "PAUSE"

    if score >= T_MONITOR:
        return "MONITOR"

    if score >= T_WATCH:
        return "WATCH"

    return "NORMAL"

# ──────────────────────────────────────────────────────────
# Overlay visuel
# ──────────────────────────────────────────────────────────

def add_overlay(
    img_arr,
    score,
    decision,
    is_error,
    error_desc=""
):

    img  = Image.fromarray(img_arr).convert("RGB")
    draw = ImageDraw.Draw(img)

    h, w = img_arr.shape[:2]

    col = DECISION_COLORS[decision]

    # ──────────────────────────────────────────────────────
    # Barre décision en haut
    # ──────────────────────────────────────────────────────

    draw.rectangle(
        [0, 0, w, 80],
        fill=col
    )

    draw.text(
        (20, 15),
        f"COBOT: {decision}",
        fill=(0, 0, 0)
    )

    draw.text(
        (20, 45),
        f"anomaly score = {score:.3f}",
        fill=(0, 0, 0)
    )

    # ──────────────────────────────────────────────────────
    # Barre score bas
    # ──────────────────────────────────────────────────────

    bar_w = int(score * w)

    draw.rectangle(
        [0, h-30, w, h],
        fill=(40, 40, 40)
    )

    draw.rectangle(
        [0, h-30, bar_w, h],
        fill=col
    )

    # Threshold lines
    thresholds = [
        (T_STOP,    (255, 0, 0)),
        (T_PAUSE,   (255, 140, 0)),
        (T_MONITOR, (255, 215, 0)),
        (T_WATCH,   (0, 200, 0)),
    ]

    for t, c in thresholds:

        x = int(t * w)

        draw.line(
            [(x, h-30), (x, h)],
            fill=c,
            width=3
        )

    # ──────────────────────────────────────────────────────
    # PSR error banner
    # ──────────────────────────────────────────────────────

    if is_error:

        draw.rectangle(
            [0, 80, w, 140],
            fill=(200, 0, 0)
        )

        draw.text(
            (20, 90),
            f"PSR: {error_desc[:60]}",
            fill=(255, 255, 255)
        )

    return np.array(img)

# ──────────────────────────────────────────────────────────
# Extraction frames
# ──────────────────────────────────────────────────────────

print("Extraction frames...")

os.makedirs(EXTRACT_DIR, exist_ok=True)

with zipfile.ZipFile(VAL_P1_ZIP) as z:

    for rec in TARGET_RECS:

        members = [
            m for m in z.namelist()
            if f"{rec}/rgb/" in m and m.endswith(".jpg")
        ]

        # 1 frame sur 3 pour réduire charge
        members_sub = members[::3]

        for m in members_sub:

            target = os.path.join(
                EXTRACT_DIR,
                m.replace("/", os.sep)
            )

            if not os.path.exists(target):
                z.extract(m, EXTRACT_DIR)

        print(f"  {rec}: {len(members_sub)} frames")

# ──────────────────────────────────────────────────────────
# Charger scores
# ──────────────────────────────────────────────────────────

print("Chargement scores...")

best_final = np.load(
    os.path.join(SCORES_DIR, "best_final.npy")
)

val_error = np.load(
    os.path.join(SCORES_DIR, "val_error.npy")
)

print(f"Scores loaded: {best_final.shape} ✅")

# ──────────────────────────────────────────────────────────
# Charger labels PSR
# ──────────────────────────────────────────────────────────

def load_psr(rec, psr_dir):

    psr_path = os.path.join(
        psr_dir,
        rec,
        "PSR_labels_with_errors.csv"
    )

    if not os.path.exists(psr_path):

        print(f"⚠️ PSR not found: {rec}")
        return {}

    df = pd.read_csv(
        psr_path,
        header=None,
        names=[
            "frame",
            "step_id",
            "description"
        ]
    )

    df["is_error"] = df["description"].str.contains(
        "Incorrect|Remove",
        case=False,
        na=False
    )

    df["frame_num"] = (
        df["frame"]
        .str.replace(".jpg", "", regex=False)
        .astype(int)
    )

    return {
        r["frame_num"]: r["description"]
        for _, r in df[df["is_error"]].iterrows()
    }

# ──────────────────────────────────────────────────────────
# Blueprint FIXED
# ──────────────────────────────────────────────────────────

print("Lancement rerun...")

blueprint = rrb.Blueprint(

    rrb.Vertical(

        rrb.Spatial2DView(
            name="Camera",
            origin="14_main_2_3",
        ),

        rrb.TimeSeriesView(
            name="Anomaly Scores",
            origin="14_main_2_3/scores",
        ),

        row_shares=[3, 1],
    )
)


rr.init(
    "CoBot Assembly Error Detection",
    spawn=True,
    default_blueprint=blueprint
)

# ──────────────────────────────────────────────────────────
# Offsets recordings
# ──────────────────────────────────────────────────────────

# Utiliser les vrais offsets depuis REC_INFO
rec_offsets = {r: REC_INFO[r]["offset"] for r in TARGET_RECS}

# ──────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────

for rec in TARGET_RECS:

    print(f"\nVisualisation {rec}...")

    rgb_dir = None

    for root, dirs, files in os.walk(EXTRACT_DIR):

        if os.path.basename(root) == "rgb" and rec in root:

            rgb_dir = root
            break

    if not rgb_dir:

        print(f"❌ RGB not found: {rec}")
        continue

    frames = sorted([
        f for f in os.listdir(rgb_dir)
        if f.endswith(".jpg")
    ])

    psr_errors = load_psr(rec, PSR_VAL_DIR)

    offset = rec_offsets.get(rec, 0)

    info = REC_INFO[rec]

    rec_scores = best_final[
     info["offset"]:
     info["offset"] + info["n_feats"]
    ]
    print(
        f"Frames: {len(frames)} | "
        f"Scores: {len(rec_scores)}"
    )

    # ──────────────────────────────────────────────────────
    # Frame loop
    # ──────────────────────────────────────────────────────

    for local_idx, frame_file in enumerate(frames):

        frame_num = int(
            frame_file.replace(".jpg", "")
        )

        # ──────────────────────────────────────────────────
        # ALIGNEMENT CORRECT FRAME -> FEATURE
        # ──────────────────────────────────────────────────

        feat_idx = round(
            (frame_num - info["f_min"]) / info["stride"]
        )

        feat_idx = max(
            0,
            min(feat_idx, len(rec_scores) - 1)
        )

        score = float(rec_scores[feat_idx])

        decision = get_decision(score)

        # ──────────────────────────────────────────────────
        # PSR nearby error
        # ──────────────────────────────────────────────────

        nearby_error = ""
        is_error = False

        for ef, ed in psr_errors.items():

            if abs(frame_num - ef) < 30:

                nearby_error = ed
                is_error = True
                break

        # ──────────────────────────────────────────────────
        # Timestamp
        # ──────────────────────────────────────────────────

        rr.set_time(
            "frame",
            sequence=frame_num
        )

        # ──────────────────────────────────────────────────
        # Image overlay
        # ──────────────────────────────────────────────────

        try:

            img_arr = np.array(
                Image.open(
                    os.path.join(
                        rgb_dir,
                        frame_file
                    )
                ).convert("RGB")
            )

            img_vis = add_overlay(
                img_arr,
                score,
                decision,
                is_error,
                nearby_error
            )

            rr.log(
                f"{rec}/camera",
                rr.Image(img_vis)
            )

        except Exception as e:

            print(f"Error: {e}")
            continue

        # ──────────────────────────────────────────────────
        # Scores
        # ──────────────────────────────────────────────────

        rr.log(
            f"{rec}/scores/anomaly",
            rr.Scalars(score)
        )

        rr.log(
            f"{rec}/scores/STOP",
            rr.Scalars(T_STOP)
        )

        rr.log(
            f"{rec}/scores/PAUSE",
            rr.Scalars(T_PAUSE)
        )

        rr.log(
            f"{rec}/scores/MONITOR",
            rr.Scalars(T_MONITOR)
        )

        rr.log(
            f"{rec}/scores/WATCH",
            rr.Scalars(T_WATCH)
        )

        # ──────────────────────────────────────────────────
        # Decision log
        # ──────────────────────────────────────────────────

        rr.log(
            f"{rec}/decision",
            rr.TextLog(
                f"[{decision}] score={score:.3f}"
            )
        )

        # ──────────────────────────────────────────────────
        # PSR error log
        # ──────────────────────────────────────────────────

        if is_error:

            rr.log(
                f"{rec}/psr_error",
                rr.TextLog(
                    f"PSR: {nearby_error}"
                )
            )

    print(f"✅ {rec} done")

# ──────────────────────────────────────────────────────────
# End
# ──────────────────────────────────────────────────────────

print("\n✅ Complete!")
print("Dans rerun :")
print("→ ralentir vitesse avec slider 1.00x")
print("→ mettre 0.1x pour slow motion")