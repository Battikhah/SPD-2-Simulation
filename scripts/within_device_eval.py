"""Within-device evaluation: can a model separate clean from infested recordings made by the SAME recorder?

Device IDs (IMEI) come from TreeVibes.xlsx. Only three recorders captured both classes in the development
data: ...7260 (infested 2, 4, 6 vs clean 8, all February), ...2538 (infested incl. 11-34 vs clean 24) and
...8391 (infested vs clean 25). Recorder ...8409 captured only clean folders 7 and 9, so any model can use
"device 8409 means clean". The scores are the existing out-of-fold scores (each recording scored by models
that never saw its folder); this only regroups them. Development data only; writes
outputs/folder-diagnostics/within_device.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grouped_cv_campaign import ROOT  # noqa: E402

MODELS = {
    "stream1s_w16 pb2 (fresh seeds)": ("full-stream1s_w16-pb2.0-s345", "stream1s_w16"),
    "stream1s_w8 pb1": ("full-pb1.0", "stream1s_w8"),
    "stream1s_w8_d4 pb2 KD (fresh seeds)": ("full-stream1s_w8_d4-pb2.0-kd-s345", "stream1s_w8_d4"),
    "stream1s_w16 pb2, mel bands < 3 kHz only": ("full-stream1s_w16-pb2.0-mb56", "stream1s_w16"),
    "stream1s_w16 pb2, 3-4 kHz bands only": ("full-stream1s_w16-pb2.0-top8", "stream1s_w16"),
    "stream1s_w16 pb2, gain-normalised": ("full-stream1s_w16-pb2.0-gn", "stream1s_w16"),
    "stream1s_w8 pb1, not trained on recorder 8409": ("full-stream1s_w8-pb1.0-no8409", "stream1s_w8"),
    "stream1s_w16 pb2, not trained on recorder 8409": ("full-stream1s_w16-pb2.0-no8409", "stream1s_w16"),
    "stream1s_w8 pb1, not trained on folder 9 (silence)": ("full-stream1s_w8-pb1.0-no9", "stream1s_w8"),
    "stream1s_w8 pb1, leave-one-recorder-out": ("full-stream1s_w8-pb1.0-lorO", "stream1s_w8"),
    "stream1s_w16 pb2, leave-one-recorder-out": ("full-stream1s_w16-pb2.0-lorO", "stream1s_w16"),
    "stream1s_w8_d4 pb2, leave-one-recorder-out": ("full-stream1s_w8_d4-pb2.0-lorO", "stream1s_w8_d4"),
    "stream1s_w8_d4 pb2, leave-one-recorder-out, seeds 3-5": ("full-stream1s_w8_d4-pb2.0-lorO-s345",
                                                              "stream1s_w8_d4"),
    "stream1s_w8_d4 pb2, leave-one-recorder-out, KD": ("full-stream1s_w8_d4-pb2.0-lorO-kd", "stream1s_w8_d4"),
    "stream1s_w4 pb2, leave-one-recorder-out": ("full-stream1s_w4-pb2.0-lorO", "stream1s_w4"),
    "stream1s_ds pb1, leave-one-recorder-out": ("full-stream1s_ds-pb1.0-lorO", "stream1s_ds"),
    "stream1s_w8 pb1, leave-one-recorder-out, folder-balanced": ("full-stream1s_w8-pb1.0-lorO-bal0.5",
                                                                 "stream1s_w8"),
    "stream1s_w8 pb1, leave-one-recorder-out, folder-balanced, seeds 3-5": (
        "full-stream1s_w8-pb1.0-lorO-bal0.5-s345", "stream1s_w8"),
    "stream1s_w8_d4 pb2, leave-one-recorder-out, folder-balanced": ("full-stream1s_w8_d4-pb2.0-lorO-bal0.5",
                                                                    "stream1s_w8_d4"),
    "stream1s_w8 pb1, not trained on folder 7 (wind)": ("full-stream1s_w8-pb1.0-no7", "stream1s_w8"),
    "stream1s_w16 pb2, leave-one-recorder-out, 3-4 kHz only": ("full-stream1s_w16-pb2.0-lorO-top8", "stream1s_w16"),
    "stream1s_w16 pb2, leave-one-recorder-out, < 3 kHz only": ("full-stream1s_w16-pb2.0-lorO-mb56", "stream1s_w16"),
}


def main():
    meta = pd.read_excel(ROOT / "TreeVibes.xlsx", dtype={"IMEI": str})[["FOLDER", "IMEI"]]
    device = dict(zip(meta.FOLDER, meta.IMEI.str[-4:]))
    shape = pd.read_csv("outputs/folder-diagnostics/spectral_shape.csv")
    scores = {"near-Nyquist roll-off (no learning)": -shape.rolloff.to_numpy()}
    for name, (d, arch) in MODELS.items():
        path = Path("outputs/esp32-tflm") / d / "oof_int8.csv"
        if not path.exists():  # not exported yet: fall back to the float scores, labelled as such
            path, name = path.with_name("oof_float.csv"), name + " [float]"
            if not path.exists() or not any(c.startswith(arch + "|seed") for c in pd.read_csv(path, nrows=1)):
                continue
        oof = pd.read_csv(path)
        assert list(oof.folder) == list(shape.folder)
        scores[name] = oof[[c for c in oof.columns if c.startswith(arch + "|seed")]].astype("float32").mean(1).to_numpy()
    y, folder = shape.label.to_numpy(), shape.folder.to_numpy()
    dev = np.array([device[f] for f in folder])
    lines = ["ROC-AUC within each recorder that captured both classes (out-of-fold scores, int8).", "",
             "| Scorer | All data | Device 7260 (infested 2/4/6 vs clean 8) | Device 2538 (vs clean 24) | "
             "Device 8391 (vs clean 25) |", "| --- | ---: | ---: | ---: | ---: |"]
    counts = {d: (int((y[dev == d] == 1).sum()), int((y[dev == d] == 0).sum())) for d in ("7260", "2538", "8391")}
    for name, s in scores.items():
        cells = [f"{roc_auc_score(y, s):.3f}"]
        for d in ("7260", "2538", "8391"):
            m = dev == d
            cells.append(f"{roc_auc_score(y[m], s[m]):.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Recordings (infested / clean): " + ", ".join(f"device {d}: {p} / {n}" for d, (p, n) in counts.items()),
              "Device 8409 recorded only clean folders 7 and 9 (683 recordings)."]
    Path("outputs/folder-diagnostics/within_device.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
