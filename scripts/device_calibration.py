"""Simulate per-device threshold calibration from known-healthy recordings. Development data only.

For a recorder-held-out run (esp32_train.py --folds recorder), each recorder that captured both classes
is treated as a newly installed device. A random half of its clean recordings plays the role of
calibration recordings from known-healthy trees. The threshold is set at their q-th score quantile
(q = 0.90 aims at about 10% false alarms on healthy trees). It is evaluated on that device's other clean
half and all of its infested recordings. 200 random splits per recorder. The models never saw the
device; only its healthy-tree scores are used, never its infested labels.
Writes outputs/folder-diagnostics/device_calibration.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grouped_cv_campaign import ROOT  # noqa: E402

RUNS = {
    "stream1s_w16 pb2": ("full-stream1s_w16-pb2.0-lorO", "stream1s_w16"),
    "stream1s_w8_d4 pb2": ("full-stream1s_w8_d4-pb2.0-lorO", "stream1s_w8_d4"),
    "stream1s_w8 pb1": ("full-stream1s_w8-pb1.0-lorO", "stream1s_w8"),
    "stream1s_w8 pb1 folder-balanced": ("full-stream1s_w8-pb1.0-lorO-bal0.5", "stream1s_w8"),
    "stream1s_w8 pb1 folder-balanced, seeds 3-5": ("full-stream1s_w8-pb1.0-lorO-bal0.5-s345", "stream1s_w8"),
    "stream1s_w8_d4 pb2 folder-balanced": ("full-stream1s_w8_d4-pb2.0-lorO-bal0.5", "stream1s_w8_d4"),
    "stream1s_w8_d4 pb2 folder-balanced, seeds 3-5": ("full-stream1s_w8_d4-pb2.0-lorO-bal0.5-s345",
                                                       "stream1s_w8_d4"),
    "stream1s_w8_d4 pb2 folder-balanced p=1": ("full-stream1s_w8_d4-pb2.0-lorO-bal1", "stream1s_w8_d4"),
}
DEVICES = ("7260", "2538", "8391")


def calibration_size(device, rng, q, reps, run=("full-stream1s_w8_d4-pb2.0-lorO-bal0.5", "stream1s_w8_d4")):
    """How many healthy-tree 10 s recordings does calibration need? Recorder 7260 (238 clean): calibrate on
    n of them, test on the clean recordings not used and all infested; report the spread of false alarms."""
    oof = pd.read_csv(Path("outputs/esp32-tflm") / run[0] / "oof_int8.csv")
    s = oof[[c for c in oof.columns if c.startswith(run[1] + "|seed")]].astype("float32").mean(1).to_numpy()
    y, dev = oof.label.to_numpy(), np.array([device[f] for f in oof.folder])
    pos, neg = s[(dev == "7260") & (y == 1)], s[(dev == "7260") & (y == 0)]
    out = ["", f"Calibration set size ({run[1]} folder-balanced, recorder 7260, {q:.0%} quantile, {reps} splits):", "",
           "| Healthy recordings used | Mean recall | Mean FPR | FPR 10th-90th percentile |",
           "| ---: | ---: | ---: | --- |"]
    for n in (5, 10, 20, 50, 100):
        r, f = [], []
        for _ in range(reps):
            perm = rng.permutation(len(neg))
            thr = np.quantile(neg[perm[:n]], q)
            r.append((pos > thr).mean())
            f.append((neg[perm[n:]] > thr).mean())
        lo, hi = np.percentile(f, [10, 90])
        out.append(f"| {n} | {100 * np.mean(r):.0f}% | {100 * np.mean(f):.0f}% | {100 * lo:.0f}–{100 * hi:.0f}% |")
    return out


def main(q=0.90, reps=200):
    meta = pd.read_excel(ROOT / "TreeVibes.xlsx", dtype={"IMEI": str})
    device = dict(zip(meta.FOLDER, meta.IMEI.str[-4:]))
    rng = np.random.default_rng(0)
    lines = [f"Per-device calibration: threshold = {q:.0%} quantile of half the device's clean recordings; "
             f"mean over {reps} splits. Cells: recall / FPR on the other half (%).", "",
             "| Model | " + " | ".join(f"Recorder {d}" for d in DEVICES) + " | Mean recall / FPR |",
             "| --- |" + " --- |" * (len(DEVICES) + 1)]
    for name, (d, arch) in RUNS.items():
        path = Path("outputs/esp32-tflm") / d / "oof_int8.csv"
        if not path.exists():
            continue
        oof = pd.read_csv(path)
        s = oof[[c for c in oof.columns if c.startswith(arch + "|seed")]].astype("float32").mean(1).to_numpy()
        y, dev = oof.label.to_numpy(), np.array([device[f] for f in oof.folder])
        cells, recs, fprs = [], [], []
        for dv in DEVICES:
            pos, neg = s[(dev == dv) & (y == 1)], s[(dev == dv) & (y == 0)]
            r, f = [], []
            for _ in range(reps):
                perm = rng.permutation(len(neg))
                cal, test = neg[perm[:len(neg) // 2]], neg[perm[len(neg) // 2:]]
                thr = np.quantile(cal, q)
                r.append((pos > thr).mean())
                f.append((test > thr).mean())
            recs.append(np.mean(r))
            fprs.append(np.mean(f))
            cells.append(f"{100 * recs[-1]:.0f} / {100 * fprs[-1]:.0f}")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {100 * np.mean(recs):.0f} / {100 * np.mean(fprs):.0f} |")
    lines += ["", "Recorders 2538 and 8391 have only 27 clean recordings each (13 for calibration), so their "
              "figures are rough."]
    lines += calibration_size(device, rng, q, reps)
    Path("outputs/folder-diagnostics/device_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
