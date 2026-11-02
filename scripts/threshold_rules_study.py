"""Which way of choosing the F1 threshold transfers best to an unseen folder? Development data only.

Every rule sees only the four training folds' out-of-fold scores (as in esp32_export.crossfit) and is
scored on the held-out fold. Rules:
- argmax: the F1-maximising threshold (current method),
- smoothed: argmax of F1 after a moving average over 101 neighbouring candidate thresholds,
- recall95 / recall97: the threshold giving 95% / 97% recall on the training folds,
- folder-bootstrap: median argmax-F1 threshold over 200 resamples of whole training folders.
Evaluated on every int8 3-seed mean-aggregation ensemble in the sweep; seed-replication pairs show how
much each rule's F1 moves between seed sets. Writes outputs/folder-diagnostics/threshold_rules.md.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esp32_export import choose_threshold  # noqa: E402

ROOT = Path("outputs/esp32-tflm")
PAIRS = [("full-stream1s_w16-pb2.0", "full-stream1s_w16-pb2.0-s345", "stream1s_w16"),
         ("full-stream1s_ds-pb2.0", "full-stream1s_ds-pb2.0-s345", "stream1s_ds"),
         ("full-stream1s_ds-pb1.0", "full-stream1s_ds-pb1.0-s345", "stream1s_ds"),
         ("full-stream1s_w8-pb1.0-bal0.5", "full-stream1s_w8-pb1.0-bal0.5-s345", "stream1s_w8"),
         ("full-stream1s_w8_d4-pb2.0-kd", "full-stream1s_w8_d4-pb2.0-kd-s345", "stream1s_w8_d4")]


def f1_curve(y, s):
    order = np.argsort(-s, kind="mergesort")
    ss, ys = s[order], y[order]
    last = np.r_[ss[1:] != ss[:-1], True]
    tp, fp = np.cumsum(ys)[last], np.cumsum(1 - ys)[last]
    return ss[last], 2 * tp / (tp + fp + ys.sum()), tp / max(ys.sum(), 1)


def rule_threshold(y, s, g, rule, rng):
    if rule == "argmax":
        return choose_threshold(y, s, "f1")
    thr, f1, rec = f1_curve(y, s)
    if rule == "smoothed":
        return thr[int(np.argmax(np.convolve(f1, np.ones(101) / 101, mode="same")))]
    if rule.startswith("recall"):
        return thr[int(np.searchsorted(rec, int(rule[6:]) / 100))]
    folders = np.unique(g)
    picks = []
    for _ in range(200):
        idx = np.concatenate([np.flatnonzero(g == f) for f in rng.choice(folders, len(folders))])
        if 0 < y[idx].sum() < len(idx):
            picks.append(choose_threshold(y[idx], s[idx], "f1"))
    return float(np.median(picks))


def evaluate(y, s, g, folds, rule):
    rng = np.random.default_rng(0)
    pred = np.zeros(len(y), bool)
    for tr, te in folds:
        pred[te] = s[te] >= rule_threshold(y[tr], s[tr], g[tr], rule, rng)
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    rec, prec = tp / (y == 1).sum(), tp / max(tp + fp, 1)
    return dict(recall=rec, precision=prec, fpr=fp / (y == 0).sum(), f1=2 * prec * rec / max(prec + rec, 1e-12))


def ensemble(directory, arch):
    oof = pd.read_csv(ROOT / directory / "oof_int8.csv")
    cols = [c for c in oof.columns if c.startswith(arch + "|seed")]
    return oof.label.to_numpy(), oof[cols].astype("float32").mean(1).to_numpy(), oof.folder.to_numpy()


def run_folds(d, y, g):
    """The run's own folds: saved recorder folds when present, else the usual folder-grouped 5 folds."""
    info = d / "train_info.json"
    fold_of = json.loads(info.read_text()).get("fold_of_recording") if info.exists() else None
    if fold_of:
        fold_of = np.array(fold_of)
        return [(np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)) for k in range(fold_of.max() + 1)]
    return list(GroupKFold(5).split(y, y, g))


def main():
    rules =["argmax", "smoothed", "recall95", "recall97", "bootstrap"]
    rows = []
    for d in sorted(ROOT.glob("full*")):
        if not (d / "oof_int8.csv").exists():
            continue
        oof = pd.read_csv(d / "oof_int8.csv", nrows=1)
        for arch in sorted({c.split("|")[0] for c in oof.columns if "|seed" in c and "@" not in c}):
            y, s, g = ensemble(d.name, arch)
            folds = run_folds(d, y, g)
            for rule in rules:
                rows.append(dict(directory=d.name, arch=arch, rule=rule) | evaluate(y, s, g, folds, rule))
    r = pd.DataFrame(rows)
    r["protocol"] = np.where(r.directory.str.contains("-lorO"), "recorder", "folder")
    window = r[r.arch.str.startswith("stream1s") & r.protocol.eq("folder")]
    lines =["## Threshold rules across all 1 s window ensembles (held-out folds)", "",
             "| Rule | Configs | Mean F1 | Median F1 | Worst F1 | Mean recall | Mean precision | Mean FPR |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for rule in rules:
        t = window[window.rule == rule]
        lines.append(f"| {rule} | {len(t)} | {t.f1.mean():.3f} | {t.f1.median():.3f} | {t.f1.min():.3f} | "
                     f"{100 * t.recall.mean():.1f}% | {100 * t.precision.mean():.1f}% | {100 * t.fpr.mean():.1f}% |")
    lines += ["", "## Seed replication: F1 on selection seeds → fresh seeds", "",
              "| Configuration | " + " | ".join(rules) + " |", "| --- |" + " --- |" * len(rules)]
    gaps = {rule: [] for rule in rules}
    for a, b, arch in PAIRS:
        cells = []
        for rule in rules:
            fa = r[(r.directory == a) & (r.arch == arch) & (r.rule == rule)].f1.iloc[0]
            fb = r[(r.directory == b) & (r.arch == arch) & (r.rule == rule)].f1.iloc[0]
            gaps[rule].append(abs(fb - fa))
            cells.append(f"{fa:.3f} → {fb:.3f}")
        lines.append(f"| {a.removeprefix('full-')} | " + " | ".join(cells) + " |")
    lines.append("| mean absolute change | " + " | ".join(f"{np.mean(gaps[k]):.3f}" for k in rules) + " |")
    rec = r[r.protocol.eq("recorder")]
    if len(rec):
        lines += ["", "## Recorder-held-out runs (threshold chosen on the other recorders)", "",
                  "| Run | " + " | ".join(rules) + " |", "| --- |" + " --- |" * len(rules)]
        for (d, arch), t in rec.groupby(["directory", "arch"]):
            cells = [f"F1 {v.f1:.3f} ({100 * v.recall:.0f}/{100 * v.precision:.0f}/{100 * v.fpr:.0f})"
                     for v in t.set_index("rule").loc[rules].itertuples()]
            lines.append(f"| {d.removeprefix('full-')} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("Cells: F1 (recall / precision / FPR, %).")
    r.to_csv("outputs/folder-diagnostics/threshold_rules.csv", index=False)
    Path("outputs/folder-diagnostics/threshold_rules.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
