"""Write mil/RESULTS.md from every finished run (rerun any time; regenerated from the raw prediction CSVs).

    .venv/Scripts/python.exe mil/report.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import decide as D  # noqa: E402
import experiments as X  # noqa: E402
import metrics as M  # noqa: E402

RESULTS = C.MIL / "RESULTS.md"
SHOW = ("roc_auc", "pr_auc", "balanced_accuracy", "recall", "specificity", "precision", "precision_at_10pct",
        "f1", "accuracy")
ORDER = ["B0", "B0-s2"] + [r for r in X.LIST_K24 if r not in ("FINAL",)] + ["FINAL", "FINAL-s2"]


def fmt(r):
    return "n/a" if pd.isna(r.value) else f"{r.value:.3f} [{r.lo:.2f}, {r.hi:.2f}]"


def _span(hours):
    return f"{hours * 60:.0f} min" if hours < 2 else f"{hours:.1f} h"


def finished():
    out = []
    for exp_dir in C.EXP.glob("*"):
        if exp_dir.name in ("smoke",) or exp_dir.name.startswith(("_", "R-", "P-", "L-", "NF8-", "H-")):  # R-*, P-*: recorder folds, own reports
            continue
        sd = D.run_dir(exp_dir.name)
        if sd is not None:
            out.append((exp_dir.name, int(sd.name[4:]), sd))

    def key(t):
        b = X.base_name(t[0])
        return (X.k_of(t[0]) == 12, ORDER.index(b) if b in ORDER else 99, t[0])

    return sorted(out, key=key)


def approach_of(name, sd):
    cfg = json.loads((sd / "fold0" / "config.json").read_text())
    return cfg.get("approach") or X.approach(name) or name


def data_section(index, fold_of):
    seqs = C.folder_sequences(index)
    g = index.groupby("group").agg(label=("label", "first"), recordings=("path", "size"),
                                   folders=("folder", lambda f: ", ".join(sorted(set(f), key=lambda x: int(x[7:])))))
    g["fold"] = g.index.map(fold_of)
    lines = ["## Data used in cross-validation", ""]
    for lab, name in ((0, "healthy"), (1, "infested")):
        d = g[g.label == lab]
        lines.append(f"- **{name.capitalize()}:** {len(d)} tree groups, {d.recordings.sum()} recordings; "
                     f"{(d.recordings >= 24).sum()} groups have at least 24 recordings in total, "
                     f"{(d.recordings >= 12).sum()} at least 12.")
    lines += [f"- **Folds:** `mil/folds.json`; folds without a healthy group: "
              f"{sorted(set(range(10)) - set(g[g.label == 0].fold))}.",
              "- **Windows are built per folder** (one device deployment); a folder with fewer than K recordings "
              "gives one window of all its recordings.", "",
              "| Group | Label | Folders | Recordings | Fold |", "| --- | --- | --- | ---: | ---: |"]
    for grp, r in g.iterrows():
        lines.append(f"| {grp} | {'infested' if r.label else 'healthy'} | {r.folders} | {r.recordings} | {r.fold} |")
    lines += ["", "**Time covered by one window, per folder** (stride-1 windows; folders shorter than K show their "
              "whole span):", "",
              "| Folder | Group | Label | Recordings | Windows K=24 | Median span K=24 | Max span K=24 | Windows K=12 | "
              "Median span K=12 |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in sorted(seqs, key=lambda s: int(s["folder"][7:])):
        n, h = len(s["rows"]), s["hours"]
        sp = {k: np.array([h[min(st + k, n) - 1] - h[st] for st in C.window_starts(n, 1, k)]) for k in (24, 12)}
        lines.append(f"| {s['folder']} | {s['group']} | {'infested' if s['label'] else 'healthy'} | {n} | "
                     f"{len(sp[24])} | {_span(np.median(sp[24]))} | {_span(sp[24].max())} | {len(sp[12])} | "
                     f"{_span(np.median(sp[12]))} |")
    return lines


def verdict_txt(name):
    k = X.k_of(name)
    b = X.base_name(name)
    if b in ("B0", "B0-s2", "B0-site25"):
        return "reference" if b != "B0-site25" else "sensitivity check"
    v = D.verdict(name, D.name_k("B0", k), k)
    if not v.get("available"):
        return "n/a"
    if v["passed"]:
        return "**improvement** (" + ", ".join(m.split("/")[-1] for m in v["better"]) + ")"
    if v["worse"]:
        return "worse (" + ", ".join(m.split("/")[-1] for m in v["worse"]) + ")"
    return "no detectable difference"


def run_section(name, seed, sd, index):
    s = pd.read_csv(sd / "summary.csv", index_col=0)
    pred = pd.read_csv(sd / "oof_predictions.csv")
    cfg = json.loads((sd / "fold0" / "config.json").read_text())
    logs = pd.concat([pd.read_csv(sd / f"fold{k}" / "train_log.csv").assign(fold=k) for k in range(len(list(sd.glob("fold[0-9]*"))))])
    last = logs[logs.epoch == logs.epoch.max()]
    timing = [json.loads((sd / f"fold{k}" / "timing.json").read_text()) for k in range(len(list(sd.glob("fold[0-9]*"))))]
    k = X.k_of(name)
    n_tree = pred.groupby("label").tree_group.nunique()
    st = pred.stride.astype(str)
    loss_txt = ", ".join(f"{t} {last[f'loss_{t}'].mean():.4f}" for t in ("healthy", "hard_healthy", "infested",
                                                                          "synthetic", ) if f"loss_{t}" in last
                         and last[f"loss_{t}"].notna().any())
    if "loss_healthy" not in last:  # B0 / B0-s2 logs from the first runner
        loss_txt = ", ".join(f"{c[5:]} {last[c].mean():.4f}" for c in ("loss_healthy", "loss_infested") if c in last)
    changes = {key: cfg.get(key) for key in ("input", "classifier", "pooling", "noise", "lab", "tilt", "specaug",
                                             "hardneg", "skip", "grouping") if cfg.get(key, X.BASE.get(key)) != X.BASE.get(key)}
    lines = [f"## {name}: {approach_of(name, sd)}", "",
             f"- **Settings that differ from B0:** {', '.join(f'{a}={b}' for a, b in changes.items()) or 'none'}; "
             f"K={cfg.get('k', 24)}; seed {seed}; {cfg['epochs']} epochs x {cfg['steps']} steps; mixed precision "
             f"{cfg['amp']}.",
             f"- **Final-epoch training loss by bag type (mean over folds):** {loss_txt}.",
             f"- **Time:** {sum(t['fold_total_s'] for t in timing) / 60:.1f} min for 10 folds; peak GPU memory "
             f"{max(t['peak_mem_mb'] for t in timing):.0f} MB; waiting for training batches "
             f"{sum(t['data_wait_s'] for t in timing):.0f} s vs GPU {sum(t['gpu_s'] for t in timing):.0f} s.",
             f"- **Evaluated on:** {n_tree.get(0, 0)} healthy and {n_tree.get(1, 0)} infested trees; stride 1: "
             f"{(st == '1').sum()} windows, stride K: {(st == 'K').sum()} windows.",
             f"- **Improvement rule vs {D.name_k('B0', k)}:** {verdict_txt(name)}.", ""]
    for stv in ("1", "K"):
        lines += [f"**Window level, stride {stv}** (threshold 0.5, 95% tree bootstrap):", "",
                  "| Weighting | " + " | ".join(SHOW) + " |", "| --- |" + " ---: |" * len(SHOW)]
        for w in M.WEIGHTINGS:
            lines.append(f"| {w} | " + " | ".join(fmt(s.loc[f"win/s{stv}/{w}/{m}"]) for m in SHOW) + " |")
        cm = []
        for w in M.WEIGHTINGS:
            v = {q: s.loc[f"win/s{stv}/{w}/{q}"].value for q in ("tp", "fn", "fp", "tn")}
            cm.append(f"{w}: TP {v['tp']:.3g}, FN {v['fn']:.3g}, FP {v['fp']:.3g}, TN {v['tn']:.3g}")
        lines += ["", "Weighted confusion matrices: " + "; ".join(cm) + ".", ""]
    lines += ["**Tree level** (from stride-1 windows):", "",
              "| Rule | Weighting | " + " | ".join(SHOW) + " |", "| --- | --- |" + " ---: |" * len(SHOW)]
    for rule in M.RULES:
        for w in ("unweighted", "class"):
            lines.append(f"| {rule} | {w} | " + " | ".join(fmt(s.loc[f"tree/s1/{rule}/{w}/{m}"]) for m in SHOW) + " |")
    lines.append("")
    base = D.name_k("B0", k)
    bd = D.run_dir(base)
    if bd is not None and X.base_name(name) != "B0" and bd != sd:
        keys = list(M.PRIMARY) + ["win/sK/tree/roc_auc", "win/sK/tree/balanced_accuracy",
                                  "tree/s1/mean>0.5/class/balanced_accuracy"]
        p = D.paired_cached(sd, bd, keys)
        lines += [f"**Paired difference vs {base}** ({name} minus {base}, same 1,000 tree draws):", "",
                  f"| Metric | {base} | {name} | Difference [95%] | Share of draws better |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for key, r in p.iterrows():
            lines.append(f"| {key} | {r.a:.3f} | {r.b:.3f} | {r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] | "
                         f"{r.share_b_better:.2f} |")
        lines.append("")
    if (st == "skip").any():
        sk = pred[st == "skip"]
        m = M.binary_metrics(sk.label.to_numpy(), sk.prob.to_numpy(),
                             M.window_weights(sk.tree_group.to_numpy(), sk.label.to_numpy(), "tree"))
        lines += [f"**Secondary: skip windows on held-out trees** (robustness to missed uploads; {len(sk)} windows): "
                  f"tree-weighted ROC-AUC {m['roc_auc']:.3f}, balanced accuracy {m['balanced_accuracy']:.3f}, "
                  f"recall {m['recall']:.3f}, specificity {m['specificity']:.3f}.", ""]
    s1 = pred[st == "1"]
    pt = s1.groupby("tree_group").agg(label=("label", "first"), windows=("prob", "size"), mean_prob=("prob", "mean"),
                                      flagged=("prob", lambda p: (p > 0.5).mean()))
    rec = index.groupby("group").agg(recordings=("path", "size"),
                                     folders=("folder", lambda f: ", ".join(sorted(set(f), key=lambda x: int(x[7:])))))
    pt = pt.join(rec)
    lines += ["**Per tree** (stride-1 windows):", "",
              "| Group | Folders | Label | Recordings | Windows | Mean prob | Share flagged |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for grp, r in pt.iterrows():
        lines.append(f"| {grp} | {r.folders} | {'infested' if r.label else 'healthy'} | {r.recordings} | {r.windows} | "
                     f"{r.mean_prob:.3f} | {r.flagged:.2f} |")
    lines += ["", "Folder 35 is excluded from all runs (see the notes above), so it has no row.", ""]
    if (sd / "mem_alarm.csv").exists():
        mem = pd.read_csv(sd / "mem_alarm.csv")
        lines += ["**MEM-alarm** (leaky-max alarm per tree on these predictions, no retraining):", "",
                  "| Decay | Rule | Weighting | Balanced accuracy | Recall | Specificity | ROC-AUC |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
        for (dec, rule, w), d in mem.groupby(["decay", "rule", "weighting"], sort=False):
            d = d.set_index("metric")
            cell = [f"{d.loc[m, 'value']:.3f} [{d.loc[m, 'lo']:.2f}, {d.loc[m, 'hi']:.2f}]" if not pd.isna(d.loc[m, "value"])
                    else "n/a" for m in ("balanced_accuracy", "recall", "specificity", "roc_auc")]
            lines.append(f"| {dec} | {rule} | {w} | " + " | ".join(cell) + " |")
        lines.append("")
    return lines


def write_results():
    index = C.load_index()
    fold_of = C.make_folds(index)
    done = finished()
    lines = ["# Tree-level MIL results", "",
             "Generated by `mil/report.py` from the raw out-of-fold prediction CSVs in `outputs/mil/experiments/`. "
             "Every result is labelled with the architecture or approach that produced it. All numbers are "
             "cross-validated on development trees only; the sealed groups (folders 10, 15, 29, 30, 32) were never "
             "loaded. None of this is field validation.", "",
             "## Notes that apply to every result", "",
             "- **Earlier clip-level work, ML-for-Senior reference** (`outputs/reference-ml-for-senior/`): "
             "**file-level split, leaky.** Its high recall must not be compared with these numbers.",
             "- **Earlier clip-level work in this repository** (CUDA campaign, grouped-CV and ESP32 models): split by "
             "component, folder or recorder, not by file. But those runs trained on folder 30, now known to be the same "
             "tree as folder 29 (1.7 m apart, recorded at the same time). **Any of their results on the sealed folder 29 "
             "would not be independent.** No such holdout result exists yet.",
             "- **Folder 35 is excluded** (neither healthy nor infested). Correction to handoff finding 6: its "
             "high-scoring windows came from a pool of clips taken from infested trees over 96 days (it contains a copy "
             "of a folder 1 recording), so they may contain real borer activity. They are not clean evidence of false "
             "alarms on healthy trees.",
             "- **Grouping:** same-tree evidence only (positions under 5.0 m apart, or shared recordings), never across "
             "labels; see `mil/groups.json`. Duplicate recordings were removed.",
             "- **Recording interval is about 5 minutes,** so a 24-recording window usually covers about 2 hours and a "
             "12-recording window about 1 hour.",
             "- **Primary metrics** (fixed in advance): tree-weighted ROC-AUC and tree-weighted balanced accuracy on "
             "stride-1 windows. Intervals are 95% stratified cluster bootstraps over trees (1,000 draws).",
             "- **Improvement rule:** a run improves on B0 (same K) only if, for at least one primary metric, the "
             "paired 95% interval is above 0 and the gain exceeds the B0 vs B0-s2 seed difference, with neither "
             "primary metric significantly worse (`mil/decide.py`).",
             "- **FINAL is chosen on the same cross-validation trees it is scored on,** so its numbers are optimistic. "
             "There is no lockbox; the sealed holdout is not evaluated without the user's explicit decision.", ""]
    lines += data_section(index, fold_of) + [""]
    if done:
        lines += ["## Master table", "",
                  "Stride-1 windows, threshold 0.5. Primary metrics are tree-weighted; recall and specificity are "
                  "unweighted (every window counts once).", "",
                  "| Run | Architecture / approach | K | Seed | Tree-weighted ROC-AUC | Tree-weighted balanced accuracy | "
                  "Window recall (unweighted) | Window specificity (unweighted) | Improvement rule |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for name, seed, sd in done:
            s = pd.read_csv(sd / "summary.csv", index_col=0)
            lines.append(f"| {name} | {approach_of(name, sd)} | {X.k_of(name)} | {seed} | {fmt(s.loc[M.PRIMARY[0]])} | "
                         f"{fmt(s.loc[M.PRIMARY[1]])} | {fmt(s.loc['win/s1/unweighted/recall'])} | "
                         f"{fmt(s.loc['win/s1/unweighted/specificity'])} | {verdict_txt(name)} |")
        lines.append("")
        for k in (24, 12):
            noise = D.seed_noise(k)
            if noise:
                lines.append(f"**Seed noise at K={k}** (|B0-s2 - B0|): tree-weighted ROC-AUC {noise[M.PRIMARY[0]]:.3f}, "
                             f"tree-weighted balanced accuracy {noise[M.PRIMARY[1]]:.3f}.")
        plans = sorted((C.EXP / "_plans").glob("*.json")) if (C.EXP / "_plans").exists() else []
        if plans:
            lines += ["", "## Composition decisions", ""]
            for p in plans:
                lines.append(f"- **{p.stem}:** {json.loads(p.read_text())['reason']}")
        lines.append("")
        for name, seed, sd in done:
            lines += run_section(name, seed, sd, index)
        fa = [d for d in C.EXP.glob("FINAL-all*/seed*/foldall/model.pt")]
        if fa:
            lines += ["## Models trained on all cross-validation trees", "",
                      *[f"- `{p.as_posix()}`: not evaluated (no lockbox; the sealed holdout needs the user's decision)."
                        for p in fa], ""]
    RESULTS.write_text("\n".join(lines), encoding="utf-8")
    return RESULTS


if __name__ == "__main__":
    print(write_results())
