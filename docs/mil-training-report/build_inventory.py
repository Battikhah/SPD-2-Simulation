"""Snapshot saved MIL metadata/results; never load audio, features or model weights.

Run from the repository root with .venv/Scripts/python.exe.
"""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXP = ROOT / "outputs/mil/experiments"


def rel(path):
    return path.relative_to(ROOT).as_posix()


def read_metrics(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r[""]: {k: float(r[k]) for k in ("value", "lo", "hi")} for r in csv.DictReader(f)}


def fmt(value):
    return f"{value:.3f}" if value is not None and math.isfinite(value) else "n/a"


def metric(run, key, ci=False):
    m = run["metrics"].get(key, {})
    v = fmt(m.get("value"))
    return f"{v} [{fmt(m.get('lo'))}, {fmt(m.get('hi'))}]" if ci and m else v


def architecture(c):
    inp = c.get("input", "cnn")
    enc = {"cnn": "4-block log-mel CNN (1→32→64→128→128)",
           "emb_ast": "frozen AudioSet AST, 768-d",
           "emb_beats": "frozen BEATs iter3+ AS2M, 768-d",
           "emb_panns": "frozen PANNs CNN14, 2048-d"}.get(inp, f"4-conv CNN, {inp}")
    return f"{enc}; {c.get('pooling', 'gated')} pooling; {c.get('classifier', 'linear')} classifier"


def clean(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    return obj


def main():
    runs, hashes = [], {}
    for rd in sorted(EXP.iterdir()):
        if not rd.is_dir() or rd.name in ("_plans", "smoke"):
            continue
        for sd in sorted(rd.glob("seed*")):
            configs = []
            for p in sorted(sd.glob("fold*/config.json")):
                configs.append({"path": rel(p), "config": json.loads(p.read_text())})
                hashes[rel(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
            cfg = configs[0]["config"] if configs else {}
            models = sorted(sd.glob("fold*/model.pt"))
            preds = sorted(sd.glob("fold*/predictions.csv"))
            all_model = (sd / "foldall/model.pt").exists()
            expected = 1 if all_model else cfg.get("n_folds", 6 if cfg.get("grouping") == "recorder" or rd.name.startswith("R-") else 10)
            summary = sd / "summary.csv"
            metrics = read_metrics(summary)
            if all_model:
                status = "trained on all development groups"
            elif len(preds) == expected and metrics:
                status = "complete fixed split" if cfg.get("grouping") == "single" else "complete CV"
            elif models or preds:
                status = "partial; no complete CV result"
            else:
                status = "started/stopped; no saved completed fold"
            if summary.exists():
                hashes[rel(summary)] = hashlib.sha256(summary.read_bytes()).hexdigest()
            runs.append(dict(run=rd.name, seed=int(sd.name[4:]), status=status,
                             completed_folds=len(preds), expected_folds=expected,
                             checkpoint_paths=[rel(p) for p in models], configs=configs,
                             metrics=metrics, source_dir=rel(sd)))
    plans = {p.name: json.loads(p.read_text()) for p in sorted((EXP / "_plans").glob("*.json"))}
    payload = clean(dict(as_of="2026-11-14", scope="MIL; production experiments excluding smoke",
                        runs=runs, plans=plans, metadata_sha256=hashes))
    (HERE / "model_evidence.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = Counter(r["status"] for r in runs)
    checkpoints = sum(len(r["checkpoint_paths"]) for r in runs)
    lines = ["# Appendix: complete saved MIL model register", "", "Prepared 14 November 2026.", "",
             "This register covers **every non-smoke run directory currently saved under `outputs/mil/experiments/`**, "
             "including each seed and partial attempt. Earlier clip-level campaigns are background to the MIL study, "
             "not enumerated as MIL models. A run name, a seed, a fold checkpoint and a distinct architecture are different counts.", "",
             f"Inventory: **{len(set(r['run'] for r in runs))} run names, {len(runs)} run/seed attempts, "
             f"{checkpoints} saved checkpoints**. Status counts: " + "; ".join(f"{n} {s}" for s, n in counts.items()) + ".", "",
             "## How to use this appendix", "",
             "The tables below report saved **pooled out-of-fold** values for each seed separately. They are not an ensemble. "
             "All threshold metrics use strict probability > 0.5. `TW S1` means stride-1 windows weighted so each tree group "
             "contributes total weight 1; `UW SK` means non-overlapping windows with one vote per window. These answer "
             "different questions. Brackets are saved 95% tree-cluster bootstrap intervals, not seed SDs. n/a means unavailable.", "",
             "Complete per-fold configuration snapshots, split membership, every saved summary metric and metadata hashes "
             "are in [model_evidence.json](model_evidence.json). Original artifact paths below are relative to the repository. "
             "`outputs/` is generally ignored by Git; copying this appendix does not copy weights or predictions.", "",
             "## Common training and architecture definitions", "",
             "- **CNN:** first 20 s of 8 kHz audio; 100–2500 Hz Butterworth bandpass; 64×626 log-mel "
             "(FFT 512, hop 256); per-clip z-score. Four 3×3 conv/BN/ReLU blocks, channels 32/64/128/128, "
             "2×2 pooling after each block, global max to 128 dimensions. 257,666 trainable parameters including MIL head.",
             "- **MFCC:** DCT-II of the same 64-band dB mel, first 13 coefficients; 626 frames mean-pooled into 15 or 30 chunks; "
             "training-fold coefficient normalisation. Four conv/BN/ReLU blocks with only two intermediate pools, global max to 128 dimensions.",
             "- **Frozen encoders:** raw first 20 s, resampled to each encoder's rate; AST takes two 10 s halves at 16 kHz "
             "and max-pools patch tokens (excluding CLS/distillation tokens) and halves. BEATs max-pools tokens at 16 kHz; "
             "PANNs CNN14 uses 32 kHz, final-conv frequency mean/time max. The frozen backbones are **not** counted in saved head parameter counts. "
             "Embedding normalisation uses training-fold mean/SD. These do not use the CNN's 100–2500 Hz filtered front end.",
             "- **Gated pooling:** softmax attention of `w[tanh(Vh) * sigmoid(Uh)]`, attention hidden size 64; weighted sum of clip vectors. "
             "Linear classifier is dropout 0.3 then one output; MLP is pooled dimension→128→1 with ReLU/dropout. "
             "`heads4` concatenates four gated summaries. Transformer adds one 4-head, feedforward-256 layer and learned positions before gated pooling.",
             "- **Default budget:** Adam 3e-4, weight decay 1e-4, cosine schedule, 30 epochs × 64 steps × 6 bags = 11,520 bags/fold; "
             "3 bags/class each step, class-balanced BCE, tree group uniform within class, folder proportional to recording count. "
             "Last epoch retained; no best-epoch selection. Seed 42 unless listed otherwise; GPU AMP on RTX 2070.",
             "- **K** counts recordings, not seconds: K=24 roughly two hours and K=12 roughly one hour at five-minute cadence. "
             "Default training is contiguous with a 36 h span preference/retry limit; skip samples K clips within 48 h, with contiguous fallback. "
             "Short training folders repeat recordings to fill K; short evaluation folders are scored once with their available recordings.",
             "- **A1-lab:** insert selected lab pulse power into 1–3 clips of a healthy bag and assign a synthetic positive label. "
             "This is distinct from L-* lab pseudo-clip training. A2 mixes healthy training-tree noise (p=0.3/clip, −12 to 0 dB); "
             "A3 mines healthy training windows scoring >0.2; A4 applies ±3 dB spectral tilt; A5 masks one 1–8-band and one 1–60-frame span.",
             "- **XS-*:** 30×640×6 = 115,200 bags/fold, skip training, 100 skip draws/evaluation folder. "
             "L-* uses this budget on one fixed development split, optionally adding 483 clean and 967 infested lab pseudo-clips. "
             "No additional independent field trees are created by either procedure.", "",
             "## Every saved run and seed", "",
             "| Run | Seed | K | Split | Completed / required folds | Status | Trainable params |", "| --- | ---: | ---: | --- | --- | --- | ---: |"]
    for r in runs:
        c = r["configs"][0]["config"] if r["configs"] else {}
        lines.append(f"| {r['run']} | {r['seed']} | {c.get('k', 'n/a')} | {c.get('grouping', 'tree' if c else 'n/a')} | "
                     f"{r['completed_folds']}/{r['expected_folds']}" + (" (all-data fit)" if 'all development' in r['status'] else "") +
                     f" | {r['status']} | {c.get('n_params', 'n/a')} |")
    lines += ["", "## Comparable results for every completed run/seed", "",
              "L-* rows are a single fixed split and must not be ranked as though they were 10-fold CV. "
              "R-* rows hold out recorders, with linked tree groups removed from training.", "",
              "| Run / seed | TW S1 AUC [95% CI] | TW S1 balanced accuracy | TW S1 recall | TW S1 specificity | UW SK AUC | UW SK balanced accuracy | UW SK recall | UW SK specificity |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in runs:
        if r["metrics"]:
            keys = ["win/s1/tree/balanced_accuracy", "win/s1/tree/recall", "win/s1/tree/specificity",
                    "win/sK/unweighted/roc_auc", "win/sK/unweighted/balanced_accuracy",
                    "win/sK/unweighted/recall", "win/sK/unweighted/specificity"]
            lines.append(f"| {r['run']} / {r['seed']} | " + metric(r, "win/s1/tree/roc_auc", True) + " | " +
                         " | ".join(metric(r, k) for k in keys) + " |")
    lines += ["", "## Additional non-overlapping-window results", "",
              "All quantities below are unweighted, stride K, threshold >0.5. Fractional values would indicate weights; "
              "these counts should be integers. Recall/specificity/AUC are in the preceding table.", "",
              "| Run / seed | TP | FN | FP | TN | Precision | F1 | Accuracy |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in runs:
        if r["metrics"]:
            keys = ("tp", "fn", "fp", "tn", "precision", "f1", "accuracy")
            lines.append(f"| {r['run']} / {r['seed']} | " +
                         " | ".join(metric(r, "win/sK/unweighted/" + k) for k in keys) + " |")
    lines += ["", "## Exact architecture and training differences by run", ""]
    for name in sorted(set(r["run"] for r in runs)):
        rr = [r for r in runs if r["run"] == name]
        candidates = [r for r in rr if r["configs"]]
        c = candidates[0]["configs"][0]["config"] if candidates else {}
        lines += [f"### {name}", "", f"**Architecture:** {architecture(c)}.", "",
                  f"**Approach:** {c.get('approach') or ('MIL v1 baseline; second seed' if name == 'B0-s2' else 'MIL v1 baseline')}.", "",
                  f"**Recipe:** K={c.get('k', 'n/a')}; {c.get('epochs', 'n/a')} epochs; {c.get('steps', 'n/a')} steps/epoch; "
                  f"{2*c.get('per_class', 3)} bags/step; grouping={c.get('grouping', 'tree')}; "
                  f"skip={c.get('skip', False)}; active options=" +
                  (", ".join(k for k in ("noise", "lab", "tilt", "specaug", "hardneg", "lab_data", "mixstyle", "randeq") if c.get(k)) or "none") +
                  f"; excluded folders={c.get('exclude_folders', [])}.", ""]
        for r in rr:
            lines += [f"- Seed {r['seed']}: {r['status']}; {len(r['checkpoint_paths'])} checkpoint(s); "
                      f"artifact directory `{r['source_dir']}`."]
        if name in ("FINAL", "FINAL-K12"):
            lines += ["", "This is a repeated configuration/seed, not an independent algorithmic gain: " +
                      ("identical to E-AST-lin." if name == "FINAL" else "identical to A-win-K12.")]
        lines += [""]
    lines += ["## All-data models and consumed holdout", "",
              "H1 = FINAL-all (K24, folder 8 included); H2 = H-AST-K12 (K12, included); "
              "H3 = H-AST-noF8 (K24, excluded); H4 = H-AST-noF8-K12 (K12, excluded). "
              "All use frozen AST + gated attention + linear classifier, seed 42 and the default training budget. "
              "FINAL-all-K12 is a CNN + noise/tilt all-data fit; it was **not** one of the four holdout models.", "",
              "| ID | Saved model | UW SK AUC | TW SK AUC | TW SK balanced accuracy | Group verdicts at 0.5 |",
              "| --- | --- | ---: | ---: | ---: | --- |",
              "| H1 | FINAL-all | 0.768 | 0.915 | 0.500 | all four called infested |",
              "| H2 | H-AST-K12 | 0.719 | 0.894 | 0.554 | all four called infested |",
              "| H3 | H-AST-noF8 | 0.498 | 0.782 | 0.500 | all four called healthy |",
              "| H4 | H-AST-noF8-K12 | 0.613 | 0.821 | 0.500 | all four called healthy |", "",
              "Source: [holdout results](../../mil/HOLDOUT_RESULTS.md). Window verdicts vary within H2; "
              "'all four' describes the mean-score group decision only. The holdout has one healthy and three infested "
              "groups, 1,284 recordings, known site/time links to development. Consumed November 11 at 21:51:48 "
              "(marker as recorded). It is no longer an independent test for any pipeline.", "",
              "## Partial attempts, planned methods and non-training analyses", "",
              "- XS-mfcc15 K24 has 3/10 completed folds; NF8-AST-lin-K12 has 6/10. No full-CV metric is assigned to either. "
              "R-B0 seed43 has no saved completed fold. A finished log epoch or a queued command is not a completed CV run.",
              "- Remaining XS queue entries (XS-mfcc30 K24; AST/BEATs/PANNs at K12/K24; log-mel CNN at K12/K24) "
              "have no production run directories at this snapshot. They are planned, not completed models.",
              "- P-base, P-fn, P-pcen, P-mix, P-randeq, P-fn-mix, P-mix-randeq, P-pcen-mix are implemented/planned; "
              "no production checkpoints were found. PHASE2_RESULTS.md contains R-* reference rows only. "
              "Do not claim frequency centring, PCEN, Freq-MixStyle or random EQ improved MIL here.",
              "- Other NF8 variants are defined but have no saved run directories. NO_F8_RESULTS.md removes folder 8 "
              "from scoring **without retraining**; it is not evidence for NF8 retraining.",
              "- A-win K24 was deliberately skipped because zero individual augmentations passed its rule; A-win K12 ran.",
              "- MEM-alarm, tree-day aggregation, day/night slices and folder-8 scoring exclusions reuse saved predictions; "
              "they are not separately trained neural models. See the findings report for results and limitations.",
              "- Smoke outputs under `outputs/mil/experiments/smoke/` are software checks, excluded from scientific model totals. "
              "The original crashed/restarted XS attempt is historical and not counted twice.", "",
              "## Instructions for a receiving AI agent", "",
              "1. Read AGENTS.md and the report before proposing experiments. Preserve user edits and existing artifact paths.",
              "2. Treat the consumed holdout as consumed for **all** pipelines. Never rerun holdout_eval.py or manufacture a fresh test from these groups.",
              "3. Keep architecture, input pipeline, K, grouping, budget, seed, threshold, weighting and decision unit attached to every number.",
              "4. Do not merge the five seed metrics into an ensemble claim; mean/SD over seed-level metrics is not averaging predictions.",
              "5. Do not count identical FINAL/precursor predictions as replication. Do not treat selected CV as an unbiased test.",
              "6. Missing evidence is n/a. Restore ignored artifacts explicitly; a Git clone or this appendix alone does not contain the trained models.",
              "7. No model here establishes species identification, field sensitivity/specificity, or MIL deployment feasibility on ESP32.", ""]
    (HERE / "MODEL_APPENDIX.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(run_names=len(set(r['run'] for r in runs)), attempts=len(runs), checkpoints=checkpoints,
                          status=dict(counts)), indent=2))


if __name__ == "__main__":
    main()
