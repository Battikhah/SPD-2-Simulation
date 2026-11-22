"""Improvement rule and the composition of A-win and FINAL (handoff Sections 6-7), from finished results.

Improvement rule (applied per K, against B0 of the same K): a run passes when, for at least one primary metric
(tree-weighted ROC-AUC or tree-weighted balanced accuracy, stride-1 windows), the paired 95% interval of
(run - B0) lies above 0 AND the gain is larger than the seed-to-seed difference |B0-s2 - B0| for that metric,
and neither primary metric is significantly worse (interval entirely below 0). Paired results are cached in
each run directory.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import experiments as X  # noqa: E402
import metrics as M  # noqa: E402


def name_k(base, k):
    return base if k == 24 else f"{base}-K12"


def run_dir(name):
    """The seed directory of a finished run (first seed found), or None."""
    d = C.EXP / name
    for sd in sorted(d.glob("seed*")):
        if (sd / "oof_predictions.csv").exists() and (sd / "summary.csv").exists():
            return sd
    return None


def oof(sd):
    p = pd.read_csv(sd / "oof_predictions.csv")
    return p[p.stride.astype(str).isin(["1", "K"])]


def paired_cached(sd, base_sd, keys=None):
    keys = keys or list(M.PRIMARY)
    cache = sd / f"paired_vs_{base_sd.parent.name}.csv"
    if cache.exists():
        p = pd.read_csv(cache, index_col=0)
        if set(keys) <= set(p.index):
            return p.loc[keys]
    p = M.paired(oof(base_sd), oof(sd), keys=sorted(set(keys) | set(M.PRIMARY) | {
        "win/sK/tree/roc_auc", "win/sK/tree/balanced_accuracy", "tree/s1/mean>0.5/class/balanced_accuracy"}))
    p.to_csv(cache)
    return p.loc[keys]


def seed_noise(k):
    b0, b1 = run_dir(name_k("B0", k)), run_dir(name_k("B0-s2", k))
    if not (b0 and b1):
        return None
    a, b = M.point(oof(b0)), M.point(oof(b1))
    return {m: abs(b[m] - a[m]) for m in M.PRIMARY}


def verdict(name, base_name, k):
    """Improvement-rule verdict for run `name` against `base_name` (both finished)."""
    sd, bd = run_dir(name), run_dir(base_name)
    noise = seed_noise(k)
    if not (sd and bd and noise):
        return dict(available=False)
    p = paired_cached(sd, bd)
    better = [m for m in M.PRIMARY if p.loc[m, "lo"] > 0 and p.loc[m, "diff"] > noise[m]]
    worse = [m for m in M.PRIMARY if p.loc[m, "hi"] < 0]
    return dict(available=True, passed=bool(better) and not worse, better=better, worse=worse,
                diff={m: float(p.loc[m, "diff"]) for m in M.PRIMARY}, lo={m: float(p.loc[m, "lo"]) for m in M.PRIMARY},
                hi={m: float(p.loc[m, "hi"]) for m in M.PRIMARY}, noise=noise)


def passed_augs(k):
    out = {}
    for run, key in X.AUG_RUNS.items():
        v = verdict(name_k(run, k), name_k("B0", k), k)
        out[key] = v.get("passed", False)
    return out


def plan(kind, k):
    """Runs to train for 'A-win' or 'FINAL' at window size k: {'reason': str, 'runs': [(name, extra), ...]}."""
    b0 = name_k("B0", k)
    if kind == "A-win":
        aug = passed_augs(k)
        good = [a for a, ok in aug.items() if ok]
        if not 2 <= len(good) < 5:
            reason = (f"A-win skipped: {len(good)} of 5 single augmentations passed the improvement rule "
                      f"({', '.join(X.AUG_NAMES[a] for a in good) or 'none'}); it runs only when 2-4 pass.")
            _save(kind, k, dict(reason=reason, runs=[]))
            return dict(reason=reason, runs=[])
        changes = {a: True for a in good} | dict(seed=42)
        appr = "MIL v1 + the augmentations that passed: " + ", ".join(X.AUG_NAMES[a] for a in good)
        res = dict(reason=appr, runs=[("A-win", dict(changes=changes, approach=appr))])
        _save(kind, k, res)
        return res
    # FINAL
    notes = []
    cands = {"cnn": None}
    for enc in ("PANNs", "BEATs", "AST"):
        for cl in ("lin", "mlp"):
            name = name_k(f"E-{enc}-{cl}", k)
            v = verdict(name, b0, k)
            if v.get("passed"):
                cands[name] = M.point(oof(run_dir(name)))[M.PRIMARY[0]]
    enc_run = max((n for n in cands if n != "cnn"), key=lambda n: cands[n], default=None)
    changes = dict(seed=42)
    if enc_run:
        enc = X.base_name(enc_run).split("-")[1]
        inp = {"PANNs": "emb_panns", "BEATs": "emb_beats", "AST": "emb_ast"}[enc]
        changes["input"] = inp
        mv = verdict(name_k(f"E-{enc}-mlp", k), name_k(f"E-{enc}-lin", k), k)
        changes["classifier"] = "mlp" if mv.get("passed") else "linear"
        notes.append(f"encoder: frozen {enc} (best passing encoder run {enc_run})")
        notes.append(f"classifier: {changes['classifier']} ({'-mlp beat -lin' if mv.get('passed') else '-mlp did not beat -lin'})")
        notes.append("augmentations: none (pretrained encoder; O6 not run)")
    else:
        notes.append("encoder: trainable CNN (no pretrained-encoder run passed the improvement rule against B0)")
        hv = verdict(name_k("H-mlp", k), b0, k)
        changes["classifier"] = "mlp" if hv.get("passed") else "linear"
        notes.append(f"classifier: {changes['classifier']} (H-mlp {'passed' if hv.get('passed') else 'did not pass'})")
        aug = passed_augs(k)
        for a, ok in aug.items():
            if ok:
                changes[a] = True
        notes.append("augmentations: " + (", ".join(X.AUG_NAMES[a] for a, ok in aug.items() if ok) or "none passed"))
    sv = verdict(name_k("S-skip", k), b0, k)
    changes["skip"] = bool(sv.get("passed"))
    notes.append(f"training windows: {'skip' if changes['skip'] else 'contiguous'} (S-skip "
                 f"{'passed' if sv.get('passed') else 'did not pass'})")
    tv, mv4 = verdict(name_k("T-transformer", k), b0, k), verdict(name_k("M-heads4", k), b0, k)
    changes["pooling"] = "transformer" if tv.get("passed") else "heads4" if mv4.get("passed") else "gated"
    notes.append(f"pooling: {changes['pooling']} (T-transformer {'passed' if tv.get('passed') else 'did not pass'}, "
                 f"M-heads4 {'passed' if mv4.get('passed') else 'did not pass'})")
    nothing = changes == dict(seed=42, classifier="linear", skip=False, pooling="gated")
    appr = "FINAL: " + ("no change passed, so identical to MIL v1 baseline" if nothing else "; ".join(notes))
    runs = [("FINAL", dict(changes=changes, approach=appr)),
            ("FINAL-s2", dict(changes=changes | dict(seed=43), approach=appr + " (seed 43)")),
            ("FINAL-all", dict(changes=changes, approach=appr + " (trained on all cross-validation trees; not evaluated)"))]
    res = dict(reason="; ".join(notes), runs=runs)
    _save(kind, k, res)
    return res


def _save(kind, k, res):
    d = C.EXP / "_plans"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{kind}-K{k}.json").write_text(json.dumps(res, indent=1, default=str))
