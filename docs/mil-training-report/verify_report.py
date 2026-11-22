"""Verify saved development predictions and the generated report, without loading audio or weights."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    evidence = json.loads((HERE / "model_evidence.json").read_text(encoding="utf-8"))
    checked = 0
    for r in evidence["runs"]:
        if not r["metrics"]:
            continue
        p = pd.read_csv(ROOT / r["source_dir"] / "oof_predictions.csv")
        assert p.prob.between(0, 1).all()
        for st in ("1", "K"):
            d = p[p.stride.astype(str) == st]
            for scheme in ("unweighted", "tree", "class"):
                key = d.tree_group if scheme == "tree" else d.label
                w = np.ones(len(d)) if scheme == "unweighted" else 1 / key.map(key.value_counts()).to_numpy()
                y, score = d.label.to_numpy(), d.prob.to_numpy()
                tn, fp, fn, tp = confusion_matrix(y, score > 0.5, sample_weight=w, labels=[0, 1]).ravel()
                rec, spec = tp / (tp + fn), tn / (tn + fp)
                precision = tp / (tp + fp) if tp + fp else np.nan
                f1 = 2 * precision * rec / (precision + rec) if precision + rec else np.nan
                expected = dict(roc_auc=roc_auc_score(y, score, sample_weight=w),
                                pr_auc=average_precision_score(y, score, sample_weight=w),
                                accuracy=(tp + tn) / w.sum(), balanced_accuracy=(rec + spec) / 2,
                                recall=rec, specificity=spec, precision=precision, f1=f1,
                                tp=tp, fn=fn, fp=fp, tn=tn)
                for name, value in expected.items():
                    saved = r["metrics"][f"win/s{st}/{scheme}/{name}"]["value"]
                    assert np.isclose(value, np.nan if saved is None else saved, atol=1e-10, equal_nan=True), (r["run"], name)
                    checked += 1
        for f in r["configs"]:
            c = f["config"]
            assert not set(c["train_groups"]) & set(c["test_groups"]), f["path"]
    for path, digest in evidence["metadata_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path

    # Independently reconstruct the report's daily example using saved predictions only.
    p = pd.read_csv(ROOT / "outputs/mil/experiments/S-skip-K12/seed42/oof_predictions.csv")
    p = p[p.stride.astype(str) == "1"].copy()
    p["day"] = (pd.to_datetime(p.start_time) + (pd.to_datetime(p.end_time) - pd.to_datetime(p.start_time)) / 2).dt.date
    d = p.groupby(["tree_group", "folder", "day", "label"]).prob.mean().reset_index()
    cm = confusion_matrix(d.label, d.prob > 0.5, labels=[0, 1]).ravel().tolist()
    assert cm == [3, 6, 2, 37], cm
    daily_auc = roc_auc_score(d.label, d.prob)
    assert round(daily_auc, 3) == 0.835, daily_auc
    tw = 1 / d.tree_group.map(d.tree_group.value_counts()).to_numpy()
    assert round(roc_auc_score(d.label, d.prob, sample_weight=tw), 3) == 0.850

    for a, b in (("FINAL", "E-AST-lin"), ("FINAL-K12", "A-win-K12")):
        x, y = [pd.read_csv(ROOT / f"outputs/mil/experiments/{n}/seed42/oof_predictions.csv") for n in (a, b)]
        assert len(x) == len(y) and np.array_equal(x.prob, y.prob), (a, b)

    bad_links = []
    for file in HERE.glob("*.md"):
        text = file.read_text(encoding="utf-8")
        assert "\ufffd" not in text
        for dest in re.findall(r"\]\(([^)]+)\)", text):
            if "://" not in dest and not dest.startswith("#"):
                if not (file.parent / dest.split("#")[0]).exists():
                    bad_links.append((file.name, dest))
    assert not bad_links, bad_links
    assert len(evidence["runs"]) == 69
    assert sum(len(r["checkpoint_paths"]) for r in evidence["runs"]) == 526
    print(f"Verified {checked} saved window metrics independently with scikit-learn/confusion counts.")
    print(f"Daily example verified: TN/FP/FN/TP={cm}; AUC={daily_auc:.9f}.")
    print("Configuration hashes, group separation, repeated FINAL predictions, inventory totals and Markdown links verified.")
    print("No audio, features, weights or holdout scoring code loaded. Bootstrap intervals were preserved, not regenerated.")


if __name__ == "__main__":
    main()
