"""Can hand-measured click features separate folders 24/25 from infested recordings, alone or fused with
the CNN? Same folder-grouped 5-fold split and cross-fitted F1 threshold as esp32_export.py; development
data only (needs outputs/folder-diagnostics/recordings.csv from folder_diagnostics.py).

Fusion uses the CNN's out-of-fold scores as a feature (standard stacking). Those scores came from models
that saw the other folds, so the fused figures are slightly optimistic; the per-folder pattern is the point.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esp32_export import crossfit  # noqa: E402

FEATS = ["rms_dbfs", "hp_noise_dbfs", "low_share", "high_share", "events_per_s", "hp_kurtosis", "click_ms",
         "click_centroid_hz", "iei_cv"]
CNN = ("outputs/esp32-tflm/full-stream1s_w16-pb2.0-s345/oof_int8.csv", "stream1s_w16|seed")


def oof_scores(X, y, groups, folds, kind):
    s = np.zeros(len(y))
    for tr, te in folds:
        if kind == "logistic":
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
        else:
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                               class_weight="balanced", random_state=0)
        m.fit(X[tr], y[tr])
        s[te] = m.predict_proba(X[te])[:, 1]
    return s


def main():
    rec = pd.read_csv("outputs/folder-diagnostics/recordings.csv")
    cnn = pd.read_csv(CNN[0])
    assert list(cnn.path) == list(rec.path)
    rec["cnn"] = cnn[[c for c in cnn.columns if c.startswith(CNN[1])]].astype("float32").mean(1).to_numpy()
    rec["cnn_logit"] = np.log(np.clip(rec.cnn, 1e-4, 1 - 1e-4) / (1 - np.clip(rec.cnn, 1e-4, 1 - 1e-4)))
    feats = rec[FEATS].copy()
    feats["hp_kurtosis"] = np.log1p(feats.hp_kurtosis.clip(lower=0))
    feats["events_per_s"] = np.log1p(feats.events_per_s)
    feats = feats.fillna(feats.median())
    y, groups = rec.label.to_numpy(), rec.folder.to_numpy()
    folds = list(GroupKFold(5).split(y, y, groups))
    variants = {
        "CNN alone (stream1s_w16, fresh seeds)": rec.cnn.to_numpy(),
        "click features, logistic": oof_scores(feats.to_numpy(), y, groups, folds, "logistic"),
        "click features, boosted trees": oof_scores(feats.to_numpy(), y, groups, folds, "trees"),
        "CNN + click features, logistic": oof_scores(np.c_[feats.to_numpy(), rec.cnn_logit], y, groups, folds,
                                                     "logistic"),
        "CNN + click features, boosted trees": oof_scores(np.c_[feats.to_numpy(), rec.cnn_logit], y, groups,
                                                          folds, "trees"),
    }
    lines = ["| Scorer | AUC | F1-rule recall / precision / FPR | F1 | folder 24 / 25 / 7 / 8 / 9 flagged |",
             "| --- | ---: | --- | ---: | --- |"]
    for name, s in variants.items():
        r = crossfit(y, s, folds, rules=("f1",))
        from esp32_export import choose_threshold
        pred = np.zeros(len(y), bool)
        for tr, te in folds:
            pred[te] = s[te] >= choose_threshold(y[tr], s[tr], "f1")
        flag = pd.Series(pred).groupby(groups).mean()
        lines.append(f"| {name} | {roc_auc_score(y, s):.3f} | {100 * r['f1_recall']:.1f}% / "
                     f"{100 * r['f1_precision']:.1f}% / {100 * r['f1_fpr']:.1f}% | {r['f1_f1']:.3f} | "
                     + " / ".join(f"{100 * flag[f]:.0f}%" for f in
                                  ("folder_24", "folder_25", "folder_7", "folder_8", "folder_9")) + " |")
    out = Path("outputs/folder-diagnostics/click_feature_cv.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
