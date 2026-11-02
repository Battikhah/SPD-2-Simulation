"""Build markdown tables for the ESP32 sweep report from outputs/esp32-tflm/sweep/leaderboard.csv.

Writes outputs/esp32-tflm/sweep/report_tables.md. All figures are folder-grouped development cross-
validation (test partition never read) with cross-fitted thresholds; with many configurations compared,
the best-looking rows are optimistic.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import esp32_sweep  # noqa: E402

OUT = Path("outputs/esp32-tflm/sweep/report_tables.md")
RULES = ("f2", "f1", "f0.5", "fpr10")


def pct(v):
    return f"{100 * v:.1f}%"


def field_precision(recall, fpr, prevalence):
    return recall * prevalence / max(recall * prevalence + fpr * (1 - prevalence), 1e-12)


def label(r):
    extra = []
    if r.teacher:
        extra.append(f"KD {r.teacher}" + (f" a={r.alpha:g}" if pd.notna(r.alpha) else ""))
    if r.balance_folders:
        extra.append(f"folder-balanced p={r.balance_folders:g}")
    if pd.notna(r.epochs) and r.epochs:
        extra.append(f"{int(r.epochs)} ep")
    if r.seeds and r.seeds != "0-1-2":
        extra.append(f"seeds {r.seeds}")
    return f"{r.arch} pb{r.pos_boost:g} {r.aggregation}" + (f" ({', '.join(extra)})" if extra else "")


def fp_by_folder(best, rule="f2"):
    """Recompute the cross-fitted predictions of `best` and break false alarms down by negative folder."""
    import numpy as np
    from esp32_export import choose_threshold
    from sklearn.model_selection import GroupKFold

    oof = pd.read_csv(Path("outputs/esp32-tflm") / best.directory / "oof_int8.csv")
    base = best.arch + ("@top3of10" if best.aggregation == "top3of10" else "")
    cols = [c for c in oof.columns if c.startswith(base + "|seed")]
    # float32 mean, as in esp32_export: a float64 mean can break score ties differently.
    s, y = oof[cols].astype("float32").mean(1).to_numpy(), oof.label.to_numpy()
    pred = np.zeros(len(y), bool)
    for tr, te in GroupKFold(5).split(y, y, oof.folder):
        pred[te] = s[te] >= choose_threshold(y[tr], s[tr], rule)
    # Independent recomputation must agree with esp32_export's results.csv for the same rule.
    assert int((pred & (y == 0)).sum()) == int(best[f"{rule}_fp"]), "false-alarm count mismatch"
    assert int((pred & (y == 1)).sum()) == int(best[f"{rule}_tp"]), "true-positive count mismatch"
    neg = oof.assign(alarm=pred)[y == 0].groupby("folder").agg(recordings=("alarm", "size"), alarms=("alarm", "sum"))
    neg["share_of_all_false_alarms"] = neg.alarms / max(neg.alarms.sum(), 1)
    out = ["| Clean folder | Recordings | False alarms | Folder FPR | Share of all false alarms |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for f, r in neg.sort_values("alarms", ascending=False).iterrows():
        out.append(f"| {f} | {int(r.recordings)} | {int(r.alarms)} | {pct(r.alarms / r.recordings)} | "
                   f"{pct(r.share_of_all_false_alarms)} |")
    pos = oof.assign(alarm=pred)[y == 1].groupby("folder").agg(recordings=("alarm", "size"), hits=("alarm", "sum"))
    pos["missed"] = pos.recordings - pos.hits
    out += ["", "Infested folders with the most missed recordings (same model and rule):", "",
            "| Infested folder | Recordings | Missed | Folder recall |", "| --- | ---: | ---: | ---: |"]
    for f, r in pos.sort_values("missed", ascending=False).head(6).iterrows():
        out.append(f"| {f} | {int(r.recordings)} | {int(r.missed)} | {pct(r.hits / r.recordings)} |")
    return out


def main():
    board = esp32_sweep.leaderboard()
    int8 = board[board.arithmetic == "int8"].copy()
    int8["config"] = [label(r) for r in int8.itertuples()]
    ens = int8[int8.variant == "ensemble"].sort_values("roc_auc", ascending=False)
    lines = ["## All configurations (int8, 3-seed ensemble unless noted; sorted by ROC-AUC)", "",
             "| # | Configuration | .tflite | Arena | AUC | F2 rule: recall / precision / FPR | "
             "FPR≤10% rule: recall / precision / FPR | Folder 24 score |",
             "| ---: | --- | ---: | ---: | ---: | --- | --- | ---: |"]
    for i, r in enumerate(ens.itertuples(), 1):
        size = f"{r.int8_tflite_bytes / 1024:.1f} KB" if pd.notna(r.int8_tflite_bytes) else "-"
        arena = f"{r.arena_lower_bound_bytes / 1024:.0f} KiB" if pd.notna(r.arena_lower_bound_bytes) else "-"
        lines.append(f"| {i} | {r.config} | {size} | {arena} | {r.roc_auc:.3f} | "
                     f"{pct(r.f2_recall)} / {pct(r.f2_precision)} / {pct(r.f2_fpr)} | "
                     f"{pct(r.fpr10_recall)} / {pct(r.fpr10_precision)} / {pct(r.fpr10_fpr)} | "
                     f"{r.folder24_mean_score:.2f} |")
    lines += ["", "## Seed spread of the top 5 configurations (int8 AUC per seed)", "",
              "| Configuration | Seeds | Per-seed AUC | Ensemble AUC |", "| --- | --- | --- | ---: |"]
    for r in ens.head(5).itertuples():
        per = int8[(int8.directory == r.directory) & (int8.arch == r.arch) & (int8.aggregation == r.aggregation)
                   & int8.variant.str.startswith("seed")]
        lines.append(f"| {r.config} | {r.seeds} | {', '.join(f'{v:.3f}' for v in per.roc_auc)} | {r.roc_auc:.3f} |")
    best = ens.iloc[0]
    rows = board[(board.directory == best.directory) & (board.arch == best.arch)
                 & (board.aggregation == best.aggregation) & (board.variant == "ensemble")]
    lines += ["", f"## Threshold rules for the top configuration: {best.config}", "",
              "| Rule (chosen on the other 4 folds) | Arithmetic | Recall | Precision | FPR | F1 | "
              "Precision at 20% prevalence | at 5% prevalence |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows.to_dict("records"):
        for rule in RULES:
            rec, prec, fpr, f1 = (r[f"{rule}_{m}"] for m in ("recall", "precision", "fpr", "f1"))
            lines.append(f"| {rule} | {r['arithmetic']} | {pct(rec)} | {pct(prec)} | {pct(fpr)} | {f1:.3f} | "
                         f"{pct(field_precision(rec, fpr, 0.20))} | {pct(field_precision(rec, fpr, 0.05))} |")
    lines += ["", f"## Where the false alarms come from ({best.config}, int8, F2 rule)", ""] + fp_by_folder(best)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
