"""10-second log-mel inputs for tracks A (unrestricted) and C (ESP32-sized), same folder-grouped folds.

Each ~20 s recording's full 640x64 log-mel (hop 256, 8 kHz; cached by `grouped_cv_fullclip.py`) is cut
into two consecutive 320-frame segments (10.24 s each). Every segment is one model input; recording
score = mean of its two segment probabilities. No per-segment CMVN (it hurt in earlier rounds).
Development data only; the test partition is never read. Balanced point chosen on pooled OOF (optimistic).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grouped_cv_campaign import OUT, RUN, cnn, metrics  # noqa: E402
from grouped_cv_fullclip import load  # noqa: E402
from grouped_cv_round2 import confound_check, train, uncapped  # noqa: E402

SEGMENT = 320  # frames x 256 hop / 8 kHz = 10.24 s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; set PYTHONPATH to outputs/torch-cu121 (GPU-only campaign).")
    out = OUT / ("10s-smoke" if a.smoke else "10s")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RUN / "splits.csv")
    df = df[df.split.isin(["train", "val"])].reset_index(drop=True)  # final holdout never read
    if a.smoke:
        df = df.groupby(df.path.str.extract(r"(folder_\d+)")[0]).head(8).reset_index(drop=True)
    y = df.label.to_numpy()
    groups = df.path.str.extract(r"(folder_\d+)")[0].to_numpy()
    folds = list(GroupKFold(5).split(y, y, groups))  # identical folds to earlier rounds
    logmel = load(df, OUT / ("clip_features_smoke.npz" if a.smoke else "clip_features.npz"))["logmel"]
    n = len(df)
    segs = logmel.reshape(n, 2, SEGMENT, 64).reshape(2 * n, SEGMENT, 64)
    rid = np.repeat(np.arange(n), 2)
    base = pd.DataFrame(dict(path=df.path, folder=groups, label=y)).merge(
        pd.read_csv(RUN / "manifest.csv")[["path", "device_id"]], on="path", how="left")
    epochs = 1 if a.smoke else 40
    seeds = [0] if a.smoke else [0, 1, 2]
    networks = [  # (track, name, builder, batch)
        ("A", "10 s log-mel 320x64 CNN w32 d4", cnn(32, 4), 64),
        ("C", "10 s log-mel 320x64 CNN w8 d4", cnn(8, 4, 16), 64),
        ("C", "10 s log-mel 320x64 CNN w4 d4", cnn(4, 4, 8), 64),
    ]
    rows, preds = [], base.copy()

    def record(track, name, variant, s, params, seconds):
        col = f"{name} | {variant}"
        preds[col] = s
        m = dict(track=track, model=name, variant=variant, parameters=params, seconds=round(seconds, 1),
                 int8_weight_bytes_estimate=params)
        m |= metrics(y, s) | uncapped(y, s) | confound_check(preds, col)
        rows.append(m)
        print(f"[{track}] {name} {variant}: AUC {m['roc_auc']:.3f} | balanced recall {m['youden_recall']:.1%} "
              f"FPR {m['youden_fpr']:.1%} acc {m['youden_accuracy']:.1%} | recall@5%FPR {m['recall@fpr0.05']:.1%} "
              f"| f24 {m['folder_24_mean_score']:.2f} f25 {m['folder_25_mean_score']:.2f} ({seconds:.0f}s)", flush=True)
        pd.DataFrame(rows).to_json(out / "results.json", orient="records", indent=1)

    for track, name, build, batch in networks:
        per_seed = []
        for sd in seeds:
            t0 = time.time()
            seg_p, params = train(build, segs, y[rid], rid, folds, n, epochs, "cuda", 100 * sd, True, True,
                                  batch=batch)
            s = np.bincount(rid, seg_p, n) / 2
            per_seed.append(s)
            record(track, name, f"seed{sd}", s, params, time.time() - t0)
        if len(seeds) > 1:
            record(track, name, f"{len(seeds)}-seed ensemble", np.mean(per_seed, 0), params * len(seeds), 0)
    preds.to_csv(out / "oof_predictions.csv", index=False)
    (out / "run_info.json").write_text(json.dumps(dict(
        smoke=a.smoke, recordings=n, segments=int(len(rid)), segment_seconds=10.24, epochs=epochs, seeds=seeds,
        test_read=False, device=torch.cuda.get_device_name(), torch=torch.__version__,
        note="10.24 s log-mel segments, two per recording, mean-pooled; folder-grouped OOF on development data."),
        indent=1))


if __name__ == "__main__":
    main()
