"""Whole-recording inputs on the same folder-grouped folds (development data only).

Track B: one 30x13 MFCC matrix per *recording*. MFCC frames use the notebook's spec (8 kHz, Hann 512,
hop 256, 40 HTK mel, DCT-II ortho, 13 coefficients) over the entire ~20 s clip; frames are split
into 30 equal time chunks and pooled by mean (as ML-for-Senior's `np.array_split` reference) or max
(keeps short clicks visible). Tracks A/C: full-recording 64-band log-mel at hop 256, fixed to 640
frames (20.48 s; the few shorter clips are wrapped). One example per recording; training reuses the
round-2 trainer (SpecAugment, mixup, OneCycle, pos-weighted BCE) with recording-level batches.
The test partition is never read. Operating points are chosen on pooled OOF scores (optimistic).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grouped_cv_campaign import DCT13, MEL40, MEL64, OUT, ROOT, RUN, cnn, frame_power, metrics, sk_record  # noqa: E402
from grouped_cv_round2 import confound_check, train, uncapped  # noqa: E402

FRAMES = 640


def clip_features(path):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    assert sr == 8000 and np.isfinite(x).all()
    x = x.mean(1)
    power = frame_power(x, 256, (len(x) - 512) // 256 + 1)
    mfcc = np.log(np.maximum(power @ MEL40.T, 1e-10)) @ DCT13.T
    chunks = np.array_split(mfcc, 30)
    logmel = np.log(np.maximum(power @ MEL64.T, 1e-10))
    logmel = np.pad(logmel, ((0, max(0, FRAMES - len(logmel))), (0, 0)), mode="wrap")[:FRAMES]
    return (np.stack([c.mean(0) for c in chunks]).astype("float32"),
            np.stack([c.max(0) for c in chunks]).astype("float32"), logmel.astype("float32"))


def load(df, cache):
    if cache.exists():
        with np.load(cache, allow_pickle=False) as d:
            if list(d["paths"]) == list(df.path):
                return {k: d[k] for k in d.files}
    rows = []
    for i, p in enumerate(df.path):
        rows.append(clip_features(ROOT / p))
        if i % 500 == 0:
            print(f"clip features {i}/{len(df)}", flush=True)
    m_mean, m_max, lm = map(np.stack, zip(*rows))
    data = dict(mfcc_mean=m_mean, mfcc_max=m_max, logmel=lm, paths=df.path.to_numpy().astype(str))
    np.savez(cache, **data)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; set PYTHONPATH to outputs/torch-cu121 (GPU-only campaign).")
    out = OUT / ("fullclip-smoke" if a.smoke else "fullclip")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RUN / "splits.csv")
    df = df[df.split.isin(["train", "val"])].reset_index(drop=True)  # final holdout never read
    if a.smoke:
        df = df.groupby(df.path.str.extract(r"(folder_\d+)")[0]).head(8).reset_index(drop=True)
    y = df.label.to_numpy()
    groups = df.path.str.extract(r"(folder_\d+)")[0].to_numpy()
    folds = list(GroupKFold(5).split(y, y, groups))  # identical folds to rounds 1-2
    f = load(df, OUT / ("clip_features_smoke.npz" if a.smoke else "clip_features.npz"))
    n, rid = len(df), np.arange(len(df))
    assert f["mfcc_mean"].shape[1:] == (30, 13) and f["logmel"].shape[1:] == (FRAMES, 64)
    cmvn = lambda X: X - X.mean(axis=1, keepdims=True)  # noqa: E731
    base = pd.DataFrame(dict(path=df.path, folder=groups, label=y)).merge(
        pd.read_csv(RUN / "manifest.csv")[["path", "device_id"]], on="path", how="left")
    epochs = 1 if a.smoke else 40
    seeds = [0] if a.smoke else [0, 1, 2]
    hgb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)  # noqa: E731
    logit = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))  # noqa: E731
    flat = lambda X: X.reshape(len(X), -1)  # noqa: E731
    classical = [
        ("B", "clip MFCC 30x13 mean-pool + logistic", lambda: (sk_record(logit, flat(f["mfcc_mean"]), y, groups, folds), 391)),
        ("B", "clip MFCC 30x13 mean-pool + HistGB", lambda: (sk_record(hgb, flat(f["mfcc_mean"]), y, groups, folds), None)),
        ("B", "clip MFCC 30x13 max-pool + HistGB", lambda: (sk_record(hgb, flat(f["mfcc_max"]), y, groups, folds), None)),
    ]
    networks = [  # (track, name, builder, X, batch)
        ("B", "clip MFCC 30x13 mean-pool CNN w32", cnn(32, 3), f["mfcc_mean"], 64),
        ("B", "clip MFCC 30x13 max-pool CNN w32", cnn(32, 3), f["mfcc_max"], 64),
        ("B", "clip MFCC 30x13 mean-pool CMVN CNN w32", cnn(32, 3), cmvn(f["mfcc_mean"]), 64),
        ("A", "clip log-mel 640x64 CNN w32 d4", cnn(32, 4), f["logmel"], 32),
        ("A", "clip log-mel 640x64 CMVN CNN w32 d4", cnn(32, 4), cmvn(f["logmel"]), 32),
        ("C", "clip log-mel 640x64 CMVN CNN w8 d4", cnn(8, 4, 16), cmvn(f["logmel"]), 32),
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

    for track, name, run in classical:
        t0 = time.time()
        s, params = run()
        record(track, name, "single", s, params, time.time() - t0)
    for track, name, build, X, batch in networks:
        per_seed = []
        for sd in seeds:
            t0 = time.time()
            s, params = train(build, X, y, rid, folds, n, epochs, "cuda", 100 * sd, True, True, batch=batch)
            per_seed.append(s)
            record(track, name, f"seed{sd}", s, params, time.time() - t0)
        if len(seeds) > 1:
            record(track, name, f"{len(seeds)}-seed ensemble", np.mean(per_seed, 0), params * len(seeds), 0)
    preds.to_csv(out / "oof_predictions.csv", index=False)
    (out / "run_info.json").write_text(json.dumps(dict(
        smoke=a.smoke, recordings=n, epochs=epochs, seeds=seeds, frames=FRAMES, test_read=False,
        device=torch.cuda.get_device_name(), torch=torch.__version__,
        note="Whole-recording inputs; folder-grouped OOF on development data; operating points on pooled OOF."),
        indent=1))


if __name__ == "__main__":
    main()
