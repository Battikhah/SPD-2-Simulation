"""Local GPU three-track campaign scored by folder-grouped cross-validation on development data only.

Tracks (requested 1 November 2026):
  A  unrestricted: any representation, including pretrained AST embeddings and log-mel CNNs
  B  30x13 MFCC per 1 s window (the notebook's exact `mfcc-8k-30x13-v1` protocol), any model size
  C  ESP32-WROOM32 candidates: smallest models, int8 weight-size estimates (not board measurements)

Only train+val rows of the locked split are read; the `test` partition is never loaded. Whole
workbook folders are held out together (folder is a proxy for tree/session, not a verified ID).
Window scores are averaged per recording, matching the notebook's AGGREGATION='mean'.
Thresholds at each FPR cap are taken from the pooled out-of-fold negatives, so recall@FPR is
mildly optimistic; ROC-AUC/AP are threshold-free. Run with --smoke first.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.fft import dct
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RUN = Path("outputs/real-training-cuda/full/treevibes-984e68c4e99668c8")
ROOT = Path.home() / ".cache/kagglehub/datasets/potamitis/treevibes/versions/1"
AST = Path("outputs/model-improvement/ast_embeddings.npz")
OUT = Path("outputs/grouped-cv-campaign")
CAPS = (0.01, 0.05, 0.10, 0.20)

# Feature code copied verbatim from notebook cell 11 (MFCC_SPEC version mfcc-8k-30x13-v1).
WINDOW = np.hanning(513)[:-1]


def mel_bank(count):
    hz = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + 4000 / 700), count + 2) / 2595) - 1)
    freq = np.arange(257) * 8000 / 512
    return np.maximum(
        0,
        np.minimum(
            (freq[None] - hz[:-2, None]) / (hz[1:-1, None] - hz[:-2, None]),
            (hz[2:, None] - freq[None]) / (hz[2:, None] - hz[1:-1, None]),
        ),
    )


MEL40, MEL64 = mel_bank(40), mel_bank(64)
DCT13 = dct(np.eye(40), type=2, norm="ortho", axis=0)[:13]


def frame_power(audio, hop, frames):
    chunks = np.stack([audio[i * hop : i * hop + 512] for i in range(frames)])
    return abs(np.fft.rfft(chunks * WINDOW, axis=1)) ** 2 / 512


def window_features(audio):
    power = frame_power(audio, 256, 30)
    mfcc = (np.log(np.maximum(power @ MEL40.T, 1e-10)) @ DCT13.T).astype("float32")
    logmel = np.log(np.maximum(frame_power(audio, 128, 59) @ MEL64.T, 1e-10)).astype("float32")
    return mfcc, logmel


def extract(df, cache):
    if cache.exists():
        with np.load(cache, allow_pickle=False) as d:
            if list(d["paths"]) == list(df.path):
                return {k: d[k] for k in d.files}
    mf, lm, rid = [], [], []
    for i, rel in enumerate(df.path):
        x, sr = sf.read(ROOT / rel, dtype="float64", always_2d=True)
        assert sr == 8000 and np.isfinite(x).all()
        x = x.mean(1)
        for chunk in np.pad(x, (0, (-len(x)) % 8000)).reshape(-1, 8000):
            m, lg = window_features(chunk)
            mf.append(m)
            lm.append(lg)
            rid.append(i)
        if i % 250 == 0:
            print(f"features {i}/{len(df)}", flush=True)
    data = dict(mfcc=np.stack(mf), logmel=np.stack(lm), rid=np.array(rid), paths=df.path.to_numpy().astype(str))
    np.savez(cache, **data)
    return data


def record_mean(window_scores, rid, n):
    return np.bincount(rid, window_scores, n) / np.maximum(np.bincount(rid, minlength=n), 1)


def ci(k, n):
    """Clopper-Pearson 95% interval."""
    lo = beta.ppf(0.025, k, n - k + 1) if k else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return [float(lo), float(hi)]


def metrics(y, s):
    neg, pos = np.sort(s[y == 0]), s[y == 1]
    out = dict(roc_auc=float(roc_auc_score(y, s)), average_precision=float(average_precision_score(y, s)))
    for cap in CAPS:
        allowed = int(np.floor(cap * len(neg)))
        thr = neg[len(neg) - allowed - 1] if allowed < len(neg) else -np.inf  # strictly above this
        tp, fp = int((pos > thr).sum()), int((neg > thr).sum())
        out[f"recall@fpr{cap:g}"] = tp / len(pos)
        out[f"recall@fpr{cap:g}_ci"] = ci(tp, len(pos))
        out[f"fpr@cap{cap:g}"] = fp / len(neg)
        out[f"fpr@cap{cap:g}_ci"] = ci(fp, len(neg))
    return out


# ---------------- models ----------------
def sk_record(model_fn, X, y, groups, folds):
    oof = np.zeros(len(y))
    for tr, te in folds:
        oof[te] = model_fn().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof


def torch_window(build, X, wy, wrid, y, folds, n, epochs, device, seed):
    import torch

    oof = np.zeros(n)
    params = None
    for k, (tr, te) in enumerate(folds):
        torch.manual_seed(seed + k)
        trw, tew = np.isin(wrid, tr), np.isin(wrid, te)
        mu, sd = X[trw].mean(), X[trw].std() + 1e-6  # training-fold normalisation only
        net = build().to(device)
        params = sum(p.numel() for p in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
        scaler = torch.amp.GradScaler("cuda")
        # Whole training fold lives on the GPU; batches are indexed there to avoid host-copy stalls.
        xt = torch.tensor((X[trw] - mu) / sd, dtype=torch.float32, device=device).unsqueeze(1)
        yt = torch.tensor(wy[trw], dtype=torch.float32, device=device)
        pw = (yt == 0).sum() / (yt == 1).sum().clamp(min=1)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        g = torch.Generator(device=device).manual_seed(seed + k)
        for _ in range(epochs):
            net.train()
            for idx in torch.randperm(len(xt), generator=g, device=device).split(512):
                xb = xt[idx]
                xb = xb + 0.05 * torch.randn_like(xb)  # training-only noise augmentation
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = lossf(net(xb).squeeze(1).float(), yt[idx])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
        assert torch.isfinite(loss)
        net.eval()
        xe = torch.tensor((X[tew] - mu) / sd, dtype=torch.float32, device=device).unsqueeze(1)
        with torch.no_grad():
            p = torch.cat([torch.sigmoid(net(b).float()).squeeze(1) for b in xe.split(4096)]).cpu().numpy()
        del xt, yt, xe
        torch.cuda.empty_cache()
        oof[te] = record_mean(p, wrid[tew], n)[te]
    return oof, params


def cnn(width, depth, dense=64):
    """Return a builder so every fold starts from freshly initialised weights."""
    import torch.nn as nn

    def build():
        layers, c = [], 1
        for i in range(depth):
            layers += [nn.Conv2d(c, width * 2**i, 3, padding=1), nn.BatchNorm2d(width * 2**i), nn.ReLU()]
            layers += [nn.MaxPool2d(2)] if i < depth - 1 else []
            c = width * 2**i
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, dense), nn.ReLU(), nn.Dropout(0.3),
                   nn.Linear(dense, 1)]
        return nn.Sequential(*layers)

    return build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    import torch

    device = "cuda" if torch.cuda.is_available() else None
    if device is None:
        raise SystemExit("CUDA not available; set PYTHONPATH to outputs/torch-cu121 (this campaign is GPU-only).")
    out = OUT / ("smoke" if a.smoke else "full")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RUN / "splits.csv")
    df = df[df.split.isin(["train", "val"])].reset_index(drop=True)  # final holdout never read
    if a.smoke:
        df = df.groupby(df.path.str.extract(r"(folder_\d+)")[0]).head(8).reset_index(drop=True)
    y = df.label.to_numpy()
    groups = df.path.str.extract(r"(folder_\d+)")[0].to_numpy()
    folds = list(GroupKFold(5).split(y, y, groups))
    for tr, te in folds:
        assert set(groups[tr]).isdisjoint(groups[te])
    feats = extract(df, OUT / ("dev_features_smoke.npz" if a.smoke else "dev_features.npz"))
    rid, n = feats["rid"], len(df)
    wy = y[rid]
    mf = feats["mfcc"]
    # Recording-level summaries from the 30x13 MFCC stream (track B/C classical inputs).
    mstat = np.stack([np.r_[m.mean((0, 1)), m.std((0, 1)), m.min((0, 1)), m.max((0, 1))]
                      for m in np.split(mf, np.flatnonzero(np.diff(rid)) + 1)])
    ast = np.load(AST, allow_pickle=False)
    amap = dict(zip(ast["paths"], ast["features"]))
    A = np.stack([amap[p] for p in df.path])
    epochs = 1 if a.smoke else 12
    hgb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)  # noqa: E731
    logit = lambda C: lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))  # noqa: E731

    seeds = [0] if a.smoke else [0, 1, 2]

    def neural(builder, X):
        return lambda s: torch_window(builder, X, wy, rid, y, folds, n, epochs, device, 100 * s)

    sklearn = [
        ("A", "AST embedding + logistic C=0.1", lambda: (sk_record(logit(0.1), A, y, groups, folds), 768 + 1)),
        ("A", "AST embedding + HistGB", lambda: (sk_record(hgb, A, y, groups, folds), None)),
        ("B", "MFCC stats(52) + HistGB", lambda: (sk_record(hgb, mstat, y, groups, folds), None)),
        ("C", "MFCC stats(52) + logistic C=0.1", lambda: (sk_record(logit(0.1), mstat, y, groups, folds), 53)),
    ]
    networks = [
        ("A", "log-mel 59x64 CNN w32 d3", neural(cnn(32, 3), feats["logmel"])),
        ("B", "MFCC 30x13 CNN w32 d3", neural(cnn(32, 3), mf)),
        ("C", "MFCC 30x13 tiny CNN w4 d2", neural(cnn(4, 2, 8), mf)),
        ("C", "log-mel 59x64 CNN w8 d3", neural(cnn(8, 3, 16), feats["logmel"])),
        ("C", "log-mel 59x64 CNN w4 d3", neural(cnn(4, 3, 8), feats["logmel"])),
    ]
    rows, preds = [], {"path": df.path, "folder": groups, "label": y}

    def record(track, name, oof, params, t0, **extra):
        m = metrics(y, oof) | dict(track=track, model=name, parameters=params,
                                   int8_weight_bytes_estimate=params, seconds=round(time.time() - t0, 1)) | extra
        rows.append(m)
        preds[name] = oof
        print(f"[{track}] {name}: AUC {m['roc_auc']:.3f} recall@5%FPR {m['recall@fpr0.05']:.2%} "
              f"({m['seconds']}s)", flush=True)
        pd.DataFrame(rows).to_json(out / "results.json", orient="records", indent=1)

    for track, name, run in sklearn:
        t0 = time.time()
        record(track, name, *run(), t0, seed="n/a")
    for track, name, run in networks:
        per_seed = []
        for s in seeds:
            t0 = time.time()
            oof, params = run(s)
            per_seed.append(oof)
            record(track, f"{name} seed{s}", oof, params, t0, seed=s)
        if len(seeds) > 1:  # equal-weight seed ensemble; size is len(seeds) x params
            record(track, f"{name} {len(seeds)}-seed ensemble", np.mean(per_seed, 0), params * len(seeds),
                   time.time(), seed="ensemble")
    pd.DataFrame(preds).to_csv(out / "oof_predictions.csv", index=False)
    (out / "run_info.json").write_text(json.dumps(dict(
        smoke=a.smoke, recordings=n, windows=int(len(rid)), positives=int(y.sum()), negatives=int((1 - y).sum()),
        folders=int(len(set(groups))), folds=5, epochs=epochs, device=torch.cuda.get_device_name(),
        torch=torch.__version__, python=platform.python_version(), test_read=False,
        split_lock_sha256_prefix="9228762b2bd9e550",
        note="Folder-grouped OOF on development data; thresholds from pooled OOF negatives; not a final-test result."), indent=1))


if __name__ == "__main__":
    main()
