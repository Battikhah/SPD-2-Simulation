"""Round 2 of the folder-grouped GPU campaign: stronger training, channel compensation, no FPR cap.

Same data, folds, features and test-set exclusion as `grouped_cv_campaign.py` (reused by import).
Changes: per-window spectral mean normalisation (CMVN over time, removes fixed device/gain
colouring but keeps transient clicks), SpecAugment + mixup, 20 epochs with cosine LR, and both
mean and top-25% window pooling. Each row reports an uncapped balanced operating point (threshold
maximising recall - FPR on pooled out-of-fold scores, mildly optimistic) plus the 0.5 point, and a
confounding check on the only negatives recorded by positive-class devices (folders 24 and 25).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grouped_cv_campaign import OUT, RUN, cnn, extract, metrics  # noqa: E402


def topk_mean(p, rid, n, frac=0.25):
    out = np.zeros(n)
    for r in np.unique(rid):
        v = np.sort(p[rid == r])[::-1]
        out[r] = v[: max(1, int(np.ceil(frac * len(v))))].mean()
    return out


def operating_point(y, s, thr):
    pred = s >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    rec, fpr = tp / (tp + fn), fp / (fp + tn)
    prec = tp / max(tp + fp, 1)
    return dict(threshold=float(thr), tp=tp, fp=fp, fn=fn, tn=tn, recall=rec, fpr=fpr, precision=prec,
                accuracy=(tp + tn) / len(y), balanced_accuracy=(rec + 1 - fpr) / 2,
                f1=2 * prec * rec / max(prec + rec, 1e-12))


def uncapped(y, s):
    f, t, th = roc_curve(y, s)
    best = th[np.argmax(t - f)]
    return {f"youden_{k}": v for k, v in operating_point(y, s, best).items()} | {
        f"p0.5_{k}": v for k, v in operating_point(y, s, 0.5).items()}


def train(build, X, wy, wrid, folds, n, epochs, device, seed, specaug, mixup, batch=512, pos_boost=1.0,
          keep=None, soft=None, alpha=0.5, predict_all=None, unit_groups=None, balance_power=0.0):
    """`pos_boost` scales the class-balance positive weight; `keep` (a list) receives each fold's
    (numpy state dict, mu, sd) for export. Distillation: `soft[k]` holds fold-k teacher probabilities
    for every unit, and the target becomes alpha * label + (1 - alpha) * teacher (class weights still
    follow the hard label). `predict_all` (a list) receives each fold model's probabilities on all units.
    Group balancing: with `unit_groups` and `balance_power` p > 0, each training unit is weighted by
    count(its group)^-p, rescaled so every class keeps its total weight (p = 1: every training folder of
    a class contributes equally; p = 0.5: square-root compromise). Uses training-fold groups only."""
    import torch

    window_p = np.zeros(len(wy))
    params = None
    for k, (tr, te) in enumerate(folds):
        torch.manual_seed(seed + k)
        trw, tew = np.isin(wrid, tr), np.isin(wrid, te)
        mu, sd = X[trw].mean(), X[trw].std() + 1e-6  # training-fold normalisation only
        net = build().to(device)
        params = sum(p.numel() for p in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-2)
        xt = torch.tensor((X[trw] - mu) / sd, dtype=torch.float32, device=device).unsqueeze(1)
        yt = torch.tensor(wy[trw], dtype=torch.float32, device=device)
        tt = yt if soft is None else torch.tensor(alpha * wy[trw] + (1 - alpha) * soft[k][trw],
                                                  dtype=torch.float32, device=device)
        uw = None
        if unit_groups is not None and balance_power > 0:
            g_tr, y_tr = unit_groups[trw], wy[trw]
            _, inverse, counts = np.unique(g_tr, return_inverse=True, return_counts=True)
            raw = counts[inverse].astype(np.float64) ** -balance_power
            for c in (0, 1):
                m = y_tr == c
                if m.any():
                    raw[m] *= m.sum() / raw[m].sum()
            uw = torch.tensor(raw, dtype=torch.float32, device=device)
        steps = epochs * int(np.ceil(len(xt) / batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
        scaler = torch.amp.GradScaler("cuda")
        pw = pos_boost * (yt == 0).sum() / (yt == 1).sum().clamp(min=1)
        g = torch.Generator(device=device).manual_seed(seed + k)
        rng = np.random.default_rng(seed + k)  # mixup coefficients
        T, F = xt.shape[2], xt.shape[3]
        for _ in range(epochs):
            net.train()
            for idx in torch.randperm(len(xt), generator=g, device=device).split(batch):
                xb, yb, tb = xt[idx].clone(), yt[idx], tt[idx]
                if specaug:  # one time mask and one frequency mask per example
                    b = len(xb)
                    t0 = torch.randint(0, T, (b, 1), device=device)
                    tw = torch.randint(0, max(T // 8, 1), (b, 1), device=device)
                    f0 = torch.randint(0, F, (b, 1), device=device)
                    fw = torch.randint(0, max(F // 8, 1), (b, 1), device=device)
                    ar_t, ar_f = torch.arange(T, device=device), torch.arange(F, device=device)
                    tm = (ar_t >= t0) & (ar_t < t0 + tw)
                    fm = (ar_f >= f0) & (ar_f < f0 + fw)
                    xb = xb.masked_fill(tm[:, None, :, None] | fm[:, None, None, :], 0.0)
                xb = xb + 0.05 * torch.randn_like(xb)
                w = torch.where(yb == 1, pw, torch.ones_like(yb))
                if uw is not None:
                    w = w * uw[idx]
                if mixup:
                    lam = float(rng.beta(0.4, 0.4))
                    perm = torch.randperm(len(xb), device=device)
                    xb = lam * xb + (1 - lam) * xb[perm]
                    tb = lam * tb + (1 - lam) * tb[perm]
                    w = lam * w + (1 - lam) * w[perm]
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = net(xb).squeeze(1).float()
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, tb, reduction="none") * w).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
        assert torch.isfinite(loss)
        net.eval()
        xe = torch.tensor((X[tew] - mu) / sd, dtype=torch.float32, device=device).unsqueeze(1)
        with torch.no_grad():
            window_p[tew] = torch.cat([torch.sigmoid(net(b).float()).squeeze(1) for b in xe.split(max(batch * 4, 1))]).cpu().numpy()
        if keep is not None:
            keep.append(({k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}, float(mu), float(sd)))
        if predict_all is not None:
            with torch.no_grad():
                predict_all.append(np.concatenate([
                    torch.sigmoid(net(torch.tensor((X[i:i + 4096] - mu) / sd, dtype=torch.float32,
                                                   device=device).unsqueeze(1)).float()).squeeze(1).cpu().numpy()
                    for i in range(0, len(X), 4096)]))
        del xt, yt, tt, xe
        torch.cuda.empty_cache()
    return window_p, params


def confound_check(o, col):
    out = {}
    for f in ("folder_24", "folder_25"):
        out[f"{f}_mean_score"] = float(o.loc[o.folder == f, col].mean())
    for f, g in o.groupby("device_id"):
        if g.label.nunique() == 2:
            neg = "/".join(sorted(g.loc[g.label == 0, "folder"].unique()))
            out[f"within_device_auc_{neg}"] = float(roc_auc_score(g.label, g[col]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; set PYTHONPATH to outputs/torch-cu121 (GPU-only campaign).")
    device = "cuda"
    out = OUT / ("round2-smoke" if a.smoke else "round2")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RUN / "splits.csv")
    df = df[df.split.isin(["train", "val"])].reset_index(drop=True)  # final holdout never read
    if a.smoke:
        df = df.groupby(df.path.str.extract(r"(folder_\d+)")[0]).head(8).reset_index(drop=True)
    y = df.label.to_numpy()
    groups = df.path.str.extract(r"(folder_\d+)")[0].to_numpy()
    folds = list(GroupKFold(5).split(y, y, groups))  # identical folds to round 1
    feats = extract(df, OUT / ("dev_features_smoke.npz" if a.smoke else "dev_features.npz"))
    rid, n = feats["rid"], len(df)
    wy = y[rid]
    logmel, mfcc = feats["logmel"], feats["mfcc"]
    cmvn = lambda X: X - X.mean(axis=1, keepdims=True)  # noqa: E731  per-window, per-band mean over time
    devices = pd.read_csv(RUN / "manifest.csv")[["path", "device_id"]]
    base = pd.DataFrame(dict(path=df.path, folder=groups, label=y)).merge(devices, on="path", how="left")
    assert len(base) == n
    epochs = 1 if a.smoke else 20
    seeds = [0] if a.smoke else [0, 1, 2]
    lm_c, mf_c = cmvn(logmel), cmvn(mfcc)
    candidates = [
        ("A", "log-mel CNN w32 +SpecAug+mixup", cnn(32, 3), logmel, True, True),
        ("A", "log-mel CNN w32 CMVN +SpecAug+mixup", cnn(32, 3), lm_c, True, True),
        ("B", "MFCC 30x13 CNN w32 CMVN +SpecAug", cnn(32, 3), mf_c, True, False),
        ("C", "log-mel CNN w8 CMVN +SpecAug+mixup", cnn(8, 3, 16), lm_c, True, True),
    ]
    rows, preds = [], base.copy()
    for track, name, build, X, spec, mix in candidates:
        seed_p = []
        for s in seeds:
            t0 = time.time()
            wp, params = train(build, X, wy, rid, folds, n, epochs, device, 100 * s, spec, mix)
            seed_p.append(wp)
            print(f"[{track}] {name} seed{s} trained ({time.time() - t0:.0f}s)", flush=True)
        variants = {f"seed{s}": p for s, p in zip(seeds, seed_p)}
        if len(seeds) > 1:
            variants[f"{len(seeds)}-seed ensemble"] = np.mean(seed_p, 0)
        for vname, wp in variants.items():
            for agg, fn in [("mean", lambda p: np.bincount(rid, p, n) / np.bincount(rid, minlength=n)),
                            ("top25%", lambda p: topk_mean(p, rid, n))]:
                col = f"{name} | {vname} | {agg}"
                s = fn(wp)
                preds[col] = s
                m = dict(track=track, model=name, variant=vname, aggregation=agg, parameters=params,
                         int8_weight_bytes_estimate=params * (len(seeds) if "ensemble" in vname else 1))
                m |= metrics(y, s) | uncapped(y, s) | confound_check(preds, col)
                rows.append(m)
                print(f"   {vname:>16} {agg:>6}: AUC {m['roc_auc']:.3f} | balanced point recall "
                      f"{m['youden_recall']:.1%} FPR {m['youden_fpr']:.1%} acc {m['youden_accuracy']:.1%} | "
                      f"f24 {m['folder_24_mean_score']:.2f} f25 {m['folder_25_mean_score']:.2f}", flush=True)
        pd.DataFrame(rows).to_json(out / "results.json", orient="records", indent=1)
    preds.to_csv(out / "oof_predictions.csv", index=False)
    (out / "run_info.json").write_text(json.dumps(dict(
        smoke=a.smoke, recordings=n, windows=int(len(rid)), epochs=epochs, seeds=seeds, test_read=False,
        device=torch.cuda.get_device_name(), torch=torch.__version__,
        note="Folder-grouped OOF on development data; operating points chosen on pooled OOF (optimistic)."),
        indent=1))


if __name__ == "__main__":
    main()
