"""Resumable 10-fold runner for the tree-level MIL run list, with live monitoring.

    $env:PYTHONPATH = (Resolve-Path outputs/torch-cu121).Path; $env:PYTHONUTF8 = '1'
    .venv/Scripts/python.exe mil/train.py --queue K24            # handoff run list at K = 24
    .venv/Scripts/python.exe mil/train.py --queue K12            # the same list at K = 12
    .venv/Scripts/python.exe mil/train.py --queue A2-noise H-mlp # named runs
    .venv/Scripts/python.exe mil/train.py --smoke A1-lab         # 1 fold, 2 epochs x 4 steps (outputs under smoke/)

Run definitions and approach labels: mil/experiments.py. "A-win" and "FINAL" are resolved from finished
results by mil/decide.py (FINAL expands to FINAL, FINAL-s2 and FINAL-all, the last trained on all
cross-validation trees and not evaluated). Outputs per run/seed/fold under
outputs/mil/experiments/<run>/seed<s>/fold<k>/: model.pt, config.json, predictions.csv (out-of-fold windows,
stride 1 and K; S-skip adds "skip" windows), train_log.csv, timing.json. A fold whose predictions.csv (or, for
FINAL-all, model.pt) exists is skipped. TensorBoard: outputs/mil/tb/. Live status: mil/LIVE_STATUS.md and
mil/live_curves.png. Held-out curves are for watching only: 30 epochs, last epoch kept, nothing selected on them.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import experiments as X  # noqa: E402
import metrics as M  # noqa: E402

STATUS, CURVES = C.MIL / "LIVE_STATUS.md", C.MIL / "live_curves.png"
TYPE_NAMES = {0: "healthy", 1: "hard_healthy", 2: "infested", 3: "synthetic"}
SMOKE = False
# Windows: worker -> main shared-memory mappings accumulate over a worker's lifetime (error 1455, commit exhausted,
# after ~77k steps). Workers are restarted every LOADER_STEPS steps (the whole fold for the 1,920-step runs, so
# their bag streams are unchanged); each later segment seeds its own stream.
LOADER_STEPS = 1920


def run_root(exp, seed):
    return C.EXP / ("smoke" if SMOKE else "") / exp / f"seed{seed}"


def fold_dir(exp, seed, fold):
    return run_root(exp, seed) / f"fold{fold}"


def fold_done(exp, seed, fold):
    d = fold_dir(exp, seed, fold)
    return (d / "model.pt").exists() if fold == "all" else (d / "predictions.csv").exists()


def gpu_util_start():
    try:
        return subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except OSError:
        return None


def gpu_util_read(p):
    if p is None:
        return None
    out, _ = p.communicate()
    try:
        u, m = out.strip().split(",")
        return int(u), int(m)
    except ValueError:
        return None


def embed(model, feat, rows, cfg, torch):
    """Clip embeddings (eval mode) for the given feature rows, in a full-size array indexed by row."""
    out = np.zeros((len(feat), model.out_dim), np.float32)
    rows = np.unique(np.asarray(rows, int))
    with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=cfg["amp"]):
        for i in range(0, len(rows), cfg["embed_batch"]):
            r = rows[i:i + cfg["embed_batch"]]
            x = torch.from_numpy(np.asarray(feat[r])).cuda(non_blocking=True).float()
            if model.is_cnn:
                x = x.unsqueeze(1)
            out[r] = model.encoder(x).float().cpu().numpy()
    return out


def windows_of(s, stride, k, rng=None, n_skip=20):
    """Row-position lists of evaluation windows for one folder sequence: '1', 'K' or 'skip' (n_skip random
    time-ordered windows per folder)."""
    n = len(s["rows"])
    if stride == "skip":  # robustness to missed uploads: n_skip random skip windows per folder (fixed seed)
        if n < k:
            return [np.arange(n)]
        out = []
        for _ in range(n_skip):
            st = int(rng.integers(0, n - k + 1))
            cand = st + np.flatnonzero(s["hours"][st:] - s["hours"][st] <= 48.0)
            out.append(np.sort(rng.choice(cand, k, replace=False)) if len(cand) >= k else np.arange(st, st + k))
        return out
    return [np.arange(st, min(st + k, n)) for st in C.window_starts(n, 1 if stride == "1" else k, k)]


def score_windows(model, emb, seqs, stride, cfg, torch, seed=0):
    k, rec = cfg["k"], []
    rng = np.random.default_rng(seed)
    with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=cfg["amp"]):
        for s in seqs:
            pos = windows_of(s, stride, k, rng, cfg.get("skip_eval_n", 20))
            H = torch.from_numpy(emb[s["rows"]]).cuda()
            probs = []
            for i in range(0, len(pos), 512):
                chunk = pos[i:i + 512]
                W = torch.stack([H[torch.as_tensor(p, device="cuda")] for p in chunk]) if len(set(map(len, chunk))) == 1 \
                    else None
                if W is None:
                    logit = torch.cat([model.pool(H[torch.as_tensor(p, device="cuda")][None])[0] for p in chunk])
                else:
                    logit = model.pool(W)[0]
                probs.append(torch.sigmoid(logit.float()).squeeze(1).cpu().numpy())
            for p, prob in zip(pos, np.concatenate(probs)):
                rec.append(dict(tree_group=s["group"], folder=s["folder"], label=s["label"], stride=stride,
                                window_start=int(p[0]), window_end=int(p[-1]), n_clips=len(p),
                                start_time=pd.Timestamp(s["times"][p[0]]), end_time=pd.Timestamp(s["times"][p[-1]]),
                                prob=float(prob)))
    return pd.DataFrame(rec)


def monitor(df):
    """Monitoring numbers from stride-K windows (watching only)."""
    if df.empty:
        return dict(loss=np.nan, auc=np.nan, balacc=np.nan, recall=np.nan, specificity=np.nan)
    y, p = df.label.to_numpy(), df.prob.to_numpy()
    m = M.binary_metrics(y, p, M.window_weights(df.tree_group.to_numpy(), y, "tree"))
    eps = 1e-6
    loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    return dict(loss=loss, auc=m["roc_auc"], balacc=m["balanced_accuracy"] if len(set(y)) == 2 else np.nan,
                recall=m["recall"], specificity=m["specificity"])


def write_status(state):
    lines = ["# Live training status", "", f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
             f"- **Run:** {state['exp']} (seed {state['seed']}): {state['approach']}",
             f"- **Fold:** {state['fold_txt']}, **epoch:** {state['epoch']}/{state['epochs']}",
             f"- **Latest training loss by bag type:** {state['losses']}",
             f"- **Held-out fold (watching only):** {state['ho']}",
             f"- **Elapsed in fold:** {state['fold_elapsed'] / 60:.1f} min; fold left about "
             f"{state['fold_left'] / 60:.1f} min; this run left about {state['exp_left'] / 60:.0f} min",
             f"- **Queue after this run:** {', '.join(state['queue']) or 'none'}", "",
             "## Last finished run", "", state.get("last", "none yet"), "",
             "TensorBoard: http://localhost:6006 (on this PC). Curves: mil/live_curves.png. Results: mil/RESULTS.md."]
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(STATUS)


def draw_curves(exp, seed, fold, log):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"healthy": "#2f6fb0", "hard_healthy": "#6aa6d8", "infested": "#c4572b", "synthetic": "#e3a33b"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    if fold != "all":
        for kk in range(int(fold)):
            f = fold_dir(exp, seed, kk) / "train_log.csv"
            if f.exists():
                d = pd.read_csv(f)
                for t, c in colors.items():
                    if f"loss_{t}" in d and d[f"loss_{t}"].notna().any():
                        ax[0].plot(d.epoch, d[f"loss_{t}"], color=c, alpha=0.15, lw=1)
                ax[1].plot(d.epoch, d.ho_loss, color="#555555", alpha=0.15, lw=1)
    d = pd.DataFrame(log)
    for t, c in colors.items():
        if f"loss_{t}" in d and d[f"loss_{t}"].notna().any():
            ax[0].plot(d.epoch, d[f"loss_{t}"], color=c, lw=2, label=f"{t} bags")
    ax[0].axhline(0.693, color="#999999", ls=":", lw=1)
    ax[0].set(title=f"{exp} seed {seed} fold {fold}: training loss per bag type", xlabel="epoch", ylabel="BCE")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(d.epoch, d.ho_loss, color="#555555", lw=2, label="held-out BCE (stride K)")
    ax[1].set(title="Held-out fold (watching only)", xlabel="epoch", ylabel="BCE")
    a2 = ax[1].twinx()
    if d.ho_auc.notna().any():
        a2.plot(d.epoch, d.ho_auc, color="#3a8f5c", lw=1.5, ls="--", label="held-out tree-weighted AUC")
    a2.plot(d.epoch, d.train_auc, color="#8a6bb5", lw=1.5, ls="-.", label="training-tree AUC (stride K)")
    a2.set_ylim(0, 1.02)
    a2.set_ylabel("AUC")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="lower left")
    tmp = CURVES.with_name("live_curves.tmp.png")
    fig.savefig(tmp, dpi=90)
    plt.close(fig)
    tmp.replace(CURVES)


def feature_paths(cfg):
    if cfg.get("lab_data"):  # field rows followed by lab pseudo-clip rows (mil/build_lab_features.py)
        base = "z" if cfg["input"] == "cnn" else cfg["input"]
        return C.CACHE / f"{base}_plus.npy", None, None
    if cfg["input"] != "cnn":  # frozen embeddings (emb_*) or whole-clip MFCC maps (mfcc13x*)
        return C.CACHE / f"{cfg['input']}.npy", None, None
    import front

    augment = any(cfg.get(a) for a in ("noise", "lab", "tilt", "specaug", "randeq"))
    return (front.cache_path(cfg.get("front", "z")), C.CACHE / "raw_db.npy" if augment else None,
            C.CACHE / "lab_bank.npy" if cfg["lab"] else None)


def run_fold(exp, seed, fold, cfg, fold_of, seqs_all, queue, done_times, last_summary):
    import torch
    from data import BagStream
    from model import MILModel, n_params
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    out = fold_dir(exp, seed, fold)
    out.mkdir(parents=True, exist_ok=True)
    recorder = None
    if cfg["grouping"] == "recorder":  # tree-safe leave-one-recorder-out (see common.recorder_folds)
        recs, f2r = C.recorder_folds()
        recorder = recs[int(fold)]
        test_s = [s for s in seqs_all if f2r[s["folder"]] == recorder]
        banned = {s["group"] for s in test_s}
        train_s = [s for s in seqs_all if s["group"] not in banned]
        train_g, test_g = sorted({s["group"] for s in train_s}), sorted(banned)
    else:
        groups = sorted(fold_of)
        train_g = groups if fold == "all" else sorted(g for g in groups if fold_of[g] != fold)
        test_g = [] if fold == "all" else sorted(g for g in groups if fold_of[g] == fold)
        train_s = [s for s in seqs_all if s["group"] in train_g]
        test_s = [s for s in seqs_all if s["group"] in test_g]
    if cfg.get("exclude_folders"):  # e.g. H-*-noF8: drop these folders from training (and testing)
        train_s = [s for s in train_s if s["folder"] not in cfg["exclude_folders"]]
        test_s = [s for s in test_s if s["folder"] not in cfg["exclude_folders"]]
    if cfg.get("lab_data"):  # lab pseudo-clips: training only, never in a test fold
        train_s = train_s + C.lab_sequences(len(C.load_index()))
    fold_seed = 10 if fold == "all" else int(fold)
    torch.manual_seed(seed * 1000 + fold_seed)
    torch.backends.cudnn.benchmark = True
    feat_path, raw_path, bank_path = feature_paths(cfg)
    feat = np.load(feat_path, mmap_mode="r")
    hard_path = out / "hard_negatives.json" if cfg["hardneg"] else None
    if hard_path and hard_path.exists():
        hard_path.unlink()
    ds = BagStream(feat_path, train_s, cfg, seed, fold_seed, raw_path, bank_path, hard_path)
    workers = cfg["workers"] if cfg["input"] == "cnn" else 4 if cfg["input"].startswith("mfcc") else 2

    def new_loader():
        return DataLoader(ds, batch_size=None, num_workers=workers, pin_memory=True, persistent_workers=True,
                          prefetch_factor=4)

    loader = new_loader()
    it = iter(loader)
    in_dim = feat.shape[1] if cfg["input"].startswith("emb") else 128
    model = MILModel(cfg["input"], in_dim, cfg["pooling"], cfg["classifier"], cfg["k"],
                     mixstyle=cfg.get("mixstyle", False)).cuda()
    if cfg["input"] != "cnn":  # standardise embeddings / MFCC coefficients with training-fold statistics only
        tr_rows = np.concatenate([s["rows"] for s in train_s])
        x = np.asarray(feat[np.sort(tr_rows)], np.float64)
        if model.is_mfcc:  # per coefficient, over clips and time chunks
            mu, sd = x.mean(axis=(0, 2)), x.std(axis=(0, 2)) + 1e-6
            model.encoder_net.mu.copy_(torch.tensor(mu, dtype=torch.float32).view(1, 1, -1, 1))
            model.encoder_net.sd.copy_(torch.tensor(sd, dtype=torch.float32).view(1, 1, -1, 1))
        else:
            model.in_mu.copy_(torch.tensor(x.mean(0), dtype=torch.float32))
            model.in_sd.copy_(torch.tensor(x.std(0) + 1e-6, dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    writer = SummaryWriter(str(C.TB / ("smoke" if SMOKE else "") / exp / f"seed{seed}" / f"fold{fold}"))
    torch.cuda.reset_peak_memory_stats()
    config = dict(cfg, experiment=exp, seed=seed, fold=fold, recorder=recorder, train_groups=train_g, test_groups=test_g,
                  n_params=n_params(model), device=torch.cuda.get_device_name(), torch=torch.__version__)
    log, t_fold, step_global = [], time.time(), 0
    timing = dict(data_wait_s=0.0, gpu_s=0.0, monitor_eval_s=0.0, util=[])
    pc = cfg["per_class"]
    order = np.stack([np.arange(pc), pc + np.arange(pc)], 1).ravel()
    eval_rows = np.concatenate([s["rows"] for s in train_s + test_s])
    tr_healthy = [s for s in train_s if s["label"] == 0]
    fold_txt = "all trees" if fold == "all" else f"{int(fold) + 1}/10"
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t_ep, by_type, smi = time.time(), {t: [] for t in TYPE_NAMES}, None
        try:
            from tqdm import tqdm
            bar = tqdm(range(cfg["steps"]), desc=f"{exp} s{seed} f{fold_txt} e{epoch}", leave=False, ncols=100,
                       disable=not sys.stderr.isatty())
        except ImportError:
            bar = range(cfg["steps"])
        for step in bar:
            if step_global and step_global % LOADER_STEPS == 0:  # restart workers (see LOADER_STEPS)
                del it, loader
                ds.segment = step_global // LOADER_STEPS
                loader = new_loader()
                it = iter(loader)
            t0 = time.time()
            xb, yb, tb = next(it)
            t1 = time.time()
            xb = xb.cuda(non_blocking=True).float()
            if model.is_cnn:  # log-mel or MFCC map: add the channel axis
                xb = xb.unsqueeze(2)
            yb = yb.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            per_bag = torch.zeros(2 * pc, device="cuda")
            for ch in np.array_split(order, cfg["accum"]):
                ch = torch.as_tensor(ch, device="cuda")
                with torch.autocast("cuda", torch.float16, enabled=cfg["amp"]):
                    logit, _ = model(xb[ch])
                    lb = bce(logit.squeeze(1).float(), yb[ch])
                scaler.scale(lb.sum() / (2 * pc)).backward()
                per_bag[ch] = lb.detach()
            scaler.step(opt)
            scaler.update()
            pb, tt = per_bag.cpu().numpy(), tb.numpy()
            t2 = time.time()
            timing["data_wait_s"] += t1 - t0
            timing["gpu_s"] += t2 - t1
            for t in TYPE_NAMES:
                if (tt == t).any():
                    by_type[t].extend(pb[tt == t].tolist())
            if step == cfg["steps"] // 2:
                smi = gpu_util_start()
            if step_global % 10 == 0:
                writer.add_scalar("train/loss", float(pb.mean()), step_global)
                for t, name in TYPE_NAMES.items():
                    if (tt == t).any():
                        writer.add_scalar(f"train/loss_{name}", float(pb[tt == t].mean()), step_global)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step_global)
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(loss=f"{pb.mean():.3f}")
            step_global += 1
        lr = opt.param_groups[0]["lr"]
        sched.step()
        u = gpu_util_read(smi)
        if u:
            timing["util"].append(u)
        ep_time = time.time() - t_ep
        t_m = time.time()
        model.eval()
        emb = embed(model, feat, eval_rows, cfg, torch)
        ho = monitor(score_windows(model, emb, test_s, "K", cfg, torch)) if test_s else monitor(pd.DataFrame())
        tr = monitor(score_windows(model, emb, train_s, "K", cfg, torch))
        n_hard = np.nan
        if cfg["hardneg"]:  # A3: score all healthy training windows; > 0.2 become hard negatives for next epoch
            hw = score_windows(model, emb, tr_healthy, "1", cfg, torch)
            seq_rows = {s["folder"]: s["rows"] for s in tr_healthy}
            hard = [seq_rows[r.folder][r.window_start:r.window_end + 1].tolist()
                    for r in hw[hw.prob > 0.2].itertuples()]
            n_hard = len(hard)
            tmp = hard_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(epoch=epoch, windows=hard)))
            tmp.replace(hard_path)
        timing["monitor_eval_s"] += time.time() - t_m
        row = dict(epoch=epoch, loss=float(np.mean(sum(by_type.values(), []))), lr=lr, epoch_time_s=ep_time,
                   hard_negatives=n_hard, train_auc=tr["auc"], ho_loss=ho["loss"], ho_auc=ho["auc"],
                   ho_balacc=ho["balacc"], ho_recall=ho["recall"], ho_specificity=ho["specificity"])
        for t, name in TYPE_NAMES.items():
            row[f"loss_{name}"] = float(np.mean(by_type[t])) if by_type[t] else np.nan
        log.append(row)
        for key, val in row.items():
            if key != "epoch" and val is not None and not (isinstance(val, float) and np.isnan(val)):
                writer.add_scalar(("heldout/" if key.startswith("ho_") else "epoch/") + key, val, epoch)
        pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        elapsed = time.time() - t_fold
        fold_left = elapsed / epoch * (cfg["epochs"] - epoch)
        mean_fold = np.mean(done_times) if done_times else elapsed / epoch * cfg["epochs"]
        remaining = 0 if fold == "all" else sum(1 for kk in range(cfg.get("n_folds", 10)) if kk > int(fold) and not fold_done(exp, seed, kk))
        if not test_s:
            ho_txt = "none (trained on all trees)"
        elif not np.isnan(ho["balacc"]):
            ho_txt = f"BCE {ho['loss']:.3f}, tree-weighted AUC {ho['auc']:.3f}, balanced acc {ho['balacc']:.3f}"
        else:
            ho_txt = f"BCE {ho['loss']:.3f}, one class only: recall {ho['recall']:.3f} / specificity {ho['specificity']:.3f}"
        losses = ", ".join(f"{name} {row[f'loss_{name}']:.4f}" for name in TYPE_NAMES.values()
                           if not np.isnan(row[f"loss_{name}"]))
        if cfg["hardneg"]:
            losses += f"; hard negatives {n_hard}"
        write_status(dict(exp=exp, seed=seed, approach=cfg["approach"], fold_txt=fold_txt, epoch=epoch,
                          epochs=cfg["epochs"], losses=losses, ho=ho_txt, fold_elapsed=elapsed, fold_left=fold_left,
                          exp_left=fold_left + remaining * mean_fold, queue=queue, last=last_summary))
        draw_curves(exp, seed, fold, log)
        print(f"{exp} s{seed} fold {fold_txt} epoch {epoch}/{cfg['epochs']}: {losses} | held-out {ho_txt} | "
              f"train AUC {tr['auc']:.3f} | {ep_time:.1f}s", flush=True)
    train_total = time.time() - t_fold
    t_e = time.time()
    model.eval()
    torch.save(model.state_dict(), out / "model.pt")
    (out / "config.json").write_text(json.dumps(config, indent=1, default=str))
    if test_s:
        emb = embed(model, feat, np.concatenate([s["rows"] for s in test_s]), cfg, torch)
        strides = ("1", "K", "skip") if cfg["skip"] else ("1", "K")
        pred = pd.concat([score_windows(model, emb, test_s, st, cfg, torch, seed=1000 + int(fold)) for st in strides],
                         ignore_index=True)
        pred.insert(0, "fold", fold)
        pred.insert(0, "seed", seed)
        pred.insert(0, "experiment", exp)
    timing.update(train_s=train_total - timing["monitor_eval_s"], final_eval_s=time.time() - t_e,
                  fold_total_s=time.time() - t_fold, peak_mem_mb=torch.cuda.max_memory_allocated() / 2**20,
                  steps=step_global)
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    if test_s:
        pred.to_csv(out / "predictions.csv", index=False)  # written last: marks the fold complete
    writer.close()
    del it, loader
    return timing


def summarize_run(exp, seed, n_folds=10):
    from report import write_results

    root = run_root(exp, seed)
    pred = pd.concat([pd.read_csv(root / f"fold{k}" / "predictions.csv") for k in range(n_folds)], ignore_index=True)
    pred.to_csv(root / "oof_predictions.csv", index=False)
    s = M.summarize(pred[pred.stride.astype(str).isin(["1", "K"])])
    s.to_csv(root / "summary.csv")
    write_results()
    a, b = (s.loc[k] for k in M.PRIMARY)
    return (f"{exp} seed {seed}: tree-weighted ROC-AUC {a.value:.3f} [{a.lo:.3f}, {a.hi:.3f}], tree-weighted "
            f"balanced accuracy {b.value:.3f} [{b.lo:.3f}, {b.hi:.3f}] (stride-1 windows, 95% tree bootstrap)")


def expand(queue):
    out = []
    for q in queue:
        if q == "K24":
            out += X.LIST_K24
        elif q == "K12":
            out += [f"{r}-K12" for r in X.LIST_K12]
        else:
            out.append(q)
    return out


def main():
    global SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", nargs="*", default=[])
    ap.add_argument("--smoke", nargs="*", default=None, help="run names to smoke-test (1 fold, 2 epochs x 4 steps)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="train every queued run once per seed")
    a = ap.parse_args()
    import decide
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; set PYTHONPATH to outputs/torch-cu121")
    if sys.platform == "win32" and not SMOKE:  # keep the PC from idle-sleeping while this process runs
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    index = C.load_index()
    seqs = C.folder_sequences(index)
    SMOKE = a.smoke is not None
    queue = expand(a.smoke if SMOKE else a.queue)
    last = "none yet"
    i = 0
    while i < len(queue):
        name = queue[i]
        i += 1
        base = X.base_name(name)
        suffix = name[len(base):]
        if base in ("A-win", "FINAL") and not SMOKE:
            plan = decide.plan(base, X.k_of(name))
            print(f"PLAN {name}: {plan['reason']}", flush=True)
            if not plan.get("runs"):
                continue
            jobs = [(f"{n}{suffix}", extra) for n, extra in plan["runs"]]
        elif base in ("A-win", "FINAL"):  # smoke: exercise the composed config with fixed changes
            jobs = [(name, dict(changes=dict(noise=True, tilt=True, seed=42), approach="smoke composition"))]
        else:
            jobs = [(name, None)]
        for job, extra in jobs:
            base_cfg = X.config(job, extra)
            for seed in (a.seeds or [base_cfg["seed"]]):
                cfg = dict(base_cfg, seed=seed)
                is_all = X.base_name(job).startswith("FINAL-all") or bool(cfg.get("train_all"))
                if cfg["grouping"] == "recorder":
                    fold_of, n_folds = None, len(C.recorder_folds()[0])
                elif cfg["grouping"] == "single":  # one fixed tree-grouped split (fold 0 = test trees)
                    fold_of = {g: 0 if g in X.SINGLE_TEST else 1 for g in sorted(index.group.unique())}
                    n_folds = 1
                else:
                    fold_of = C.make_site_folds(index) if cfg["grouping"] == "site25" else C.make_folds(index)
                    n_folds = 10
                cfg["n_folds"] = n_folds
                if SMOKE:
                    cfg.update(epochs=2, steps=4)
                    folds = ["all"] if is_all else [0]
                else:
                    folds = ["all"] if is_all else list(range(n_folds))
                done_times = []
                for fold in folds:
                    if fold_done(job, seed, fold):
                        print(f"{job} seed {seed} fold {fold}: done, skipping", flush=True)
                        continue
                    t = run_fold(job, seed, fold, cfg, fold_of, seqs, queue[i:], done_times, last)
                    done_times.append(t["fold_total_s"])
                    print(f"FOLD DONE {job} s{seed} fold {fold}: " + json.dumps(
                        {kk: (round(v, 1) if isinstance(v, float) else v) for kk, v in t.items() if kk != "util"})
                        + f" util samples {t['util'][:5]}", flush=True)
                if not SMOKE and not is_all and all(fold_done(job, seed, kk) for kk in range(n_folds)):
                    if not (run_root(job, seed) / "summary.csv").exists():
                        last = summarize_run(job, seed, n_folds)
                        print("RUN DONE " + last, flush=True)


if __name__ == "__main__":
    main()
