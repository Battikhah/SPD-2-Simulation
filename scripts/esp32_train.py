"""Stage 1 (GPU venv): train the TFLM-friendly candidates on folder-grouped folds and save per-fold
weights for int8 export by `esp32_export.py` (TensorFlow venv).

Development data only; the test partition is never read. Recall-oriented training: positive weight
= 2x class balance, SpecAugment, mixup, OneCycle. Saves outputs/esp32-tflm/<candidate>/seed{s}_fold{k}.npz
(PyTorch state dict + training-fold mu/sd) and oof_float.csv (float recording scores).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esp32_arch import CANDIDATES, TEACHERS, torch_model  # noqa: E402
from grouped_cv_campaign import OUT, RUN, extract  # noqa: E402
from grouped_cv_fullclip import load  # noqa: E402
from grouped_cv_round2 import train  # noqa: E402

EXPORT = Path("outputs/esp32-tflm")


def run_root(smoke, tag=""):
    """Untagged runs use full/ (the original pos_boost=2 results); tagged runs never overwrite them."""
    suffix = f"-{tag}" if tag else ""
    return EXPORT / (("smoke" if smoke else "full") + suffix)


def dev_frame(smoke):
    df = pd.read_csv(RUN / "splits.csv")
    df = df[df.split.isin(["train", "val"])].reset_index(drop=True)  # final holdout never read
    if smoke:
        df = df.groupby(df.path.str.extract(r"(folder_\d+)")[0]).head(8).reset_index(drop=True)
    groups = df.path.str.extract(r"(folder_\d+)")[0].to_numpy()
    return df, groups


def units(name, df, smoke):
    """Model inputs and their recording ids for a candidate's input unit."""
    if CANDIDATES[name]["unit"] == "window":
        f = extract(df, OUT / ("dev_features_smoke.npz" if smoke else "dev_features.npz"))
        return f["logmel"], f["rid"]
    lm = load(df, OUT / ("clip_features_smoke.npz" if smoke else "clip_features.npz"))["logmel"]
    n = len(df)
    return lm.reshape(n, 2, 320, 64).reshape(2 * n, 320, 64), np.repeat(np.arange(n), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", default=list(CANDIDATES))
    ap.add_argument("--pos-boost", type=float, default=2.0, help="multiplier on the class-balance positive weight")
    ap.add_argument("--tag", default="", help="write to outputs/esp32-tflm/full-<tag>/ instead of full/")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="default 0 1 2 (0 for --smoke)")
    ap.add_argument("--epochs", type=int, default=None, help="default 20 (1 s windows) / 40 (10 s segments)")
    ap.add_argument("--teacher", default="", help="distil from this esp32_arch.TEACHERS model (same folds)")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of the hard label in the distillation target")
    ap.add_argument("--balance-folders", type=float, default=0.0,
                    help="weight units by folder_count^-p within each class (0 off, 0.5 sqrt, 1 equal folders)")
    a = ap.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; set PYTHONPATH to outputs/torch-cu121 (GPU-only campaign).")
    root = run_root(a.smoke, a.tag)
    root.mkdir(parents=True, exist_ok=True)
    df, groups = dev_frame(a.smoke)
    y, n = df.label.to_numpy(), len(df)
    folds = list(GroupKFold(5).split(y, y, groups))  # identical folds to earlier rounds
    seeds = a.seeds if a.seeds else ([0] if a.smoke else [0, 1, 2])
    oof = pd.DataFrame(dict(path=df.path, folder=groups, label=y))
    if (root / "oof_float.csv").exists():  # retraining a subset keeps the other candidates' columns
        prior = pd.read_csv(root / "oof_float.csv")
        if list(prior.path) == list(df.path):
            oof = prior
    for name in a.only:
        cfg = CANDIDATES[name]
        X, rid = units(name, df, a.smoke)
        assert X.shape[1:] == cfg["input"]
        window = cfg["unit"] == "window"
        epochs = 1 if a.smoke else (a.epochs or (20 if window else 40))
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        batch = 512 if window else 64
        for s in seeds:
            keep, t0, soft = [], time.time(), None
            if a.teacher:  # teacher sees only the same training folds; its fold-k outputs become soft targets
                teacher = TEACHERS[a.teacher]
                assert teacher["input"] == cfg["input"]
                soft = []
                tp, _ = train(torch_model(teacher["spec"]), X, y[rid], rid, folds, n, epochs, "cuda", 100 * s + 50,
                              True, True, batch=batch, pos_boost=a.pos_boost, predict_all=soft,
                              unit_groups=groups[rid], balance_power=a.balance_folders)
                print(f"teacher {a.teacher} seed{s}: held-out window AUC "
                      f"{roc_auc_score(y[rid], tp):.3f}", flush=True)
            p, params = train(torch_model(cfg["spec"]), X, y[rid], rid, folds, n, epochs, "cuda", 100 * s,
                              True, True, batch=batch, pos_boost=a.pos_boost, keep=keep, soft=soft, alpha=a.alpha,
                              unit_groups=groups[rid], balance_power=a.balance_folders)
            for k, (state, mu, sd) in enumerate(keep):
                arrays = {key: v for key, v in state.items() if not key.endswith("num_batches_tracked")}
                np.savez(out / f"seed{s}_fold{k}.npz", __mu=mu, __sd=sd, **arrays)
            oof[f"{name}|seed{s}"] = np.bincount(rid, p, n) / np.bincount(rid, minlength=n)
            print(f"{name} seed{s}: params {params}, float AUC {roc_auc_score(y, oof[f'{name}|seed{s}']):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        oof.to_csv(root / "oof_float.csv", index=False)
    (root / "train_info.json").write_text(json.dumps(dict(
        smoke=a.smoke, seeds=seeds, pos_boost=a.pos_boost, tag=a.tag, candidates=a.only, test_read=False,
        epochs=a.epochs, teacher=a.teacher, alpha=a.alpha if a.teacher else None, balance_folders=a.balance_folders,
        device=torch.cuda.get_device_name(), torch=torch.__version__), indent=1))


if __name__ == "__main__":
    main()
