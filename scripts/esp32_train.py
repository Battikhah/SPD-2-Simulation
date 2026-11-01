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
from esp32_arch import CANDIDATES, torch_model  # noqa: E402
from grouped_cv_campaign import OUT, RUN, extract  # noqa: E402
from grouped_cv_fullclip import load  # noqa: E402
from grouped_cv_round2 import train  # noqa: E402

EXPORT = Path("outputs/esp32-tflm")


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
    a = ap.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; set PYTHONPATH to outputs/torch-cu121 (GPU-only campaign).")
    root = EXPORT / ("smoke" if a.smoke else "full")
    df, groups = dev_frame(a.smoke)
    y, n = df.label.to_numpy(), len(df)
    folds = list(GroupKFold(5).split(y, y, groups))  # identical folds to earlier rounds
    seeds = [0] if a.smoke else [0, 1, 2]
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
        epochs = 1 if a.smoke else (20 if window else 40)
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            keep, t0 = [], time.time()
            p, params = train(torch_model(cfg["spec"]), X, y[rid], rid, folds, n, epochs, "cuda", 100 * s,
                              True, True, batch=512 if window else 64, pos_boost=2.0, keep=keep)
            for k, (state, mu, sd) in enumerate(keep):
                arrays = {key: v for key, v in state.items() if not key.endswith("num_batches_tracked")}
                np.savez(out / f"seed{s}_fold{k}.npz", __mu=mu, __sd=sd, **arrays)
            oof[f"{name}|seed{s}"] = np.bincount(rid, p, n) / np.bincount(rid, minlength=n)
            print(f"{name} seed{s}: params {params}, float AUC {roc_auc_score(y, oof[f'{name}|seed{s}']):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        oof.to_csv(root / "oof_float.csv", index=False)
    (root / "train_info.json").write_text(json.dumps(dict(
        smoke=a.smoke, seeds=seeds, pos_boost=2.0, candidates=a.only, test_read=False,
        device=torch.cuda.get_device_name(), torch=torch.__version__), indent=1))


if __name__ == "__main__":
    main()
