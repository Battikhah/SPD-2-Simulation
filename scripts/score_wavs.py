"""Score WAV recordings with an exported int8 TFLite model, using the same front end as training.

Run in the TensorFlow venv:
  outputs/tf-venv/Scripts/python.exe scripts/score_wavs.py folder_1 folder_24 --calibrate-folder folder_24

Inputs are WAV files, directories (searched recursively) or TreeVibes folder names such as folder_24
(that folder's development recordings from the locked manifest). Each recording is cut into 1 s windows
(8 kHz, 59x64 log-mel, the training front end), every window is scored by the int8 reference interpreter,
and the recording score is the mean window score (as in oof_int8.csv); the on-device top-3-of-10 score is
also written.

Safety: recordings in the sealed `test` partition of the locked split are refused, whatever path is
given. The exported model_int8.tflite is the seed-0, fold-0 model of its run, so for dev recordings the
output says whether that model trained on the recording (from train_info.json); only recordings it did
not train on give a fair check. Decisions need a threshold: --threshold, or --calibrate-folder/-wavs,
which sets it at the 90th percentile of known-healthy recordings (per-device calibration). Calibration
recordings are marked and should not be counted as test results.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esp32_export import interpreter, run_int8, top_k_decisions  # noqa: E402
from esp32_train import dev_frame, transform  # noqa: E402
from grouped_cv_campaign import ROOT, RUN, window_features  # noqa: E402

DEFAULT_MODEL = Path("outputs/esp32-tflm/full-stream1s_w8_d4-pb2.0-lorO-bal0.5/stream1s_w8_d4/model_int8.tflite")


def manifest_key(path):
    """Path relative to the dataset root with forward slashes, as in splits.csv; None outside the dataset."""
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return None


def manifest_keys(paths, sealed):
    """Manifest keys of the paths; refuses (exits) if any is in the sealed test partition."""
    keys = [manifest_key(p) for p in paths]
    refused = [k for k in keys if k in sealed]
    if refused:
        raise SystemExit(f"refused: {len(refused)} recording(s) are in the sealed test partition, e.g. "
                         f"{refused[0]}. The final holdout is scored once, through the notebook's gate.")
    return keys


def expand(items, split):
    """WAV paths for files, directories and folder_N names (dev rows of the manifest only)."""
    paths = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            paths += sorted(p.rglob("*.wav"))
        elif p.is_file():
            paths.append(p)
        elif item.startswith("folder_"):
            rows = split[(split.path.str.contains(f"/{item}/", regex=False)) & (split.split != "test")]
            if rows.empty:
                raise SystemExit(f"{item}: no development recordings in the locked manifest")
            paths += [ROOT / r for r in rows.path]
        else:
            raise SystemExit(f"{item}: not a file, directory or folder_N name")
    return paths


def windows(path):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    if sr != 8000:
        raise SystemExit(f"{path}: sample rate {sr}, the model expects 8000 Hz")
    x = x.mean(1)
    return np.stack([window_features(c)[1] for c in np.pad(x, (0, (-len(x)) % 8000)).reshape(-1, 8000)])


def score(blob, paths, info):
    """Recording scores (mean and top-3-of-10 of the 1 s window scores)."""
    X, rid = [], []
    for i, p in enumerate(paths):
        w = windows(p)
        X.append(w)
        rid += [i] * len(w)
    X = transform(np.concatenate(X), info.get("max_band", 64), info.get("gain_norm", False), info.get("min_band", 0))
    rid = np.array(rid)
    s = run_int8(blob, X)
    n = len(paths)
    return np.bincount(rid, s, n) / np.bincount(rid, minlength=n), top_k_decisions(s, rid, n), np.bincount(rid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="WAV files, directories or folder_N names")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, help="fixed decision threshold on the mean score")
    ap.add_argument("--calibrate-folder", nargs="*", default=[], help="folder_N names of known-healthy trees")
    ap.add_argument("--calibrate-wavs", nargs="*", default=[], help="WAVs/directories of known-healthy trees")
    ap.add_argument("--quantile", type=float, default=0.90, help="calibration quantile (0.90: ~10%% false alarms)")
    ap.add_argument("--calibration-share", type=float, default=1.0,
                    help="use this random share (seed 0) of the healthy recordings for calibration and score "
                         "the rest as held-out healthy recordings, e.g. 0.5")
    ap.add_argument("--out", type=Path, default=Path("outputs/score-wavs/scores.csv"))
    a = ap.parse_args()

    split = pd.read_csv(RUN / "splits.csv")
    sealed = set(split.path[split.split == "test"])
    info_path = a.model.parent.parent / "train_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    seen = {}  # manifest path -> did the exported (fold-0) model train on it?
    if info.get("fold_of_recording"):
        df, groups = dev_frame(False)
        excluded = set(info.get("exclude_train_folders") or [])
        for p, g, k in zip(df.path, groups, info["fold_of_recording"]):
            seen[p] = k != 0 and g not in excluded
    else:
        print(f"note: {info_path} has no recorder folds; cannot tell which recordings the model trained on")

    paths = expand(a.inputs, split)
    cal_paths = expand(a.calibrate_folder + a.calibrate_wavs, split) if (a.calibrate_folder or a.calibrate_wavs) else []
    if cal_paths and a.calibration_share < 1:
        perm = np.random.default_rng(0).permutation(len(cal_paths))
        cut = max(1, round(a.calibration_share * len(cal_paths)))
        rest = {Path(cal_paths[i]).resolve() for i in perm[cut:]}
        cal_paths = [cal_paths[i] for i in sorted(perm[:cut])]
        paths += [p for p in sorted(rest) if p not in {Path(q).resolve() for q in paths}]
    keys = manifest_keys(paths, sealed)
    manifest_keys(cal_paths, sealed)
    blob = a.model.read_bytes()
    q = interpreter(blob).get_input_details()[0]["quantization"]
    print(f"model {a.model} ({len(blob)} B, input scale {q[0]:.5f}, zero point {q[1]}); {len(paths)} recordings")

    threshold, source = a.threshold, "--threshold"
    if cal_paths:
        cal_scores, _, _ = score(blob, cal_paths, info)
        threshold = float(np.quantile(cal_scores, a.quantile))
        source = f"{a.quantile:.0%} quantile of {len(cal_paths)} healthy calibration recordings"
    mean, top3, nwin = score(blob, paths, info)

    labels = dict(zip(split.path, split.label))
    cal_set = {Path(p).resolve() for p in cal_paths}
    rows = pd.DataFrame(dict(
        path=[k or str(p) for k, p in zip(keys, paths)],
        folder=[(Path(k).parent.name if k else "") for k in keys],
        label=[labels.get(k, "") for k in keys],
        model_trained_on_it=[("yes" if seen[k] else "no") if k in seen else "unknown" for k in keys],
        used_for_calibration=[Path(p).resolve() in cal_set for p in paths],
        windows=nwin, score_mean=mean, score_top3of10=top3))
    if threshold is not None:
        rows["flagged"] = rows.score_mean >= threshold
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(a.out, index=False)

    summary = rows.groupby(["folder", "label", "model_trained_on_it", "used_for_calibration"], dropna=False).agg(
        recordings=("path", "size"), mean_score=("score_mean", "mean"),
        **({"flagged": ("flagged", "mean")} if threshold is not None else {}))
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
    if threshold is None:
        print("no threshold: pass --threshold or --calibrate-folder/--calibrate-wavs to get decisions")
    else:
        print(f"threshold {threshold:.4f} ({source}); 'flagged' is the share at or above it")
        fair = rows[(rows.model_trained_on_it == "no") & ~rows.used_for_calibration & (rows.label != "")]
        if len(fair):
            y = fair.label.astype(int)
            for cls, name in ((1, "infested recall"), (0, "clean false-alarm rate")):
                sub = fair[y == cls]
                if len(sub):
                    print(f"{name} on recordings the model never trained on and not used for calibration: "
                          f"{sub.flagged.sum()}/{len(sub)} = {sub.flagged.mean():.1%}")
    print(f"per-recording scores: {a.out}")


if __name__ == "__main__":
    main()
