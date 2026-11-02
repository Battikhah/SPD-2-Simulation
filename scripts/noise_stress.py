"""Stress test: how does stream1s_w16 (fresh seeds 3-5, float) react to real wind/rain noise and white noise?

Leakage-free pairing: each held-out fold's models only score recordings and noise from that same fold.
- fold 2: infested recordings mixed with rain noise from clean folder 8 (both held out),
- fold 4: infested recordings mixed with wind noise from clean folder 7 (both held out),
- folds 0, 2, 3, 4: infested recordings mixed with white noise,
- fold 1: clean folder 9 mixed with white noise (does noise alone cause alarms?).
SNR = clip RMS / noise RMS over the whole clip. Decision: mean of 10+ window probabilities over the 3 seed
models, compared with that fold's F1 threshold chosen on the other folds' float out-of-fold scores.
Development data only. Writes outputs/folder-diagnostics/noise_stress.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esp32_arch import CANDIDATES, torch_model  # noqa: E402
from esp32_export import choose_threshold  # noqa: E402
from esp32_train import dev_frame  # noqa: E402
from grouped_cv_campaign import MEL64, ROOT, frame_power  # noqa: E402

RUN = Path("outputs/esp32-tflm/full-stream1s_w16-pb2.0-s345")
SEEDS = (3, 4, 5)
SNRS = (None, 20, 10, 5, 0)
PER_FOLD = 80


def logmel_windows(x):
    x = np.pad(x, (0, (-len(x)) % 8000)).reshape(-1, 8000)
    return np.stack([np.log(np.maximum(frame_power(c, 128, 59) @ MEL64.T, 1e-10)) for c in x]).astype("float32")


def read(rel):
    x, sr = sf.read(ROOT / rel, dtype="float64", always_2d=True)
    assert sr == 8000
    return x.mean(1)


def mix(x, noise, snr, rng):
    if snr is None:
        return x
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    start = rng.integers(0, len(noise) - len(x) + 1)
    n = noise[start:start + len(x)]
    n = n * (np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(n ** 2)) + 1e-12)) / 10 ** (snr / 20)
    return np.clip(x + n, -1, 1)


def main():
    import torch

    df, groups = dev_frame(False)
    y = df.label.to_numpy()
    folds = list(GroupKFold(5).split(y, y, groups))
    oof = pd.read_csv(RUN / "oof_float.csv")
    ens = oof[[f"stream1s_w16|seed{s}" for s in SEEDS]].mean(1).to_numpy()
    rng = np.random.default_rng(0)
    nets = {}
    for k in range(5):
        for s in SEEDS:
            d = np.load(RUN / "stream1s_w16" / f"seed{s}_fold{k}.npz")
            net = torch_model(CANDIDATES["stream1s_w16"]["spec"])()
            net.load_state_dict({n: torch.tensor(d[n]) for n in d.files if not n.startswith("__")}, strict=False)
            nets[k, s] = (net.eval(), float(d["__mu"]), float(d["__sd"]))

    def score(k, audio):
        w = logmel_windows(audio)
        ps = []
        for s in SEEDS:
            net, mu, sd = nets[k, s]
            with torch.no_grad():
                ps.append(torch.sigmoid(net(torch.tensor((w - mu) / sd).unsqueeze(1))).mean().item())
        return float(np.mean(ps))

    cases = []  # (fold, label, noise name, noise source paths)
    white = ("white", None)
    for k, tag, noise_folder in ((2, "rain (folder 8)", "folder_8"), (4, "wind (folder 7)", "folder_7")):
        cases.append((k, 1, tag, list(df.path[groups == noise_folder])))
    for k in (0, 2, 3, 4):
        cases.append((k, 1, white[0], None))
    cases.append((1, 0, white[0], None))
    rows = []
    for k, label, tag, noise_paths in cases:
        te = folds[k][1]
        thr = choose_threshold(y[folds[k][0]], ens[folds[k][0]], "f1")
        pick = rng.choice(te[y[te] == label], min(PER_FOLD, int((y[te] == label).sum())), replace=False)
        for i in pick:
            x = read(df.path[i])
            noise = rng.standard_normal(len(x)) if noise_paths is None else read(rng.choice(noise_paths))
            for snr in SNRS:
                rows.append(dict(fold=k, label=label, noise=tag, snr="clean" if snr is None else f"{snr} dB",
                                 alarm=score(k, mix(x, noise, snr, rng)) >= thr))
        print(f"fold {k} {tag} done", flush=True)
    r = pd.DataFrame(rows)
    t = r.groupby(["label", "noise", "snr"], sort=False).alarm.agg(["mean", "size"]).reset_index()
    lines = ["Alarm rate at each fold's cross-fitted F1 threshold (stream1s_w16 pb2, seeds 3-5, float).",
             "Infested rows = recall; clean rows = false-positive rate.", "",
             "| Recordings | Noise | No noise | 20 dB | 10 dB | 5 dB | 0 dB | n |", "| --- | --- |" + " ---: |" * 6]
    for (label, noise), g in t.groupby(["label", "noise"], sort=False):
        vals = {row.snr: row.mean for row in g.itertuples()}
        who = "infested" if label else "clean folder 9"
        lines.append(f"| {who} | {noise} | " + " | ".join(f"{100 * vals[c]:.0f}%" for c in
                                                         ("clean", "20 dB", "10 dB", "5 dB", "0 dB"))
                     + f" | {int(g['size'].iloc[0])} |")
    Path("outputs/folder-diagnostics/noise_stress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
