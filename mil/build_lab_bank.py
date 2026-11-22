"""Lab pulse bank for A1 (handoff 7.5). Lab clips are never used in evaluation.

Features: the same pipeline as the field clips (band-pass 100-2500 Hz, 64-band mel, hop 256 = 32 ms) on the
10 s lab clip -> mel power (64, 313). Impulse score = number of frames whose band-summed energy is at least
6 dB above the clip's median frame energy. Threshold = 99th percentile of the score over all lab-clean clips.
Score 15,000 random lab-infested clips (seed 0), keep a random 3,000 above the threshold, remove the lab noise
floor by spectral subtraction P_out = max(P - 2 * median_t(P), 0), drop clips left empty.

Writes outputs/mil/cache/lab_bank.npy (float32 mel power, N x 64 x 313), lab_bank.csv, and
mil/lab_bank_examples.png (before/after for a few clips).
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

LAB_SAMPLES = C.SR * 10
_BA = None


def mel_power(path):
    import librosa
    import soundfile as sf
    from scipy.signal import butter, filtfilt

    global _BA
    if _BA is None:
        _BA = butter(4, [100, 2500], btype="bandpass", fs=C.SR)
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    if sr != C.SR:
        return None
    x = x.mean(1)[:LAB_SAMPLES]
    x = np.pad(x, (0, LAB_SAMPLES - len(x)))
    x = filtfilt(*_BA, x)
    return librosa.feature.melspectrogram(y=x, sr=C.SR, n_fft=512, hop_length=256, n_mels=C.N_MELS, fmin=100,
                                          fmax=2500).astype(np.float32)


def impulse_score(p):
    e = p.sum(0)
    return int((e >= np.median(e) * 10 ** 0.6).sum())


def main():
    rng = np.random.default_rng(0)
    clean = sorted((C.ROOT / "lab/lab/clean").glob("*.wav"))
    infested = sorted((C.ROOT / "lab/lab/infested").glob("*.wav"))
    pick = [infested[i] for i in rng.choice(len(infested), 15000, replace=False)]
    with Pool(8) as pool:
        pc = pool.map(mel_power, clean, chunksize=32)
        pi = pool.map(mel_power, pick, chunksize=32)
    bad = sum(p is None for p in pc + pi)
    sc = np.array([impulse_score(p) for p in pc if p is not None])
    thr = np.percentile(sc, 99)
    si = np.array([impulse_score(p) if p is not None else -1 for p in pi])
    above = np.flatnonzero(si > thr)
    keep = rng.choice(above, min(3000, len(above)), replace=False)
    bank, rows = [], []
    for j in keep:
        p = pi[j]
        q = np.maximum(p - 2 * np.median(p, axis=1, keepdims=True), 0)
        if q.sum() <= 0:
            continue
        bank.append(q)
        rows.append(dict(path=pick[j].relative_to(C.ROOT).as_posix(), impulse_score=si[j]))
    np.save(C.CACHE / "lab_bank.npy", np.stack(bank).astype(np.float32))
    pd.DataFrame(rows).to_csv(C.CACHE / "lab_bank.csv", index=False)
    print(f"non-8 kHz clips skipped: {bad}; lab-clean impulse score 99th pct = {thr:.1f} "
          f"(median {np.median(sc):.0f}); lab-infested above threshold: {len(above)}/15000 "
          f"(median score {np.median(si):.0f}); kept {len(bank)} after dropping empty", flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 4, figsize=(14, 5.5), constrained_layout=True)
    for c, j in enumerate(keep[:4]):
        p = pi[j]
        q = np.maximum(p - 2 * np.median(p, axis=1, keepdims=True), 0)
        for r, (img, title) in enumerate(((p, "lab clip"), (q, "after noise-floor removal"))):
            ax[r, c].imshow(10 * np.log10(img + 1e-12), origin="lower", aspect="auto", cmap="magma",
                            vmin=10 * np.log10(p.max()) - 60, vmax=10 * np.log10(p.max()))
            ax[r, c].set_title(f"{title} (score {si[j]})", fontsize=9)
    fig.savefig(C.MIL / "lab_bank_examples.png", dpi=90)


if __name__ == "__main__":
    main()
