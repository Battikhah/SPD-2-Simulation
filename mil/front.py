"""Alternative per-clip front ends for the log-mel CNN, applied identically in training (data.py, after
augmentation) and evaluation (cached by `python mil/front.py`).

  z     per-clip z-score of the dB log-mel (MIL v1 / B0)
  fn    frequency-wise centering: subtract each mel band's mean over the clip, then per-clip z-score. Removes a
        stationary spectral colouring (device response, coupling, steady background) per clip; motivated by
        frequency-wise normalisation for unseen recording devices (Kim et al. 2022, RFN; Primus & Widmer 2023).
  pcen  per-channel energy normalisation of the mel power (librosa.pcen, time constant 0.4 s), then per-clip
        z-score; adaptive gain control suppresses stationary background (Lostanlen et al. 2019).

All transforms are per clip and use no labels or other clips, so caching them over all development clips leaks
nothing between folds. Sealed groups are never indexed.

    .venv/Scripts/python.exe mil/front.py        # writes outputs/mil/cache/z_fn.npy and z_pcen.npy
"""

import sys
from pathlib import Path

import numpy as np

HOP, SR = 256, 8000


def zscore(x):
    return (x - x.mean(axis=(-2, -1), keepdims=True)) / (x.std(axis=(-2, -1), keepdims=True) + 1e-6)


def apply(db, kind):
    """db: (..., 64, T) dB log-mel (float32). Returns the normalised front end (float32)."""
    if kind == "z":
        return zscore(db)
    if kind == "fn":
        return zscore(db - db.mean(axis=-1, keepdims=True))
    if kind == "pcen":
        import librosa

        power = 10.0 ** (db.astype(np.float64) / 10.0)
        p = librosa.pcen(power * (2 ** 31), sr=SR, hop_length=HOP, time_constant=0.4, gain=0.98, bias=2.0,
                         power=0.5, eps=1e-6, axis=-1)
        return zscore(p.astype(np.float32))
    raise ValueError(kind)


def cache_path(kind):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common as C

    return C.CACHE / ("z.npy" if kind == "z" else f"z_{kind}.npy")


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common as C

    raw = np.load(C.CACHE / "raw_db.npy", mmap_mode="r")
    for kind in ("fn", "pcen"):
        out = np.lib.format.open_memmap(C.CACHE / f"z_{kind}.tmp.npy", "w+", np.float16, raw.shape)
        for i in range(0, len(raw), 128):
            out[i:i + 128] = apply(np.asarray(raw[i:i + 128], np.float32), kind).astype(np.float16)
        out.flush()
        del out
        (C.CACHE / f"z_{kind}.tmp.npy").replace(cache_path(kind))
        print(cache_path(kind), flush=True)


if __name__ == "__main__":
    main()
