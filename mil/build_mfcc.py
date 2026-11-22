"""Cache whole-clip MFCC features (13 coefficients x T time chunks) for the development clips.

Derived from the cached MIL v1 log-mel (outputs/mil/cache/raw_db.npy: 100-2500 Hz band-pass, n_fft 512, hop 256,
64 mels, dB), so no audio is re-read and the sealed groups are never touched. Per clip:
    MFCC = DCT-II (orthonormal) over the 64 mel bands of the dB log-mel, first 13 coefficients -> (13, 626)
    (identical to librosa.feature.mfcc(S=db, n_mfcc=13)); then the 626 frames are split into T contiguous chunks
    (np.array_split) and each chunk is mean-pooled -> (13, T).
Writes outputs/mil/cache/mfcc13x{T}.npy (float32) for T = 15 and 30. Each chunk covers about 1.33 s (T=15) or
0.67 s (T=30) of the 20 s clip.

    .venv/Scripts/python.exe mil/build_mfcc.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

N_MFCC = 13
CHUNKS = (15, 30)


def main():
    raw = np.load(C.CACHE / "raw_db.npy", mmap_mode="r")
    n = len(raw)
    bounds = {t: np.array_split(np.arange(raw.shape[2]), t) for t in CHUNKS}
    out = {t: np.zeros((n, N_MFCC, t), np.float32) for t in CHUNKS}
    for i in range(0, n, 256):
        db = np.asarray(raw[i:i + 256], np.float64)
        mfcc = dct(db, type=2, axis=1, norm="ortho")[:, :N_MFCC]  # (b, 13, 626)
        for t in CHUNKS:
            out[t][i:i + 256] = np.stack([mfcc[:, :, b].mean(2) for b in bounds[t]], axis=2)
    for t in CHUNKS:
        path = C.CACHE / f"mfcc13x{t}.npy"
        np.save(path.with_suffix(".tmp.npy"), out[t])
        path.with_suffix(".tmp.npy").replace(path)
        print(f"{path}: {out[t].shape}", flush=True)


if __name__ == "__main__":
    main()
