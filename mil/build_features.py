"""Cache the MIL v1 features for development clips (Section 5 of the handoff), once.

Per clip: first 20 s at 8 kHz (zero-padded), 4th-order Butterworth band-pass 100-2500 Hz with filtfilt, mel
spectrogram (n_fft 512, hop 256, 64 mels, 100-2500 Hz), power_to_db(ref=1, top_db=None) -> (64, 626).
Writes outputs/mil/cache/{clips.csv, raw_db.npy, z.npy} (float16; z = per-clip z-score). Sealed groups are
never indexed or read.

    .venv/Scripts/python.exe mil/build_features.py
"""

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

_BA = None


def features(rel):
    import librosa
    import soundfile as sf
    from scipy.signal import butter, filtfilt

    global _BA
    if _BA is None:
        _BA = butter(4, [100, 2500], btype="bandpass", fs=C.SR)
    x, sr = sf.read(C.ROOT / rel, dtype="float64", always_2d=True)
    assert sr == C.SR, (rel, sr)
    x = x.mean(1)[: C.N_SAMPLES]
    x = np.pad(x, (0, C.N_SAMPLES - len(x)))
    x = filtfilt(*_BA, x)
    mel = librosa.feature.melspectrogram(y=x, sr=C.SR, n_fft=512, hop_length=256, n_mels=C.N_MELS, fmin=100,
                                         fmax=2500)
    db = librosa.power_to_db(mel, ref=1.0, top_db=None)
    assert db.shape == (C.N_MELS, C.N_FRAMES), db.shape
    z = (db - db.mean()) / (db.std() + 1e-6)
    return db.astype(np.float16), z.astype(np.float16)


def main():
    C.CACHE.mkdir(parents=True, exist_ok=True)
    index = C.clip_index()
    print(f"{len(index)} development clips in {index.group.nunique()} groups / {index.folder.nunique()} folders; "
          f"{index.attrs['dropped_duplicates']} duplicate recordings dropped", flush=True)
    shape = (len(index), C.N_MELS, C.N_FRAMES)
    raw = np.lib.format.open_memmap(C.CACHE / "raw_db.npy.tmp", "w+", np.float16, shape)
    z = np.lib.format.open_memmap(C.CACHE / "z.npy.tmp", "w+", np.float16, shape)
    t0 = time.time()
    with Pool(8) as pool:
        for i, (d, zz) in enumerate(pool.imap(features, index.path, chunksize=16)):
            raw[i], z[i] = d, zz
            if i % 500 == 0:
                print(f"features {i}/{len(index)} ({time.time() - t0:.0f}s)", flush=True)
    raw.flush()
    z.flush()
    del raw, z
    for name in ("raw_db.npy", "z.npy"):
        (C.CACHE / f"{name}.tmp").replace(C.CACHE / name)
    index.to_csv(C.CACHE / "clips.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
