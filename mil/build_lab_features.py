"""Lab recordings as extra training data for the L-* runs (training only; never evaluated).

TreeVibes `lab/lab/` holds 10 s, 8 kHz clips: 967 clean and 53,676 infested. They are not in the locked split
manifest, so they are outside the sealed holdout. Field clips are 20 s, so lab clips are paired within a class
(random permutation, seed 0) and concatenated into 20 s pseudo-clips: all 967 clean files -> 483 clean pseudo-clips,
1,934 random infested files -> 967 infested pseudo-clips (about the field infested:healthy ratio of 2:1 by groups).

Each pseudo-clip goes through exactly the field pipelines:
  log-mel  build_features.features (band-pass 100-2500 Hz, 64 mels, hop 256, dB) -> z-scored (64, 626)
  MFCC     build_mfcc (DCT of the dB mel, 13 coefficients, 30 mean-pooled chunks) -> (13, 30)
  AST      build_embeddings.ast_model (16 kHz, two 10 s halves, patch tokens max-pooled) -> 768-d
Writes outputs/mil/cache/lab_clips.csv and combined caches (field rows first, unchanged, then lab rows):
z_plus.npy, mfcc13x30_plus.npy, emb_ast_plus.npy.

    $env:PYTHONPATH = (Resolve-Path outputs/torch-cu121).Path
    .venv/Scripts/python.exe mil/build_lab_features.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

LAB = C.ROOT / "lab" / "lab"
N_INFESTED_FILES = 1934


def read10(path):
    import soundfile as sf

    x, sr = sf.read(path, dtype="float64", always_2d=True)
    assert sr == C.SR, (path, sr)
    x = x.mean(1)[: C.SR * 10]
    return np.pad(x, (0, C.SR * 10 - len(x)))


def pseudo_clips():
    rng = np.random.default_rng(0)
    clean = sorted(p.name for p in (LAB / "clean").glob("*.wav"))
    inf = sorted(p.name for p in (LAB / "infested").glob("*.wav"))
    inf = list(rng.choice(inf, N_INFESTED_FILES, replace=False))
    rows = []
    for label, cls, files in ((0, "clean", clean), (1, "infested", inf)):
        files = list(rng.permutation(files))
        for i in range(0, len(files) - 1, 2):
            rows.append(dict(label=label, a=f"{cls}/{files[i]}", b=f"{cls}/{files[i + 1]}"))
    return pd.DataFrame(rows)


def main():
    import librosa
    import torch
    from build_embeddings import ast_model
    from scipy.fft import dct
    from scipy.signal import butter, filtfilt, resample_poly

    t0 = time.time()
    lab = pseudo_clips()
    lab.to_csv(C.CACHE / "lab_clips.csv", index=False)
    print(f"{(lab.label == 0).sum()} clean / {(lab.label == 1).sum()} infested pseudo-clips", flush=True)
    ba = butter(4, [100, 2500], btype="bandpass", fs=C.SR)
    n = len(lab)
    z = np.zeros((n, C.N_MELS, C.N_FRAMES), np.float16)
    mfcc = np.zeros((n, 13, 30), np.float32)
    emb = np.zeros((n, 768), np.float32)
    bounds = np.array_split(np.arange(C.N_FRAMES), 30)
    f, _, bs = ast_model(torch)
    for i0 in range(0, n, bs):  # streamed in AST-sized batches to keep memory (and the paging file) small
        wav16 = []
        for i in range(i0, min(i0 + bs, n)):
            r = lab.iloc[i]
            x = np.concatenate([read10(LAB / r.a), read10(LAB / r.b)])  # 20 s, same as a field clip
            mel = librosa.feature.melspectrogram(y=filtfilt(*ba, x), sr=C.SR, n_fft=512, hop_length=256,
                                                 n_mels=C.N_MELS, fmin=100, fmax=2500)
            db = librosa.power_to_db(mel, ref=1.0, top_db=None)
            assert db.shape == (C.N_MELS, C.N_FRAMES), db.shape
            z[i] = ((db - db.mean()) / (db.std() + 1e-6)).astype(np.float16)
            m = dct(db, type=2, axis=0, norm="ortho")[:13]
            mfcc[i] = np.stack([m[:, b].mean(1) for b in bounds], axis=1)
            wav16.append(resample_poly(x.astype(np.float32), 2, 1).astype(np.float32))  # raw, as for AST
        with torch.no_grad():
            emb[i0:i0 + len(wav16)] = f(torch.from_numpy(np.stack(wav16)).cuda()).float().cpu().numpy()
        if i0 % (bs * 25) == 0:
            print(f"lab features {i0}/{n} ({time.time() - t0:.0f}s)", flush=True)
    assert np.isfinite(emb).all()
    for name, part in (("z", z), ("mfcc13x30", mfcc), ("emb_ast", emb)):
        field = np.load(C.CACHE / f"{name}.npy", mmap_mode="r")
        tmp = C.CACHE / f"{name}_plus.tmp.npy"
        out = np.lib.format.open_memmap(tmp, "w+", field.dtype, (len(field) + n, *field.shape[1:]))
        for i in range(0, len(field), 256):
            j = min(i + 256, len(field))
            out[i:j] = field[i:j]
        out[len(field):] = part.astype(field.dtype)
        out.flush()
        del out
        tmp.replace(C.CACHE / f"{name}_plus.npy")
        print(f"{name}_plus: {(len(field) + n, *field.shape[1:])}", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
