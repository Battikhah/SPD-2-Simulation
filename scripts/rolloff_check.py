"""Recording-chain fingerprint check: fine spectrum shape near the 4 kHz Nyquist limit, per folder.

For up to 40 recordings per folder (development data only), a Welch power spectrum (512-point, 15.6 Hz
bins) is averaged in dB. Reported relative to each folder's 1-2 kHz level: the 2.5-3.0, 3.0-3.5, 3.5-3.8
and 3.8-4.0 kHz levels and the frequency above 2 kHz where the spectrum first falls 20 dB below that
reference. A device- or processing-specific anti-aliasing roll-off would show as a folder- or
period-specific shape, independent of what was recorded. Writes outputs/folder-diagnostics/rolloff.md
and rolloff_spectra.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esp32_train import dev_frame  # noqa: E402
from grouped_cv_campaign import ROOT  # noqa: E402

BANDS = [(2500, 3000), (3000, 3500), (3500, 3800), (3800, 4000)]


def main():
    df, groups = dev_frame(False)
    rng = np.random.default_rng(0)
    spectra, meta = {}, []
    for folder in sorted(set(groups), key=lambda f: int(f.split("_")[1])):
        idx = np.flatnonzero(groups == folder)
        idx = rng.choice(idx, min(40, len(idx)), replace=False)
        ps = []
        for i in idx:
            x, sr = sf.read(ROOT / df.path[i], dtype="float64", always_2d=True)
            f, p = welch(x.mean(1), fs=sr, nperseg=512)
            ps.append(10 * np.log10(p + 1e-20))
        spec = np.mean(ps, 0)
        ref = spec[(f >= 1000) & (f < 2000)].mean()
        rel = spec - ref
        spectra[folder] = rel
        drop = f[(f > 2000) & (rel < -20)]
        name = Path(df.path[idx[0]]).name.split("_")[1]
        meta.append(dict(folder=folder, label=int(df.label[idx[0]]), month=int(name[4:6]),
                         rolloff_20db_hz=float(drop[0]) if len(drop) else np.nan)
                    | {f"{lo}-{hi} Hz": float(rel[(f >= lo) & (f < hi)].mean()) for lo, hi in BANDS})
    m = pd.DataFrame(meta).sort_values(["label", "month", "folder"])
    pd.DataFrame(spectra, index=f).to_csv("outputs/folder-diagnostics/rolloff_spectra.csv")
    cols = [f"{lo}-{hi} Hz" for lo, hi in BANDS]
    lines = ["Level (dB) relative to each folder's 1-2 kHz level; 20 dB roll-off = first frequency above 2 kHz",
             "more than 20 dB below that reference.", "",
             "| Folder | Label | Month | " + " | ".join(cols) + " | 20 dB roll-off |",
             "| --- | --- | ---: |" + " ---: |" * (len(cols) + 1)]
    for r in m.to_dict("records"):
        lines.append(f"| {r['folder']} | {'infested' if r['label'] else 'clean'} | {r['month']} | "
                     + " | ".join(f"{r[c]:.1f}" for c in cols) + f" | {r['rolloff_20db_hz']:.0f} Hz |")
    Path("outputs/folder-diagnostics/rolloff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
