"""Why are the same-device clean folders 24/25 scored as infested? Development-data acoustic diagnostics.

Per recording (development split only; the test partition is never read):
- loudness (RMS, dBFS) and the share of energy below 250 Hz (wind/handling) and above 1 kHz,
- impulsiveness of the >500 Hz band: transient events per second above 6 robust sigmas (borer
  activity is impulsive) and the kurtosis of the high-passed signal,
- the mean 64-band log-mel spectrum (from the cached 1 s window features).

Recordings are grouped by label and recording month (from the file name, F_YYYYMMDD...), because every
clean folder except 24/25 was recorded in February, while 24/25 were recorded in May alongside most
infested folders. Writes outputs/folder-diagnostics/{recordings.csv, folders.csv, summary.md}.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esp32_train import dev_frame  # noqa: E402
from grouped_cv_campaign import OUT, ROOT, extract  # noqa: E402

DEST = Path("outputs/folder-diagnostics")
SR = 8000
HP = butter(4, 500, "highpass", fs=SR, output="sos")
LP = butter(4, 250, "lowpass", fs=SR, output="sos")
HP1K = butter(4, 1000, "highpass", fs=SR, output="sos")


def group_of(label, month, folder):
    if folder in ("folder_24", "folder_25"):
        return "clean, May (24/25)"
    return f"{'infested' if label else 'clean'}, {'Feb-Mar' if month in (2, 3) else 'May-Jun'}"


def audio_stats(x):
    x = x - x.mean()
    total = np.mean(x ** 2) + 1e-20
    hp = sosfiltfilt(HP, x)
    sigma = 1.4826 * np.median(np.abs(hp)) + 1e-12
    above = np.flatnonzero(np.abs(hp) > 6 * sigma)
    stats = dict(rms_dbfs=10 * np.log10(total), low_share=np.mean(sosfiltfilt(LP, x) ** 2) / total,
                 high_share=np.mean(sosfiltfilt(HP1K, x) ** 2) / total, hp_kurtosis=float(kurtosis(hp)),
                 hp_noise_dbfs=20 * np.log10(sigma), events_per_s=0.0, click_ms=np.nan, click_centroid_hz=np.nan,
                 iei_cv=np.nan)
    if len(above) == 0:
        return stats, None
    # Events: runs of above-threshold samples separated by > 20 ms.
    breaks = np.flatnonzero(np.diff(above) > SR // 50)
    starts, ends = above[np.r_[0, breaks + 1]], above[np.r_[breaks, len(above) - 1]]
    peaks = np.array([s + np.argmax(np.abs(hp[s:e + 1])) for s, e in zip(starts, ends)])
    spectra = []
    for p in peaks:  # 32 ms around each peak, Hann window, power spectrum of the unfiltered signal
        seg = x[max(p - 128, 0):p + 128]
        if len(seg) == 256:
            spectra.append(np.abs(np.fft.rfft(seg * np.hanning(256))) ** 2)
    spec = np.mean(spectra, 0) if spectra else None
    freqs = np.fft.rfftfreq(256, 1 / SR)
    iei = np.diff(peaks) / SR
    stats |= dict(events_per_s=len(peaks) / (len(x) / SR), click_ms=1000 * np.median(ends - starts + 1) / SR,
                  click_centroid_hz=float((spec * freqs).sum() / spec.sum()) if spec is not None else np.nan,
                  iei_cv=float(iei.std() / iei.mean()) if len(iei) > 2 else np.nan)
    return stats, spec


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    df, groups = dev_frame(False)
    rows, click_spectra = [], []
    for rel, label, folder in zip(df.path, df.label, groups):
        x, sr = sf.read(ROOT / rel, dtype="float64", always_2d=True)
        assert sr == SR
        stamp = Path(rel).name.split("_")[1]
        stats, spec = audio_stats(x.mean(1))
        rows.append(dict(path=rel, folder=folder, label=int(label), month=int(stamp[4:6]), hour=int(stamp[8:10]))
                    | stats)
        click_spectra.append(np.full(129, np.nan) if spec is None else spec)
    rec = pd.DataFrame(rows)
    rec["group"] = [group_of(r.label, r.month, r.folder) for r in rec.itertuples()]
    # Mean click spectrum per group (dB, each recording normalised to its own total click power first).
    cs = np.array(click_spectra)
    cs_db = 10 * np.log10(cs / np.nansum(cs, 1, keepdims=True))
    click_table = pd.DataFrame(cs_db, columns=[f"{f:.0f}Hz" for f in np.fft.rfftfreq(256, 1 / SR)])
    click_table.groupby(rec.group.to_numpy()).mean().to_csv(DEST / "click_spectra_by_group.csv")
    click_table.groupby(groups).mean().to_csv(DEST / "click_spectra_by_folder.csv")
    f = extract(df, OUT / "dev_features.npz")
    n = len(df)
    mel = np.stack([np.bincount(f["rid"], f["logmel"][:, :, b].mean(1), n) / np.bincount(f["rid"], minlength=n)
                    for b in range(64)], 1)
    rec.to_csv(DEST / "recordings.csv", index=False)
    feats = ["rms_dbfs", "hp_noise_dbfs", "low_share", "high_share", "events_per_s", "hp_kurtosis", "click_ms",
             "click_centroid_hz", "iei_cv"]
    folders = rec.groupby(["folder", "group"])[feats + ["hour"]].median().reset_index()
    folders.to_csv(DEST / "folders.csv", index=False)

    lines = ["# Folder diagnostics (development data only)", "",
             "Medians per group; events/s = >500 Hz transients above 6 robust sigmas.", "",
             "| Group | Recordings | " + " | ".join(feats) + " |", "| --- | ---: |" + " ---: |" * len(feats)]
    for g, s in rec.groupby("group"):
        lines.append(f"| {g} | {len(s)} | " + " | ".join(f"{s[c].median():.3g}" for c in feats) + " |")
    # Single-feature separability: infested vs all clean, and May infested vs 24/25 only.
    may_pos = rec.group.eq("infested, May-Jun")
    dev24 = rec.group.eq("clean, May (24/25)")
    lines += ["", "| Feature | AUC infested vs all clean | AUC May infested vs 24/25 |", "| --- | ---: | ---: |"]
    for c in feats:
        ok = rec[c].notna()
        a1 = roc_auc_score(rec.label[ok], rec[c][ok])
        sub = rec[(may_pos | dev24) & ok]
        a2 = roc_auc_score(sub.label, sub[c])
        lines.append(f"| {c} | {a1:.3f} | {a2:.3f} |")
    # Which folders' mean spectrum is closest to 24/25 (spectral colour only, per-band mean log-mel)?
    spec = pd.DataFrame(mel).groupby(groups).mean()
    lines += ["", "Nearest folders by mean log-mel spectrum (Euclidean, 64 bands):", ""]
    for target in ("folder_24", "folder_25"):
        d = np.sqrt(((spec - spec.loc[target]) ** 2).sum(1)).drop(target).sort_values()
        lab = rec.groupby("folder").label.first()
        lines.append(f"- {target}: " + ", ".join(f"{k} ({'inf' if lab[k] else 'clean'}, {v:.1f})"
                                                 for k, v in d.head(6).items()))
    np.save(DEST / "folder_mean_logmel.npy", spec.to_numpy())
    spec.to_csv(DEST / "folder_mean_logmel.csv")
    (DEST / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
