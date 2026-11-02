"""Write host parity fixtures for firmware/logmel: int16 windows + expected float log-mel + int8 input.

Windows come from development recordings only (train/val rows of the locked split; the test
partition is never read), two 1 s windows from each of 8 clips across both classes (including the
same-device negative folder 24), plus a silent window and a full-scale tone. Expected log-mel uses
the exact training code (`window_features` in grouped_cv_campaign.py); int8 uses the stream1s_w8
input scale/zero point with NumPy float32 arithmetic, as in esp32_export.py.

Binary layout (little-endian): char[4] "LMEL", int32 count, float32 scale, int32 zero_point, then per
window: int16[8000] pcm, float32[59*64] logmel, int8[59*64] quantised.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grouped_cv_campaign import ROOT, RUN, window_features  # noqa: E402

OUT = Path("outputs/logmel-parity/fixtures.bin")
DEVICE = Path("outputs/esp32-tflm/full/device_report.json")


def main():
    report = json.loads(DEVICE.read_text())["stream1s_w8"]
    scale, zp = report["input_scale"], report["input_zero_point"]
    df = pd.read_csv(RUN / "splits.csv")
    df = df[df.split.isin(["train", "val"])]  # final holdout never read
    df["folder"] = df.path.str.extract(r"(folder_\d+)")[0]
    picks = pd.concat([df[df.folder == "folder_24"].head(1),
                       df[(df.label == 0) & (df.folder != "folder_24")].groupby("folder").head(1).head(3),
                       df[df.label == 1].groupby("folder").head(1).head(4)])
    windows = []
    for rel in picks.path:
        pcm, sr = sf.read(ROOT / rel, dtype="int16", always_2d=True)
        assert sr == 8000 and pcm.shape[1] == 1
        windows += [pcm[i * 8000:(i + 1) * 8000, 0] for i in range(2)]
    windows.append(np.zeros(8000, np.int16))
    t = np.arange(8000) / 8000
    windows.append(np.round(32767 * np.sin(2 * np.pi * 440 * t)).astype(np.int16))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("wb") as f:
        f.write(b"LMEL" + struct.pack("<ifi", len(windows), scale, zp))
        for pcm in windows:
            audio = pcm.astype(np.float64) / 32768.0  # soundfile's PCM_16 float convention
            logmel = window_features(audio)[1]
            assert logmel.shape == (59, 64) and logmel.dtype == np.float32
            q = np.clip(np.round(logmel / scale + zp), -128, 127).astype(np.int8)
            f.write(pcm.astype("<i2").tobytes() + logmel.astype("<f4").tobytes() + q.tobytes())
    print(f"wrote {OUT}: {len(windows)} windows from {len(picks)} development clips "
          f"(folders {', '.join(picks.folder)}), scale {scale}, zero point {zp}")


if __name__ == "__main__":
    main()
