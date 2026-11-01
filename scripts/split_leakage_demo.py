"""Development-only diagnostic: how much does a recording-level random split inflate recall?

Uses only train/validation rows of the locked split; the `test` partition is never read.
Features are simple MFCC summaries with a fixed logistic/forest model, not a tuned competitor.
"""

import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold

RUN = Path("outputs/real-training-cuda/full/treevibes-984e68c4e99668c8")
ROOT = Path.home() / ".cache/kagglehub/datasets/potamitis/treevibes/versions/1"


def features(path):
    y, sr = sf.read(path, dtype="float32")
    y = y if y.ndim == 1 else y.mean(1)
    m = librosa.feature.mfcc(y=y[: sr * 10], sr=sr, n_mfcc=20, n_fft=512, hop_length=256)
    return np.r_[m.mean(1), m.std(1)]


def rates(y, p):
    pos, neg = y == 1, y == 0
    return (p[pos] == 1).mean(), (p[neg] == 1).mean()


def main():
    df = pd.read_csv(RUN / "splits.csv")
    df = df[df.split.isin(["train", "val"])].reset_index(drop=True)  # never the final holdout
    X = np.stack([features(ROOT / p) for p in df.path])
    y = df.label.to_numpy()
    folder = df.path.str.extract(r"(folder_\d+)")[0].to_numpy()
    print(f"{len(df)} development recordings, {len(set(folder))} folders, {y.mean():.1%} positive")
    model = lambda: RandomForestClassifier(300, random_state=0, n_jobs=-1)  # noqa: E731
    for name, cv, groups in [
        ("random recording-level (leaky)", StratifiedKFold(5, shuffle=True, random_state=0), None),
        ("held-out folders (GroupKFold)", GroupKFold(5), folder),
    ]:
        pred = np.zeros(len(y), int)
        for tr, te in cv.split(X, y, groups):
            pred[te] = model().fit(X[tr], y[tr]).predict(X[te])
        r, f = rates(y, pred)
        print(f"{name}: recall {r:.2%}, FPR {f:.2%}")


if __name__ == "__main__":
    sys.exit(main())
