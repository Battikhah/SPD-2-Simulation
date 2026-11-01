"""Quantify ESP32-WROOM32 limiting factors for candidate log-mel CNNs and the evaluation protocol.

1. Analytic int8 sizes per architecture: parameters, MACs, and a sequential tensor-arena lower bound
   (largest live input+output activation pair, plus the skip tensor for residual blocks). These are
   estimates for planning; only a TFLM build on the board measures arena, latency and energy.
2. Evaluation optimism: compares an operating point chosen on the pooled out-of-fold scores with a
   cross-fitted one (threshold chosen on the other four folds' scores, applied to the held-out fold).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grouped_cv_campaign import RUN  # noqa: E402


def plan(name, shape, layers):
    """layers: (kind, out_channels, kernel, stride); kinds conv, dw (depthwise+pointwise), pool, gap, dense."""
    h, w, c = shape
    params = macs = 0
    peak, cur = 0, h * w * c
    for kind, out, k, s in layers:
        if kind == "conv":
            h2, w2 = -(-h // s), -(-w // s)
            params += (k * k * c + 1) * out
            macs += h2 * w2 * k * k * c * out
        elif kind == "dw":  # depthwise kxk then pointwise 1x1
            h2, w2 = -(-h // s), -(-w // s)
            params += (k * k + 1) * c + (c + 1) * out
            macs += h2 * w2 * (k * k * c + c * out)
        elif kind == "pool":
            h2, w2, out = h // s, w // s, c
        elif kind == "gap":
            h2, w2, out = 1, 1, c
        elif kind == "dense":
            h2, w2 = 1, 1
            params += (h * w * c + 1) * out
            macs += h * w * c * out
        nxt = h2 * w2 * out
        peak = max(peak, cur + nxt)
        h, w, c, cur = h2, w2, out, nxt
    return dict(model=name, input=f"{shape[0]}x{shape[1]}", parameters=params, int8_weight_kib=params / 1024,
                macs_millions=macs / 1e6, arena_lower_bound_kib=peak / 1024)


def current_cnn(width, depth, dense, shape):
    layers = []
    for i in range(depth):
        layers.append(("conv", width * 2**i, 3, 1))
        if i < depth - 1:
            layers.append(("pool", 0, 2, 2))
    return layers + [("gap", 0, 0, 0), ("dense", dense, 0, 0), ("dense", 1, 0, 0)]


ARCHS = [
    ("current 10 s CNN w32 d4", (320, 64, 1), current_cnn(32, 4, 64, None)),
    ("current 10 s CNN w8 d4", (320, 64, 1), current_cnn(8, 4, 16, None)),
    ("current 10 s CNN w4 d4", (320, 64, 1), current_cnn(4, 4, 8, None)),
    ("current 1 s CNN w8 d3 (x10 windows)", (59, 64, 1), current_cnn(8, 3, 16, None)),
    ("strided-stem CNN w16 (10 s)", (320, 64, 1),
     [("conv", 16, 3, 2), ("pool", 0, 2, 2), ("conv", 32, 3, 1), ("pool", 0, 2, 2), ("conv", 64, 3, 1),
      ("pool", 0, 2, 2), ("conv", 64, 3, 1), ("gap", 0, 0, 0), ("dense", 32, 0, 0), ("dense", 1, 0, 0)]),
    ("DS-CNN 64ch x4 (10 s)", (320, 64, 1),
     [("conv", 64, 3, 2), ("pool", 0, 2, 2)] + [("dw", 64, 3, 1)] * 4 + [("gap", 0, 0, 0), ("dense", 1, 0, 0)]),
    ("DS-CNN 32ch x4 (10 s)", (320, 64, 1),
     [("conv", 32, 3, 2), ("pool", 0, 2, 2)] + [("dw", 32, 3, 1)] * 4 + [("gap", 0, 0, 0), ("dense", 1, 0, 0)]),
]


def youden_threshold(y, s):
    f, t, th = roc_curve(y, s)
    return th[np.argmax(t - f)]


def point(y, s, thr):
    p = s >= thr
    tp, fp = (p & (y == 1)).sum(), (p & (y == 0)).sum()
    return tp / (y == 1).sum(), fp / (y == 0).sum(), tp / max(tp + fp, 1)


def main():
    pd.set_option("display.width", 200)
    print(pd.DataFrame([plan(*a) for a in ARCHS]).round(1).to_string(index=False))
    print("\n10 s audio buffered as int16 = 160 KiB; streamed log-mel input 320x64 int8 = 20 KiB.\n")
    o = pd.read_csv("outputs/grouped-cv-campaign/10s/oof_predictions.csv")
    dev = pd.read_csv(RUN / "splits.csv")
    dev = dev[dev.split.isin(["train", "val"])].reset_index(drop=True)
    assert list(dev.path) == list(o.path)
    y = o.label.to_numpy()
    folds = list(GroupKFold(5).split(y, y, o.folder))
    rows = []
    for col in [c for c in o.columns if "|" in c]:
        s = o[col].to_numpy()
        pooled = point(y, s, youden_threshold(y, s))
        pred = np.zeros(len(y), bool)
        for tr, te in folds:  # threshold from the other folds only
            pred[te] = s[te] >= youden_threshold(y[tr], s[tr])
        tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
        rows.append(dict(model=col, pooled_recall=pooled[0], pooled_fpr=pooled[1], pooled_precision=pooled[2],
                         crossfit_recall=tp / y.sum(), crossfit_fpr=fp / (1 - y).sum(),
                         crossfit_precision=tp / max(tp + fp, 1)))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
