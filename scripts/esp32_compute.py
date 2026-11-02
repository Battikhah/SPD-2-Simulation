"""Multiply-accumulates (MACs) per inference and per 10 s decision for every esp32_arch candidate.

Analytic counts from the shared spec (convolution, depthwise + pointwise, dense); pooling, ReLU and
Mean are ignored. A 1 s window model runs 10 times per 10 s decision; a 10.24 s model runs once. The
sustained rate needed to keep up in real time is MACs per decision / 10 s. Whether the ESP32-WROOM32
sustains it is NOT known: latency has not been measured on a board.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esp32_arch import CANDIDATES  # noqa: E402


def macs(spec, shape):
    h, w = shape
    c, total = 1, 0
    for item in spec:
        kind = item[0]
        if kind == "conv":
            _, out, k, s = item
            h, w = -(-h // s), -(-w // s)
            total += h * w * k * k * c * out
            c = out
        elif kind == "dw":
            _, out, k, _ = item
            total += h * w * (k * k * c + c * out)
            c = out
        elif kind == "pool":
            h, w = h // 2, w // 2
        elif kind == "gap":
            h = w = 1
        elif kind == "dense":
            total += c * item[1]
            c = item[1]
        elif kind == "logit":
            total += c
    return total


def main():
    print("| Model | MACs per inference | Inferences per 10 s | MACs per 10 s decision, 3-model ensemble | "
          "Sustained rate needed |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, cfg in CANDIDATES.items():
        m = macs(cfg["spec"], cfg["input"])
        runs = 10 if cfg["unit"] == "window" else 1
        per = 3 * runs * m
        print(f"| {name} | {m / 1e6:.2f} M | {runs} | {per / 1e6:.0f} M | {per / 10 / 1e6:.1f} M MAC/s |")


if __name__ == "__main__":
    main()
