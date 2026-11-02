"""TFLM-friendly architectures defined once and built identically in PyTorch (GPU training) and Keras
(int8 TFLite export). Ops: Conv2D, DepthwiseConv2D, MaxPool, Mean, FullyConnected, Add, Logistic.

Spec items: ("conv", out, k, stride) | ("dw", out, k, residual) | ("pool",) | ("gap",) | ("dense", units)
| ("logit",). Convolutions have no bias and are followed by BatchNorm + ReLU (folded by the converter).
Padding is explicit (k // 2 zeros, then valid convolution) so PyTorch and TensorFlow agree for stride 2.
"""

CANDIDATES = {
    # 1 s windows (59x64); the MCU scores ten consecutive windows of a 10 s capture and averages them.
    "stream1s_w8": dict(input=(59, 64), unit="window", spec=[
        ("conv", 8, 3, 1), ("pool",), ("conv", 16, 3, 1), ("pool",), ("conv", 32, 3, 1), ("gap",),
        ("dense", 16), ("logit",)]),
    # Overnight sweep variants of the 1 s family (all well under the ~96 KiB arena target).
    "stream1s_w4": dict(input=(59, 64), unit="window", spec=[
        ("conv", 4, 3, 1), ("pool",), ("conv", 8, 3, 1), ("pool",), ("conv", 16, 3, 1), ("gap",),
        ("dense", 8), ("logit",)]),
    "stream1s_w16": dict(input=(59, 64), unit="window", spec=[
        ("conv", 16, 3, 1), ("pool",), ("conv", 32, 3, 1), ("pool",), ("conv", 64, 3, 1), ("gap",),
        ("dense", 32), ("logit",)]),
    "stream1s_w8_d4": dict(input=(59, 64), unit="window", spec=[
        ("conv", 8, 3, 1), ("pool",), ("conv", 16, 3, 1), ("pool",), ("conv", 32, 3, 1), ("pool",),
        ("conv", 64, 3, 1), ("gap",), ("dense", 16), ("logit",)]),
    "stream1s_ds": dict(input=(59, 64), unit="window", spec=[
        ("conv", 16, 3, 1), ("pool",), ("dw", 32, 3, False), ("dw", 32, 3, True), ("pool",),
        ("dw", 64, 3, False), ("dw", 64, 3, True), ("gap",), ("logit",)]),
    # Ablation: stream1s_w8 with the stride-2 stem the 10 s models need, to test whether early loss of
    # time-frequency resolution (not input length) is what makes the 10 s models worse.
    "stream1s_w8_s2": dict(input=(59, 64), unit="window", spec=[
        ("conv", 8, 3, 2), ("pool",), ("conv", 16, 3, 1), ("pool",), ("conv", 32, 3, 1), ("gap",),
        ("dense", 16), ("logit",)]),
    # 10.24 s segments (320x64) with a stride-2 stem so the first activation is 160x32.
    "strided10s_w4": dict(input=(320, 64), unit="segment", spec=[
        ("conv", 4, 3, 2), ("pool",), ("conv", 16, 3, 1), ("pool",), ("conv", 32, 3, 1), ("pool",),
        ("conv", 32, 3, 1), ("gap",), ("dense", 16), ("logit",)]),
    "dscnn10s_w16": dict(input=(320, 64), unit="segment", spec=[
        ("conv", 8, 3, 2), ("pool",), ("conv", 16, 3, 1), ("pool",), ("dw", 16, 3, True), ("dw", 16, 3, True),
        ("pool",), ("dw", 32, 3, False), ("dw", 32, 3, True), ("gap",), ("logit",)]),
    "strided10s_w8": dict(input=(320, 64), unit="segment", spec=[
        ("conv", 8, 3, 2), ("pool",), ("conv", 32, 3, 1), ("pool",), ("conv", 64, 3, 1), ("pool",),
        ("conv", 64, 3, 1), ("gap",), ("dense", 32), ("logit",)]),
    # 10.24 s depthwise-separable residual network: more depth per byte and per MAC. The extra pool
    # before the residual blocks keeps the int8 arena under ~96 KiB (123 KiB without it).
    "dscnn10s_res": dict(input=(320, 64), unit="segment", spec=[
        ("conv", 8, 3, 2), ("pool",), ("conv", 32, 3, 1), ("pool",), ("dw", 32, 3, True), ("dw", 32, 3, True),
        ("pool",), ("dw", 64, 3, False), ("dw", 64, 3, True), ("gap",), ("logit",)]),
}


# Distillation teachers: larger models trained on the same training folds; never exported to the device.
TEACHERS = {
    "teacher1s_w32": dict(input=(59, 64), unit="window", spec=[
        ("conv", 32, 3, 1), ("pool",), ("conv", 64, 3, 1), ("pool",), ("conv", 128, 3, 1), ("gap",),
        ("dense", 64), ("logit",)]),
    "teacher10s_w32": dict(input=(320, 64), unit="segment", spec=[
        ("conv", 32, 3, 1), ("pool",), ("conv", 64, 3, 1), ("pool",), ("conv", 128, 3, 1), ("pool",),
        ("conv", 128, 3, 1), ("gap",), ("dense", 64), ("logit",)]),
}


def torch_model(spec):
    import torch.nn as nn
    import torch.nn.functional as F

    class Conv(nn.Module):
        def __init__(self, c, out, k, stride):
            super().__init__()
            self.conv = nn.Conv2d(c, out, k, stride, k // 2, bias=False)
            self.bn = nn.BatchNorm2d(out)

        def forward(self, x):
            return F.relu(self.bn(self.conv(x)))

    class DW(nn.Module):
        def __init__(self, c, out, k, residual):
            super().__init__()
            self.dw = nn.Conv2d(c, c, k, 1, k // 2, groups=c, bias=False)
            self.bn1 = nn.BatchNorm2d(c)
            self.pw = nn.Conv2d(c, out, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out)
            self.residual = residual and c == out

        def forward(self, x):
            y = self.bn2(self.pw(F.relu(self.bn1(self.dw(x)))))
            return F.relu(y + x if self.residual else y)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            layers, c = [], 1
            for item in spec:
                kind = item[0]
                if kind == "conv":
                    layers.append(Conv(c, *item[1:]))
                    c = item[1]
                elif kind == "dw":
                    layers.append(DW(c, *item[1:]))
                    c = item[1]
                elif kind == "pool":
                    layers.append(nn.MaxPool2d(2))
                elif kind == "gap":
                    layers.append(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()))
                elif kind == "dense":
                    layers.append(nn.Sequential(nn.Linear(c, item[1]), nn.ReLU()))
                    c = item[1]
                elif kind == "logit":
                    layers.append(nn.Sequential(nn.Dropout(0.2), nn.Linear(c, 1)))
            self.layers = nn.ModuleList(layers)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return Net


def keras_model(spec, input_hw, weights, mu, sd):
    """Build the Keras twin, load ported PyTorch weights, and fold input normalisation into the graph,
    so the device feeds raw log-mel quantised with the model's input scale. Output is a probability."""
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers as L

    def bn(prefix):
        return [weights[f"{prefix}.weight"], weights[f"{prefix}.bias"], weights[f"{prefix}.running_mean"],
                weights[f"{prefix}.running_var"]]

    inp = tf.keras.Input((*input_hw, 1))
    x = L.Rescaling(1.0 / sd, offset=-mu / sd)(inp)
    pending, c = [], 1
    for i, item in enumerate(spec):
        kind, p = item[0], f"layers.{i}"
        if kind == "conv":
            _, out, k, stride = item
            x = L.ZeroPadding2D(k // 2)(x)
            conv = L.Conv2D(out, k, stride, "valid", use_bias=False)
            norm = L.BatchNormalization(epsilon=1e-5)
            x = L.ReLU()(norm(conv(x)))
            pending += [(conv, [weights[f"{p}.conv.weight"].transpose(2, 3, 1, 0)]), (norm, bn(f"{p}.bn"))]
            c = out
        elif kind == "dw":
            _, out, k, residual = item
            skip = x
            y = L.ZeroPadding2D(k // 2)(x)
            dw = L.DepthwiseConv2D(k, 1, "valid", use_bias=False)
            n1 = L.BatchNormalization(epsilon=1e-5)
            pw = L.Conv2D(out, 1, use_bias=False)
            n2 = L.BatchNormalization(epsilon=1e-5)
            y = n2(pw(L.ReLU()(n1(dw(y)))))
            x = L.ReLU()(L.Add()([y, skip]) if residual and c == out else y)
            pending += [(dw, [weights[f"{p}.dw.weight"].transpose(2, 3, 0, 1)]), (n1, bn(f"{p}.bn1")),
                        (pw, [weights[f"{p}.pw.weight"].transpose(2, 3, 1, 0)]), (n2, bn(f"{p}.bn2"))]
            c = out
        elif kind == "pool":
            x = L.MaxPooling2D(2)(x)
        elif kind == "gap":
            x = L.GlobalAveragePooling2D()(x)
        elif kind == "dense":
            d = L.Dense(item[1], activation="relu")
            x = d(x)
            pending.append((d, [weights[f"{p}.0.weight"].T, weights[f"{p}.0.bias"]]))
        elif kind == "logit":
            d = L.Dense(1, activation="sigmoid")
            x = d(x)
            pending.append((d, [weights[f"{p}.1.weight"].T, weights[f"{p}.1.bias"]]))
    model = tf.keras.Model(inp, x)
    for layer, w in pending:
        layer.set_weights([np.asarray(a, dtype=np.float32) for a in w])
    return model
