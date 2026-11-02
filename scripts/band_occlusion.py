"""Which mel bands make stream1s_w16 call folders 24/25 infested? Development data only.

For the held-out folds that contain folder 24 (fold 4) and folder 25 (fold 2), each fold model (seeds 3-5)
scores every window with one block of 8 mel bands replaced by the training-fold mean (0 after
normalisation). The drop in mean logit shows how much the model relies on that block, separately for the
24/25 windows and the infested windows of the same folds. CPU inference; writes
outputs/folder-diagnostics/band_occlusion.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esp32_arch import CANDIDATES, torch_model  # noqa: E402
from esp32_train import dev_frame  # noqa: E402
from grouped_cv_campaign import MEL64, OUT, extract  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

RUN = Path("outputs/esp32-tflm/full-stream1s_w16-pb2.0-s345/stream1s_w16")
SEEDS = (3, 4, 5)
BLOCK = 8


def main():
    import torch

    df, groups = dev_frame(False)
    y = df.label.to_numpy()
    f = extract(df, OUT / "dev_features.npz")
    X, rid = f["logmel"], f["rid"]
    folds = list(GroupKFold(5).split(y, y, groups))
    centre_hz = MEL64.argmax(1) * 8000 / 512
    wgroup = groups[rid]
    rows = []
    for k in (2, 4):
        te = np.isin(rid, folds[k][1])
        for s in SEEDS:
            d = np.load(RUN / f"seed{s}_fold{k}.npz")
            net = torch_model(CANDIDATES["stream1s_w16"]["spec"])()
            res = net.load_state_dict({n: torch.tensor(d[n]) for n in d.files if not n.startswith("__")},
                                      strict=False)
            assert not res.unexpected_keys and all(m.endswith("num_batches_tracked") for m in res.missing_keys)
            net.eval()
            mu, sd = float(d["__mu"]), float(d["__sd"])
            xs = (X[te] - mu) / sd

            def logits(arr):
                with torch.no_grad():
                    return np.concatenate([net(torch.tensor(arr[i:i + 2048], dtype=torch.float32).unsqueeze(1))
                                           .squeeze(1).numpy() for i in range(0, len(arr), 2048)])

            base = logits(xs)
            for b in range(0, 64, BLOCK):
                occ = xs.copy()
                occ[:, :, b:b + BLOCK] = 0.0
                delta = base - logits(occ)
                for name, m in (("folders 24/25", np.isin(wgroup[te], ["folder_24", "folder_25"])),
                                ("infested", y[rid[te]] == 1)):
                    rows.append(dict(fold=k, seed=s, band=f"{centre_hz[b]:.0f}-{centre_hz[b + BLOCK - 1]:.0f} Hz",
                                     who=name, drop=float(delta[m].mean()), base=float(base[m].mean())))
    r = pd.DataFrame(rows)
    t = r.groupby(["band", "who"]).drop.mean().unstack("who")
    t = t.loc[sorted(t.index, key=lambda s: float(s.split("-")[0]))]
    base = r.groupby("who").base.mean()
    lines = ["Mean logit drop when a block of 8 mel bands is replaced by the training mean "
             f"(stream1s_w16, seeds 3-5, folds 2 and 4). Baseline mean logit: folders 24/25 {base['folders 24/25']:.2f}, "
             f"infested {base['infested']:.2f}.", "", "| Mel bands | Drop, folders 24/25 | Drop, infested |",
             "| --- | ---: | ---: |"]
    lines += [f"| {i} | {v['folders 24/25']:.2f} | {v['infested']:.2f} |" for i, v in t.iterrows()]
    out = Path("outputs/folder-diagnostics/band_occlusion.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
