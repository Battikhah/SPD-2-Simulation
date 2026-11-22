"""MEM-alarm (handoff 7.8a): a non-learned leaky-max alarm per held-out tree, from existing stride-1 out-of-fold
window predictions (no retraining). After every recording: state = max(decay * state, p).

Two decay definitions, because recordings are about 5 minutes apart rather than hourly:
  per-recording: decay = 0.8 ** (1/24) per recording (the handoff's formula; with 5-minute recordings this is a
                 factor 0.8 per ~2 hours);
  per-day:       decay = 0.8 ** (hours since the previous recording / 24), i.e. 0.8 per 24 hours of wall time
                 (the handoff's intended meaning).
Tree verdicts: alarm on at the end (largest final state over the tree's folders > 0.5) and alarm on for more
than 50% of the updates. Metrics over trees, unweighted and class-weighted, with a stratified tree bootstrap.

    .venv/Scripts/python.exe mil/mem_alarm.py B0 FINAL B0-K12 FINAL-K12
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decide  # noqa: E402
import metrics as M  # noqa: E402

DECAYS = ("per-recording", "per-day")


def tree_alarms(pred, decay_kind):
    s1 = pred[pred.stride.astype(str) == "1"].sort_values(["tree_group", "folder", "window_start"])
    rows = []
    for (tree, folder), d in s1.groupby(["tree_group", "folder"], sort=False):
        p = d.prob.to_numpy()
        t = pd.to_datetime(d.end_time).to_numpy("datetime64[s]").astype(np.int64) / 3600.0
        state, states = 0.0, []
        for i in range(len(p)):
            if decay_kind == "per-recording":
                dec = 0.8 ** (1 / 24)
            else:
                dec = 0.8 ** ((t[i] - t[i - 1]) / 24) if i else 1.0
            state = max(dec * state, p[i])
            states.append(state)
        rows.append(dict(tree=tree, label=int(d.label.iloc[0]), final=states[-1], on=np.sum(np.array(states) > 0.5),
                         n=len(states)))
    r = pd.DataFrame(rows).groupby("tree").agg(label=("label", "first"), final=("final", "max"), on=("on", "sum"),
                                               n=("n", "sum"))
    r["share_on"] = r.on / r.n
    return r


def metrics_for(r):
    out = {}
    for rule, col, thr in (("alarm at end", "final", 0.5), ("on >50% of time", "share_on", 0.5)):
        for w in ("unweighted", "class"):
            weights = M.window_weights(r.index.to_numpy(), r.label.to_numpy(), w)
            m = M.binary_metrics(r.label.to_numpy(), r[col].to_numpy(), weights, (r[col] > thr).to_numpy())
            for key in ("balanced_accuracy", "recall", "specificity", "roc_auc", "precision_at_10pct"):
                out[(rule, w, key)] = m[key]
    return out


def evaluate(name, n=1000, seed=0):
    sd = decide.run_dir(name)
    if sd is None:
        return None
    pred = pd.read_csv(sd / "oof_predictions.csv")
    rows = []
    for decay in DECAYS:
        r = tree_alarms(pred, decay)
        pt = metrics_for(r)
        rng = np.random.default_rng(seed)
        h, i = r.index[r.label == 0].to_numpy(), r.index[r.label == 1].to_numpy()
        bs = []
        for _ in range(n):
            pick = np.concatenate([rng.choice(h, len(h)), rng.choice(i, len(i))])
            rb = r.loc[pick].reset_index(drop=True)
            rb.index = np.arange(len(rb))  # copies count separately
            bs.append(metrics_for(rb))
        bs = pd.DataFrame(bs)
        for key, v in pt.items():
            rows.append(dict(decay=decay, rule=key[0], weighting=key[1], metric=key[2], value=v,
                             lo=bs[key].quantile(0.025), hi=bs[key].quantile(0.975)))
        r.assign(decay=decay).to_csv(sd / f"mem_alarm_trees_{decay}.csv")
    out = pd.DataFrame(rows)
    out.to_csv(sd / "mem_alarm.csv", index=False)
    return out


if __name__ == "__main__":
    for name in sys.argv[1:]:
        res = evaluate(name)
        print(name, "not finished" if res is None else "done")
