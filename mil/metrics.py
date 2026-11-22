"""Metrics for pooled out-of-fold MIL predictions (Section 6 of the handoff).

Window level (separately for stride 1 and stride K, threshold 0.5) with three weightings: unweighted,
tree-weighted (1 / windows of the tree) and class-weighted (1 / windows of the class). Tree level: three fixed
rules, unweighted and class-weighted. Confidence intervals: stratified cluster bootstrap over tree groups
(healthy and infested resampled separately, 1,000 draws; repeated trees count as separate copies). Paired
comparisons reuse the same draws for both experiments.
"""

import numpy as np
import pandas as pd

THRESHOLD = 0.5
WEIGHTINGS = ("unweighted", "tree", "class")
METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc",
           "precision_at_10pct")
RULES = {"mean>0.5": "mean", "any>0.5": "max", "share>25%": "share"}
PRIMARY = ("win/s1/tree/roc_auc", "win/s1/tree/balanced_accuracy")


def weighted_auc(y, s, w):
    pos, neg = y == 1, y == 0
    wp, wn = w[pos].sum(), w[neg].sum()
    if wp == 0 or wn == 0:
        return np.nan
    u, inv = np.unique(s, return_inverse=True)
    p_u = np.bincount(inv, w * pos, len(u))
    n_u = np.bincount(inv, w * neg, len(u))
    below = np.cumsum(n_u) - n_u
    return float((p_u * (below + 0.5 * n_u)).sum() / (wp * wn))


def weighted_ap(y, s, w):
    """Average precision as in scikit-learn (step-wise, thresholds at unique scores, descending)."""
    wp = (w * (y == 1)).sum()
    if wp == 0:
        return np.nan
    u, inv = np.unique(-s, return_inverse=True)  # ascending -s = descending s
    tp = np.cumsum(np.bincount(inv, w * (y == 1), len(u)))
    fp = np.cumsum(np.bincount(inv, w * (y == 0), len(u)))
    prec = tp / np.maximum(tp + fp, 1e-300)
    rec = tp / wp
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def binary_metrics(y, s, w, pred=None):
    y, s, w = np.asarray(y), np.asarray(s, float), np.asarray(w, float)
    pred = (s > THRESHOLD) if pred is None else np.asarray(pred, bool)
    tp = w[(y == 1) & pred].sum()
    fn = w[(y == 1) & ~pred].sum()
    fp = w[(y == 0) & pred].sum()
    tn = w[(y == 0) & ~pred].sum()
    rec = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and not np.isnan(prec + rec) else np.nan
    p10 = rec * 0.1 / (rec * 0.1 + (1 - spec) * 0.9) if not np.isnan(rec + spec) and (rec + (1 - spec)) else np.nan
    return dict(accuracy=(tp + tn) / (tp + fn + fp + tn), balanced_accuracy=np.nanmean([rec, spec]),
                precision=prec, recall=rec, specificity=spec, f1=f1, roc_auc=weighted_auc(y, s, w),
                pr_auc=weighted_ap(y, s, w), precision_at_10pct=p10, tp=tp, fn=fn, fp=fp, tn=tn)


def window_weights(tree_ids, labels, scheme):
    if scheme == "unweighted":
        return np.ones(len(tree_ids))
    key = tree_ids if scheme == "tree" else labels
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def tree_scores(tree_ids, labels, probs):
    """Per tree: label, mean prob, max prob, share of windows > threshold."""
    d = pd.DataFrame(dict(t=tree_ids, y=labels, p=probs, f=probs > THRESHOLD))
    g = d.groupby("t", sort=False)
    return pd.DataFrame(dict(y=g.y.first(), mean=g.p.mean(), max=g.p.max(), share=g.f.mean()))


def all_metrics(tree_ids, labels, probs, strides):
    """Every reported metric for one set of windows. tree_ids: one id per tree copy."""
    out = {}
    for st in ("1", "K"):
        m = strides == st
        t, y, p = tree_ids[m], labels[m], probs[m]
        for scheme in WEIGHTINGS:
            for k, v in binary_metrics(y, p, window_weights(t, y, scheme)).items():
                out[f"win/s{st}/{scheme}/{k}"] = v
        ts = tree_scores(t, y, p)
        for rule, col in RULES.items():
            pred = ts[col] > (0.25 if col == "share" else THRESHOLD)
            for scheme in ("unweighted", "class"):
                w = window_weights(ts.index.to_numpy(), ts.y.to_numpy(), scheme)
                for k, v in binary_metrics(ts.y.to_numpy(), ts[col].to_numpy(), w, pred.to_numpy()).items():
                    out[f"tree/s{st}/{rule}/{scheme}/{k}"] = v
    return out


def _arrays(pred):
    return {t: (d.label.to_numpy(), d.prob.to_numpy(), d.stride.astype(str).to_numpy())
            for t, d in pred.groupby("tree_group", sort=True)}


def draws(pred, n=1000, seed=0):
    """Bootstrap draws: list of tree lists (healthy and infested resampled separately)."""
    lab = pred.groupby("tree_group").label.first().sort_index()
    h, i = lab.index[lab == 0].to_numpy(), lab.index[lab == 1].to_numpy()
    rng = np.random.default_rng(seed)
    return [np.concatenate([rng.choice(h, len(h)), rng.choice(i, len(i))]) for _ in range(n)]


def evaluate_draws(pred, tree_draws):
    arr = _arrays(pred)
    rows = []
    for trees in tree_draws:
        ys, ps, ss, ts = [], [], [], []
        for copy, t in enumerate(trees):
            y, p, s = arr[t]
            ys.append(y), ps.append(p), ss.append(s), ts.append(np.full(len(y), copy))
        rows.append(all_metrics(np.concatenate(ts), np.concatenate(ys), np.concatenate(ps), np.concatenate(ss)))
    return pd.DataFrame(rows)


def point(pred):
    return all_metrics(pred.tree_group.to_numpy(), pred.label.to_numpy(), pred.prob.to_numpy(),
                       pred.stride.astype(str).to_numpy())


def summarize(pred, n=1000, seed=0):
    """Point estimates with 95% percentile intervals."""
    pt = point(pred)
    bs = evaluate_draws(pred, draws(pred, n, seed))
    lo, hi = bs.quantile(0.025), bs.quantile(0.975)
    return pd.DataFrame(dict(value=pd.Series(pt), lo=lo, hi=hi))


def paired(pred_a, pred_b, n=1000, seed=0, keys=None):
    """Difference b - a on the same bootstrap draws (same trees in both)."""
    if sorted(pred_a.tree_group.unique()) != sorted(pred_b.tree_group.unique()):
        raise ValueError("paired comparison needs the same trees")
    d = draws(pred_a, n, seed)
    a, b = evaluate_draws(pred_a, d), evaluate_draws(pred_b, d)
    keys = keys or list(a.columns)
    diff = b[keys] - a[keys]
    pa, pb = point(pred_a), point(pred_b)
    return pd.DataFrame({"a": pd.Series({k: pa[k] for k in keys}), "b": pd.Series({k: pb[k] for k in keys}),
                         "diff": pd.Series({k: pb[k] - pa[k] for k in keys}), "lo": diff.quantile(0.025),
                         "hi": diff.quantile(0.975), "share_b_better": (diff > 0).mean()})
