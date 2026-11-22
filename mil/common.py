"""Shared definitions for the tree-level MIL experiments: paths, the development clip index, tree groups,
folds and window positions. Sealed groups (mil/groups.json) are never indexed, so their audio is never read.

Windows are built per folder (one device deployment = one recording stream, in filename/time order). Tree
groups (which may merge several folders of the same tree) decide folds, weights and tree-level verdicts.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / ".cache/kagglehub/datasets/potamitis/treevibes/versions/1"
MANIFEST = Path("outputs/real-training-cuda/full/treevibes-984e68c4e99668c8/splits.csv")
MIL = Path("mil")
OUT = Path("outputs/mil")
CACHE = OUT / "cache"
EXP = OUT / "experiments"
TB = OUT / "tb"
GROUPS_JSON = MIL / "groups.json"
FOLDS_JSON = MIL / "folds.json"

SR, SECONDS, K = 8000, 20, 24
N_SAMPLES = SR * SECONDS
N_MELS, N_FRAMES = 64, 626
NAME = re.compile(r"F_(\d{14})_(\d+)(?:_T(-?[\d.]+))?\.wav$", re.I)


def load_groups():
    g = json.loads(GROUPS_JSON.read_text(encoding="utf-8"))
    if not g["status"].startswith("confirmed"):
        raise SystemExit("mil/groups.json is not confirmed; do not train on it")
    return g


def dev_groups():
    """{group: (label, [folders])} for development (non-sealed) groups."""
    return {k: (v["label"], v["folders"]) for k, v in load_groups()["groups"].items() if not v.get("sealed")}


def sealed_folders():
    return {f for v in load_groups()["groups"].values() if v.get("sealed") for f in v["folders"]}


def clip_index():
    """Development clips in feature-row order, deduplicated (no recording counts twice)."""
    groups = dev_groups()
    folder_group = {f: g for g, (_, fs) in groups.items() for f in fs}
    folder_label = {f: groups[g][0] for f, g in folder_group.items()}
    m = pd.read_csv(MANIFEST)
    m["folder"] = m.path.str.extract(r"(folder_\d+)")[0]
    m = m[m.folder.isin(folder_group)]
    m = m[~m.path.str.contains("/folder_9/folder_9/", regex=False)]  # byte copies of folder_9 files
    if m.folder.isin(sealed_folders()).any() or (m.split == "test").any():
        raise SystemExit("sealed recordings reached the MIL index")
    m = m.sort_values("path")
    n0 = len(m)
    m = m.drop_duplicates("pcm_sha256", keep="first")  # identical audio across or within folders
    dropped = n0 - len(m)
    parsed = m.path.str.extract(r"F_(\d{14})_(\d+)")
    m["time"] = [datetime.strptime(t, "%Y%m%d%H%M%S") for t in parsed[0]]
    m["file_index"] = parsed[1].astype(int)
    m["group"] = m.folder.map(folder_group)
    m["label"] = m.folder.map(folder_label)
    bad = m[m.label != m.label.astype(int)]
    assert bad.empty
    m = m.sort_values(["group", "folder", "time", "file_index", "path"]).reset_index(drop=True)
    m.attrs["dropped_duplicates"] = dropped
    return m[["path", "folder", "group", "label", "time", "file_index", "pcm_sha256"]]


def load_index():
    df = pd.read_csv(CACHE / "clips.csv", parse_dates=["time"])
    return df


def make_folds(index):
    """StratifiedGroupKFold(10, shuffle, 42) over tree groups (one row per group), saved once and reused."""
    from sklearn.model_selection import StratifiedGroupKFold

    g = index.groupby("group").label.first().sort_index()
    if FOLDS_JSON.exists():
        saved = json.loads(FOLDS_JSON.read_text())
        if sorted(saved["fold_of_group"]) != list(g.index):
            raise SystemExit("mil/folds.json does not match the current groups; never silently refold")
        return {k: int(v) for k, v in saved["fold_of_group"].items()}
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 5 healthy groups < 10 splits: some folds have no healthy group
        splits = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42).split(g.index, g.values, g.index)
        fold = {grp: k for k, (_, te) in enumerate(splits) for grp in g.index[te]}
    FOLDS_JSON.write_text(json.dumps(dict(
        method="StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42), one row per tree group, "
               "stratified by label", fold_of_group=fold), indent=1))
    return fold


def recorder_folds():
    """Tree-safe leave-one-recorder-out folds: fold k holds out every development folder recorded by recorder
    k; training uses only tree groups with no folder on that recorder (a group spanning two recorders is never
    split between training and test). Returns (sorted recorders, {folder: recorder}); recorder = last 4 IMEI
    digits from TreeVibes.xlsx."""
    wb = pd.read_excel(ROOT / "TreeVibes.xlsx", dtype={"IMEI": str})
    wb = wb[wb.FOLDER.astype(str).str.fullmatch(r"folder_\d+")]
    dev = {f for _, fs in dev_groups().values() for f in fs}
    f2r = {f: str(i)[-4:] for f, i in zip(wb.FOLDER, wb.IMEI) if f in dev}
    assert set(f2r) == dev and all(r != "nan" for r in f2r.values()), "development folder without a recorder"
    return sorted(set(f2r.values())), f2r


def make_site_folds(index):
    """Sensitivity check: StratifiedGroupKFold(10, shuffle, 42) with whole 25 m site groups held out together
    (groups.json site_groups_25m). Returns fold per tree group; saved once to mil/folds_site25.json."""
    import warnings

    from sklearn.model_selection import StratifiedGroupKFold

    path = MIL / "folds_site25.json"
    g = index.groupby("group").agg(label=("label", "first"), folder=("folder", "first")).sort_index()
    if path.exists():
        return {k: int(v) for k, v in json.loads(path.read_text())["fold_of_group"].items()}
    site_of = {f: s for s, fs in load_groups()["site_groups_25m"]["groups"].items() for f in fs}
    sites = g.folder.map(site_of)
    assert sites.notna().all(), "a tree group has no 25 m site"
    for grp, d in index.groupby("group"):  # every folder of a tree group must share one site
        assert d.folder.map(site_of).nunique() == 1, grp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        splits = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42).split(g.index, g.label, sites)
        fold = {grp: k for k, (_, te) in enumerate(splits) for grp in g.index[te]}
    path.write_text(json.dumps(dict(method="StratifiedGroupKFold(10, shuffle=True, random_state=42) over tree "
                                           "groups, grouped by 25 m site", site_of_group=sites.to_dict(),
                                    fold_of_group=fold), indent=1))
    return fold


def folder_sequences(index, groups=None):
    """[{group, folder, label, rows, hours}] with rows in time order, for the given groups."""
    out = []
    sel = index if groups is None else index[index.group.isin(groups)]
    for (grp, folder), d in sel.groupby(["group", "folder"], sort=True):
        t = d.time.to_numpy("datetime64[s]").astype(np.int64) / 3600.0
        out.append(dict(group=grp, folder=folder, label=int(d.label.iloc[0]), rows=d.index.to_numpy(),
                        hours=t - t[0], times=d.time.to_numpy()))
    return out


def lab_sequences(offset):
    """Lab pseudo-clips (outputs/mil/cache/lab_clips.csv, rows offset.. of the *_plus caches) as two training-only
    sequences, lab_clean (0) and lab_infested (1). They have no trees or real times; a 5-minute spacing is assigned
    so skip windows draw K random time-ordered pseudo-clips within a stretch."""
    lab = pd.read_csv(CACHE / "lab_clips.csv")
    out = []
    for label, name in ((0, "lab_clean"), (1, "lab_infested")):
        rows = offset + np.flatnonzero(lab.label.to_numpy() == label)
        n = len(rows)
        out.append(dict(group=name, folder=name, label=label, rows=rows, hours=np.arange(n) * 5 / 60,
                        times=np.datetime64("2000-01-01T00:00:00") + np.arange(n) * np.timedelta64(300, "s")))
    return out


def window_starts(n, stride, k=K):
    """Start positions of evaluation windows over a sequence of n clips; n < k gives one window of all n."""
    if n < k:
        return [0]
    starts = list(range(0, n - k + 1, stride))
    return starts
