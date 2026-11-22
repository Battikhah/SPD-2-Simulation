"""Section 3 data audit for the tree-level MIL handoff (no model, no features, no scores).

Reads only directory listings, filenames, raw file bytes (MD5) and TreeVibes.xlsx. Audio samples are
never decoded, so the sealed test partition of the existing locked split is not loaded for modelling.
Writes CSV tables to mil/audit/ and prints a summary used for mil/DATA_REPORT.md.

    .venv/Scripts/python.exe mil/data_audit.py
"""

import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path.home() / ".cache/kagglehub/datasets/potamitis/treevibes/versions/1"
SPLIT = Path("outputs/real-training-cuda/full/treevibes-984e68c4e99668c8/splits.csv")
OUT = Path("mil/audit")
NAME = re.compile(r"F_(\d{14})_(\d+)(?:_T(-?[\d.]+))?\.wav$", re.I)  # temperature is missing in some folders


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metres(a, b):
    """Haversine distance in metres between (lat, lon) pairs."""
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(ROOT.rglob("*.wav")):
        rel = p.relative_to(ROOT).as_posix()
        m = re.findall(r"folder_\d+", rel)
        t = NAME.search(p.name)
        rows.append(dict(path=rel, area=rel.split("/")[0], directory=rel.rsplit("/", 1)[0],
                         folder=m[0] if m else None, nested=len(m) > 1, name=p.name, bytes=p.stat().st_size,
                         time=datetime.strptime(t.group(1), "%Y%m%d%H%M%S") if t else None,
                         index=int(t.group(2)) if t else None,
                         temp=float(t.group(3)) if t and t.group(3) else None,
                         md5=md5(p) if rel.startswith("field/") else None))  # lab is never evaluated
        if len(rows) % 5000 == 0:
            print(f"hashed {len(rows)}", flush=True)
    files = pd.DataFrame(rows)
    files.to_csv(OUT / "files.csv", index=False)

    # 1. directories
    dirs = files.groupby("directory").size().rename("wavs").reset_index()
    dirs.to_csv(OUT / "directories.csv", index=False)
    print("\n== 1. directories\n" + dirs.to_string(index=False))
    print("filenames not matching F_<14 digits>_<n>_T<temp>.wav:", int(files.time.isna().sum()),
          files.loc[files.time.isna(), "directory"].value_counts().to_dict())

    # 2. workbook vs disk
    wb = pd.read_excel(ROOT / "TreeVibes.xlsx", dtype={"IMEI": str})
    wb = wb[wb.FOLDER.astype(str).str.fullmatch(r"folder_\d+")].copy()
    field = files[files.area == "field"]
    disk = field.groupby("folder").agg(wavs=("path", "size"), dir_label=("directory", lambda d: sorted(
        {"infested" if "/infested" in x else "clean" if "/clean" in x else "test(unlabelled dir)" for x in d})))
    match = wb.set_index("FOLDER")[["IMEI", "GPS_LAT", "GPS_LONG", "VISUAL", "AUDIO", "CONFIRMATION", "COMMENT"]]
    match = match.join(disk, how="outer")
    match["on_disk"] = match.wavs.notna()
    match["in_workbook"] = match.AUDIO.notna()
    split = pd.read_csv(SPLIT)
    split["folder"] = split.path.str.extract(r"(folder_\d+)")[0]
    match["locked_split"] = split.groupby("folder").split.agg(lambda s: "/".join(sorted(set(s))))
    match = match.loc[sorted(match.index, key=lambda f: int(f.split("_")[1]))]
    match.to_csv(OUT / "workbook_vs_disk.csv")
    print("\n== 2. workbook vs disk\n" + match.drop(columns=["COMMENT"]).to_string())

    # 3. duplicates
    hashed = files.dropna(subset=["md5"])
    dup = hashed[hashed.duplicated("md5", keep=False)].sort_values(["md5", "path"])
    dup.to_csv(OUT / "duplicate_md5.csv", index=False)
    groups = dup.groupby("md5")
    across = [(h, sorted(set(g.folder.fillna(g.directory)))) for h, g in groups if g.folder.fillna(g.directory).nunique() > 1]
    print(f"\n== 3. identical files: {len(dup)} files in {groups.ngroups} MD5 groups; "
          f"{groups.ngroups - len(across)} groups within one folder/directory, {len(across)} across")
    print("cross-folder MD5 groups by folder set:", pd.Series([" + ".join(a) for _, a in across]).value_counts().to_dict())
    nested = files[files.nested]
    if len(nested):
        parent = files[(files.folder == nested.folder.iloc[0]) & ~files.nested]
        print(f"nested {nested.directory.iloc[0]}: {len(nested)} files; identical bytes in parent folder: "
              f"{nested.md5.isin(parent.md5).sum()}; same filename in parent: {nested.name.isin(parent.name).sum()}")
    fn = field.groupby("name").folder.nunique()
    shared = field[field.name.isin(fn[fn > 1].index)].sort_values("name")
    shared.to_csv(OUT / "shared_filenames.csv", index=False)
    print("field filenames appearing in >1 folder:", int((fn > 1).sum()),
          shared.groupby("name").folder.agg(lambda f: " + ".join(sorted(set(f)))).value_counts().to_dict())
    # PCM-level duplicates from the existing manifest (computed when the manifest was built)
    pcm = split[split.duplicated("pcm_sha256", keep=False)]
    print("manifest PCM-identical recordings:", len(pcm), "| across folders:",
          int(pcm.groupby("pcm_sha256").folder.nunique().gt(1).sum()))

    # 5. timing per folder (before 4 so pairs can report overlap)
    tm = field.dropna(subset=["time"]).sort_values("time").groupby("folder")
    timing = []
    for f, g in tm:
        gap = g.time.diff().dt.total_seconds().div(3600).dropna()
        timing.append(dict(folder=f, n=len(g), first=g.time.min(), last=g.time.max(),
                           span_days=(g.time.max() - g.time.min()).total_seconds() / 86400,
                           median_gap_h=gap.median() if len(gap) else None, max_gap_h=gap.max() if len(gap) else None,
                           gaps_over_7d=int((gap > 168).sum()), gaps_over_36h=int((gap > 36).sum()),
                           distinct_days=g.time.dt.date.nunique(),
                           duplicate_timestamps=int(g.time.duplicated().sum())))
    timing = pd.DataFrame(timing).set_index("folder")
    timing = timing.loc[sorted(timing.index, key=lambda f: int(f.split("_")[1]))]
    timing.to_csv(OUT / "timing.csv")
    print("\n== 5. timing per folder\n" + timing.round(2).to_string())

    # 4. nearby folder pairs
    coords = wb.dropna(subset=["GPS_LAT", "GPS_LONG"]).set_index("FOLDER")
    pairs = []
    for a, b in combinations(coords.index, 2):
        d = metres((coords.GPS_LAT[a], coords.GPS_LONG[a]), (coords.GPS_LAT[b], coords.GPS_LONG[b]))
        if d < 25:
            ta, tb = timing.loc[a] if a in timing.index else None, timing.loc[b] if b in timing.index else None
            overlap = None
            if ta is not None and tb is not None:
                overlap = max(0.0, (min(ta["last"], tb["last"]) - max(ta["first"], tb["first"])).total_seconds() / 86400)
            pairs.append(dict(a=a, b=b, metres=round(d, 1), same_device=coords.IMEI[a] == coords.IMEI[b],
                              label_a=coords.AUDIO[a], label_b=coords.AUDIO[b], overlap_days=overlap,
                              split_a=match.locked_split.get(a), split_b=match.locked_split.get(b)))
    pairs = pd.DataFrame(pairs).sort_values("metres")
    pairs.to_csv(OUT / "near_pairs.csv", index=False)
    print("\n== 4. folder pairs < 25 m\n" + pairs.to_string(index=False))

    # 6. connected components of the < 25 m graph (the "merge when in doubt" grouping)
    parent = {f: f for f in wb.FOLDER}

    def find(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f

    for a, b in zip(pairs.a, pairs.b):
        parent[find(a)] = find(b)
    comp = defaultdict(list)
    for f in wb.FOLDER:
        comp[find(f)].append(f)
    comps = sorted((sorted(v, key=lambda f: int(f.split("_")[1])) for v in comp.values()),
                   key=lambda v: int(v[0].split("_")[1]))
    (OUT / "components_25m.json").write_text(json.dumps(comps, indent=1))
    print("\n== 6. components of the < 25 m graph")
    lab = wb.set_index("FOLDER").AUDIO
    for c in comps:
        print(f"  {' + '.join(c)}  labels={sorted(set(lab[c]))}")


if __name__ == "__main__":
    main()
