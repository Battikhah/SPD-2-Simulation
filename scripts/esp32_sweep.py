"""Overnight ESP32 configuration sweep (resumable, one GPU job at a time).

Phase 1: every architecture in esp32_arch.CANDIDATES x pos_boost {0.5, 1, 2}, seeds 0-2, each trained on
the GPU (esp32_train.py) and converted/scored as int8 TFLite (esp32_export.py). Window models are also
scored with on-device top-3-of-10 aggregation. Phase 2: the three best configurations by int8 ensemble
ROC-AUC (threshold-free) are retrained with new seeds 3-5 to check that the ranking is not seed luck.

Development data only; the test partition is never read. Completed configurations are skipped, failures
are logged and the sweep continues. Keeps the PC awake through an app-level SetThreadExecutionState
request (no system settings are changed). Run from the repository root:
  .venv/Scripts/python.exe scripts/esp32_sweep.py
Outputs: outputs/esp32-tflm/sweep/ (sweep.log, logs/, leaderboard.csv).
"""

import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esp32_arch import CANDIDATES  # noqa: E402

ROOT = Path("outputs/esp32-tflm")
SWEEP = ROOT / "sweep"
GPU_PY = Path(".venv/Scripts/python.exe")
TF_PY = Path("outputs/tf-venv/Scripts/python.exe")
BOOSTS = (0.5, 1.0, 2.0)
ARENA_LIMIT = 64 * 1024
# Runs made before the sweep keep their original directories.
LEGACY = {("stream1s_w8", 2.0): "full", ("strided10s_w8", 2.0): "full", ("dscnn10s_res", 2.0): "full",
          ("stream1s_w8", 1.0): "full-pb1.0", ("stream1s_w8", 0.5): "full-pb0.5"}


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with (SWEEP / "sweep.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def other_jobs_running():
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
           "Where-Object { $_.CommandLine -match 'esp32_(train|export)\\.py' } | Measure-Object | "
           "Select-Object -ExpandProperty Count")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
    return int(out.stdout.strip() or 0) > 0


def directory(arch, pb, suffix=""):
    return ROOT / (LEGACY[(arch, pb)] if not suffix and (arch, pb) in LEGACY else f"full-{arch}-pb{pb}{suffix}")


def tag_of(d):
    return d.name[len("full-"):] if d.name.startswith("full-") else ""


def trained(d, arch, seeds):
    f = d / "oof_float.csv"
    return f.exists() and all(f"{arch}|seed{s}" in pd.read_csv(f, nrows=1).columns for s in seeds)


def exported(d, arch):
    f = d / "results.csv"
    if not f.exists():
        return False
    models = set(pd.read_csv(f).model)
    need = [f"{arch}|ensemble"] + ([f"{arch}@top3of10|ensemble"] if CANDIDATES[arch]["unit"] == "window" else [])
    return all(m in models for m in need)


def run(cmd, env_extra, logfile):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # TF venv must not see torch-cu121
    env |= {"PYTHONUTF8": "1", "TF_CPP_MIN_LOG_LEVEL": "2"} | env_extra
    with logfile.open("w", encoding="utf-8") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode


def do_config(arch, pb, seeds=(0, 1, 2), suffix="", extra=()):
    d = directory(arch, pb, suffix)
    name = f"{arch} pb{pb}{suffix}"
    if exported(d, arch):
        log(f"skip {name}: already exported")
        return True
    t0 = time.time()
    logs = SWEEP / "logs"
    if not trained(d, arch, seeds):
        torch_path = str(Path("outputs/torch-cu121").resolve())
        code = run([str(GPU_PY), "scripts/esp32_train.py", "--only", arch, "--pos-boost", str(pb),
                    "--tag", tag_of(d), "--seeds", *map(str, seeds), *extra],
                   {"PYTHONPATH": torch_path}, logs / f"train-{d.name}.log")
        if code:
            log(f"FAILED train {name} (exit {code}); see logs/train-{d.name}.log")
            return False
    code = run([str(TF_PY), "scripts/esp32_export.py", "--tag", tag_of(d)], {}, logs / f"export-{d.name}.log")
    if code:
        log(f"FAILED export {name} (exit {code}); see logs/export-{d.name}.log")
        return False
    log(f"done {name} in {(time.time() - t0) / 60:.1f} min")
    return True


def leaderboard():
    rows = []
    for d in sorted(ROOT.glob("full*")):
        if not (d / "results.csv").exists():
            continue
        res = pd.read_csv(d / "results.csv")
        dev = json.loads((d / "device_report.json").read_text()) if (d / "device_report.json").exists() else {}
        info = json.loads((d / "train_info.json").read_text()) if (d / "train_info.json").exists() else {}
        for r in res.to_dict("records"):
            base, variant = r["model"].split("|")
            arch = base.split("@")[0]
            if arch not in CANDIDATES:
                continue
            pb = next((p for (a, p), dd in LEGACY.items() if dd == d.name and a == arch), info.get("pos_boost"))
            rows.append(dict(directory=d.name, arch=arch, pos_boost=pb, teacher=info.get("teacher") or "",
                             alpha=info.get("alpha"), epochs=info.get("epochs"),
                             balance_folders=info.get("balance_folders") or 0.0,
                             seeds="-".join(map(str, info.get("seeds", []))),
                             aggregation="top3of10" if "@top3of10" in base else "mean", variant=variant,
                             int8_tflite_bytes=dev.get(arch, {}).get("int8_tflite_bytes"),
                             arena_lower_bound_bytes=dev.get(arch, {}).get("arena_lower_bound_bytes")) | r)
    if rows:
        out = pd.DataFrame(rows).sort_values(["arithmetic", "roc_auc"], ascending=[True, False])
        out.to_csv(SWEEP / "leaderboard.csv", index=False)
        return out
    return pd.DataFrame()


def sweep_running_elsewhere():
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
           f"Where-Object {{ $_.CommandLine -match 'esp32_sweep\\.py' -and $_.ProcessId -ne {os.getpid()} -and "
           "$_.CommandLine -notmatch '--phase3' } | Measure-Object | Select-Object -ExpandProperty Count")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
    return int(out.stdout.strip() or 0) > 0


def phase3(deadline_hour=7, deadline_minute=15):
    """Improvement queue after phases 1-2: distillation, longer training and extra seeds, in priority
    order. No job starts after the deadline so results are ready for the morning report."""
    while sweep_running_elsewhere() or other_jobs_running():
        time.sleep(120)
    board = leaderboard()
    # Only models whose activation arena fits the 64 KiB arena of firmware/esp32_stream1s are improved.
    base = board[(board.arithmetic == "int8") & (board.variant == "ensemble") & (board.teacher == "")
                 & (board.balance_folders == 0) & ~board.directory.str.contains("-s345")
                 & (board.arena_lower_bound_bytes <= ARENA_LIMIT)]
    best = base.sort_values("roc_auc", ascending=False).drop_duplicates(["arch", "pos_boost"])
    win = best[best.arch.map(lambda a: CANDIDATES[a]["unit"] == "window")]
    seg = best[best.arch.map(lambda a: CANDIDATES[a]["unit"] == "segment")]
    w1, w2, s1 = win.iloc[0], win[win.arch != win.iloc[0].arch].iloc[0], seg.iloc[0]
    overall = best.iloc[0]
    kd1, kd10 = ("--teacher", "teacher1s_w32"), ("--teacher", "teacher10s_w32")
    bal = lambda p: ("--balance-folders", str(p))  # noqa: E731
    # False alarms concentrate in a few clean folders (wind/rain, same-device), so folder balancing leads.
    jobs = [
        (w1.arch, w1.pos_boost, (0, 1, 2), "-bal0.5", bal(0.5)),
        (w1.arch, w1.pos_boost, (0, 1, 2), "-bal1", bal(1)),
        (w2.arch, w2.pos_boost, (0, 1, 2), "-bal0.5", bal(0.5)),
        (w1.arch, w1.pos_boost, (0, 1, 2), "-kd", kd1),
        (w1.arch, w1.pos_boost, (0, 1, 2), "-kd-bal0.5", kd1 + bal(0.5)),
        (s1.arch, s1.pos_boost, (0, 1, 2), "-bal0.5", bal(0.5)),
        (w2.arch, w2.pos_boost, (0, 1, 2), "-kd", kd1),
        (w1.arch, w1.pos_boost, (0, 1, 2), "-kd-a0.2", kd1 + ("--alpha", "0.2")),
        (s1.arch, s1.pos_boost, (0, 1, 2), "-kd", kd10),
        (w1.arch, w1.pos_boost, (0, 1, 2), "-e40", ("--epochs", "40")),
        (overall.arch, overall.pos_boost, (6, 7, 8, 9, 10), "-s6to10", ()),
        (w1.arch, 0.5, (0, 1, 2), "-bal1", bal(1)),
        (w1.arch, w1.pos_boost, (0, 1, 2), "-kd-e40", kd1 + ("--epochs", "40")),
    ]
    log("phase 3 start; queue: " + "; ".join(f"{a} pb{p}{s}" for a, p, _, s, _ in jobs))
    for arch, pb, seeds, suffix, extra in jobs:
        now = datetime.now()
        if 3 <= now.hour < 12 and (now.hour, now.minute) >= (deadline_hour, deadline_minute):
            log(f"phase 3 deadline reached; not starting {arch} pb{pb}{suffix}")
            break
        do_config(arch, float(pb), seeds=seeds, suffix=suffix, extra=extra)
        leaderboard()
    log("phase 3 complete")


def main():
    (SWEEP / "logs").mkdir(parents=True, exist_ok=True)
    if "--phase3" in sys.argv:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        try:
            phase3()
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        return
    if "--dry-run" in sys.argv:
        for arch in CANDIDATES:
            for pb in BOOSTS:
                d = directory(arch, pb)
                print(f"{arch:16s} pb{pb}: {d.name:32s} trained={trained(d, arch, (0, 1, 2))} "
                      f"exported={exported(d, arch)}")
        return
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | SYSTEM_REQUIRED
    try:
        while other_jobs_running():
            log("waiting for a running esp32_train/esp32_export job to finish")
            time.sleep(60)
        log("phase 1 start: %d architectures x pos_boost %s" % (len(CANDIDATES), BOOSTS))
        legacy = [(a, p) for (a, p) in LEGACY]
        new = [(a, p) for a in CANDIDATES for p in BOOSTS if (a, p) not in LEGACY]
        for arch, pb in new + legacy:
            do_config(arch, pb)
            leaderboard()
        board = leaderboard()
        ens = board[(board.arithmetic == "int8") & (board.variant == "ensemble") & board.directory.str.match(r"^full")]
        ens = ens[~ens.directory.str.endswith("-s345")]
        top = ens.sort_values("roc_auc", ascending=False).drop_duplicates(["arch", "pos_boost"]).head(3)
        log("phase 2 (new seeds 3-5) for: " + "; ".join(f"{r.arch} pb{r.pos_boost} {r.aggregation} AUC {r.roc_auc:.3f}"
                                                         for r in top.itertuples()))
        for r in top.itertuples():
            do_config(r.arch, float(r.pos_boost), seeds=(3, 4, 5), suffix="-s345")
            leaderboard()
        log("sweep complete")
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
