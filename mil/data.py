"""Training bag stream for DataLoader workers (picklable; arrays opened per worker with mmap).

Each item is one training step: `per_class` healthy bags then `per_class` infested bags, plus labels and bag
types (0 healthy, 1 hard healthy, 2 real infested, 3 synthetic infested). A bag: pick a training tree group of
the class uniformly, a folder of that group with probability proportional to its clip count, then K
consecutive clips from a random start (rejecting windows spanning more than `max_span_h`, up to `tries`
attempts, then accepting the last). Folders shorter than K repeat their clips to fill K (np.resize).

CNN inputs without augmentation use the cached per-clip z-scored log-mel (B0 exactly). With augmentation the
raw dB log-mel is processed per clip in this order (handoff 7.5): mixing (A2 noise, A1 lab pulses) -> random
circular time shift -> EQ tilt (A4) -> per-clip z-score -> SpecAugment (A5). Mixing is in the power domain.
Embedding inputs (E-* runs) are gathered as (K, D) with no augmentation.
"""

import json
import os

import front
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


def mix_db(a, b, g):
    return 10 * np.log10(10 ** (a / 10) + 10 ** ((b + g) / 10) + 1e-12)


class BagStream(IterableDataset):
    def __init__(self, feat_path, sequences, cfg, seed, fold, raw_path=None, bank_path=None, hard_path=None):
        super().__init__()
        self.feat_path, self.raw_path, self.bank_path = str(feat_path), raw_path and str(raw_path), bank_path and str(bank_path)
        self.hard_path = hard_path and str(hard_path)
        self.cfg, self.seed, self.fold = dict(cfg), seed, fold
        self.segment = 0  # loader segment (train.py restarts workers every LOADER_STEPS steps)
        self.k, self.per_class = cfg["k"], cfg["per_class"]
        self.emb = cfg["input"].startswith("emb")  # MFCC maps take the CNN path (circular time shift, no aug)
        self.augment = (not self.emb) and any(cfg.get(a) for a in ("noise", "lab", "tilt", "specaug", "randeq"))
        self.front = cfg.get("front", "z")  # front end applied after augmentation (mil/front.py)
        by_class = {0: {}, 1: {}}
        for s in sequences:
            by_class[s["label"]].setdefault(s["group"], []).append((np.asarray(s["rows"]), np.asarray(s["hours"])))
        self.groups = {c: [by_class[c][g] for g in sorted(by_class[c])] for c in (0, 1)}
        self.group_names = {c: sorted(by_class[c]) for c in (0, 1)}
        # noise sources: healthy training clips, by group (a clip from another healthy tree is mixed in)
        self.healthy_rows = {g: np.concatenate([r for r, _ in by_class[0][g]]) for g in by_class[0]}
        assert all(self.groups[c] for c in (0, 1)), "training fold lacks a class"

    # ---- windows ----
    def window(self, rows, hours, rng):
        n, k = len(rows), self.k
        if n < k:
            return np.resize(rows, k)
        if self.cfg["skip"]:  # K recordings in time order from a random stretch spanning at most skip_span_h
            for _ in range(self.cfg["tries"]):
                s = int(rng.integers(0, n - k + 1))
                cand = s + np.flatnonzero(hours[s:] - hours[s] <= self.cfg["skip_span_h"])
                if len(cand) >= k:
                    return rows[np.sort(rng.choice(cand, k, replace=False))]
        for _ in range(self.cfg["tries"]):
            s = int(rng.integers(0, n - k + 1))
            if hours[s + k - 1] - hours[s] <= self.cfg["max_span_h"]:
                break
        return rows[s:s + k]

    def draw(self, c, rng):
        gi = int(rng.integers(len(self.groups[c])))
        folders = self.groups[c][gi]
        sizes = np.array([len(r) for r, _ in folders], float)
        rows, hours = folders[int(rng.choice(len(folders), p=sizes / sizes.sum()))]
        return self.window(rows, hours, rng), self.group_names[c][gi]

    # ---- hard negatives (A3): the main process rewrites the file after every epoch ----
    def hard_list(self):
        if not self.hard_path or not os.path.exists(self.hard_path):
            return []
        m = os.path.getmtime(self.hard_path)
        if m != self._hard_mtime:
            try:
                self._hard = [np.asarray(w) for w in json.loads(open(self.hard_path).read())["windows"]]
                self._hard_mtime = m
            except (OSError, ValueError, KeyError):
                pass
        return self._hard

    # ---- per-clip processing ----
    def process(self, w, group, synthetic, rng):
        if self.emb:
            return np.asarray(self.feat[w], np.float32)
        if not self.augment:
            x = np.asarray(self.feat[w])
            shifts = rng.integers(0, x.shape[-1], len(x))
            return np.stack([np.roll(clip, int(s), axis=-1) for clip, s in zip(x, shifts)])
        x = np.asarray(self.raw[w], np.float32)
        cfg = self.cfg
        if synthetic:  # A1: lab pulses into 1-3 clips of a healthy window
            for i in rng.choice(len(x), int(rng.integers(1, 4)), replace=False):
                pulse = np.asarray(self.bank[int(rng.integers(len(self.bank)))], np.float64)
                field = 10 ** (x[i].astype(np.float64) / 10)
                target = np.median(field.sum(0)) * 10 ** (rng.uniform(3, 20) / 10)
                gain = target / max(pulse.sum(0).max(), 1e-30)
                off = int(rng.integers(0, field.shape[1] - pulse.shape[1] + 1))
                field[:, off:off + pulse.shape[1]] += gain * pulse
                x[i] = 10 * np.log10(field + 1e-12)
        if cfg["noise"]:  # A2: another healthy training tree's clip, p = 0.3 per clip, gain -12..0 dB
            others = [g for g in self.healthy_rows if g != group] or list(self.healthy_rows)
            for i in range(len(x)):
                if rng.random() < 0.3:
                    src = self.healthy_rows[others[int(rng.integers(len(others)))]]
                    x[i] = mix_db(x[i], np.asarray(self.raw[int(src[int(rng.integers(len(src)))])], np.float32),
                                  rng.uniform(-12, 0))
        shifts = rng.integers(0, x.shape[-1], len(x))
        x = np.stack([np.roll(clip, int(s), axis=-1) for clip, s in zip(x, shifts)])
        if cfg["tilt"]:  # A4: linear ramp over mel bands, +/-3 dB at the outermost bands
            ramp = np.linspace(-1, 1, x.shape[1], dtype=np.float32)[None, :, None]
            x = x + rng.uniform(-3, 3, (len(x), 1, 1)).astype(np.float32) * ramp
        if cfg.get("randeq"):  # device-response proxy: random smooth EQ curve per clip, 6 knots in +/-6 dB
            knots = rng.uniform(-6, 6, (len(x), 6)).astype(np.float32)
            grid = np.linspace(0, 5, x.shape[1], dtype=np.float32)
            x = x + np.stack([np.interp(grid, np.arange(6), kn) for kn in knots])[:, :, None]
        x = front.apply(x, self.front)  # per-clip z-score (B0), frequency centering or PCEN
        if cfg["specaug"]:  # A5: one band span (1-8 mels) and one time span (1-60 frames) zeroed per clip
            for i in range(len(x)):
                fw, tw = int(rng.integers(1, 9)), int(rng.integers(1, 61))
                f0, t0 = int(rng.integers(0, x.shape[1] - fw + 1)), int(rng.integers(0, x.shape[2] - tw + 1))
                x[i, f0:f0 + fw] = 0
                x[i, :, t0:t0 + tw] = 0
        return x.astype(np.float16)

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info is not None else 0
        key = [self.seed, self.fold, wid] + ([self.segment] if self.segment else [])  # segment 0 = original stream
        rng = np.random.default_rng(key)
        self.feat = np.load(self.feat_path, mmap_mode="r")
        self.raw = np.load(self.raw_path, mmap_mode="r") if self.raw_path else None
        self.bank = np.load(self.bank_path, mmap_mode="r") if self.bank_path else None
        self._hard, self._hard_mtime = [], None
        labels = torch.tensor([0.0] * self.per_class + [1.0] * self.per_class)
        while True:
            bags, types = [], []
            hard = self.hard_list() if self.cfg["hardneg"] else []
            for _ in range(self.per_class):  # healthy bags
                if hard and rng.random() < 0.5:
                    w, g, t = hard[int(rng.integers(len(hard)))], None, 1
                else:
                    (w, g), t = self.draw(0, rng), 0
                bags.append(self.process(w, g, False, rng))
                types.append(t)
            for _ in range(self.per_class):  # infested bags
                if self.cfg["lab"] and rng.random() < 0.5:
                    w, g = self.draw(0, rng)  # a healthy training window becomes a synthetic infested bag
                    bags.append(self.process(w, g, True, rng))
                    types.append(3)
                else:
                    w, g = self.draw(1, rng)
                    bags.append(self.process(w, g, False, rng))
                    types.append(2)
            yield torch.from_numpy(np.stack(bags)), labels, torch.tensor(types)
