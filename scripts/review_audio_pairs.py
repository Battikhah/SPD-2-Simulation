"""Examine development-only suspected duplicate pairs without accessing holdout audio."""
import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import correlate, correlation_lags

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/real-training-cuda/full/treevibes-984e68c4e99668c8'
DATA = Path(os.environ['USERPROFILE']) / '.cache/kagglehub/datasets/potamitis/treevibes/versions/1'


def best_waveform_match(a, b, min_overlap):
    """Peak absolute Pearson correlation at any lag with sufficient shared samples."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    lags = correlation_lags(len(a), len(b), mode='full')
    starts_a, starts_b = np.maximum(lags, 0), np.maximum(-lags, 0)
    sizes = np.minimum(len(a) - starts_a, len(b) - starts_b)
    valid = sizes >= min_overlap
    lags, starts_a, starts_b, sizes = (x[valid] for x in (lags, starts_a, starts_b, sizes))
    if not len(sizes):
        return None, None, None
    cross = correlate(a, b, mode='full', method='fft')[valid]
    def sums(x, starts, lengths):
        cs = np.concatenate(([0.], np.cumsum(x)))
        return cs[starts + lengths] - cs[starts]
    sa, sb = sums(a, starts_a, sizes), sums(b, starts_b, sizes)
    ea = sums(a*a, starts_a, sizes) - sa*sa/sizes
    eb = sums(b*b, starts_b, sizes) - sb*sb/sizes
    denominator = np.sqrt(np.maximum(ea, 0) * np.maximum(eb, 0))
    scores = np.divide(np.abs(cross-sa*sb/sizes), denominator, out=np.zeros_like(cross), where=denominator > 1e-12)
    if not scores.any():
        return None, None, None
    i = int(np.argmax(scores))
    return float(min(scores[i], 1.0)), int(sizes[i]), int(lags[i])


def main():
    manifest = pd.read_csv(RUN/'manifest.csv').merge(pd.read_csv(RUN/'splits.csv')[['path', 'split']], on='path', validate='one_to_one').set_index('path')
    candidates = pd.read_csv(RUN/'near_duplicate_candidates.csv')
    candidates = candidates[candidates.path_a.map(manifest.split) != candidates.path_b.map(manifest.split)]
    assert len(candidates) == 65
    @lru_cache(maxsize=None)
    def read(relative):
        row = manifest.loc[relative]
        if row['split'] not in {'train', 'val'}:
            raise ValueError('Refusing to read non-development audio')
        file = (DATA / relative).resolve()
        if not file.is_relative_to(DATA.resolve()):
            raise ValueError('Unsafe source path')
        if hashlib.sha256(file.read_bytes()).hexdigest() != row.file_sha256:
            raise ValueError('Source hash mismatch: ' + relative)
        x, sr = sf.read(file, dtype='float64', always_2d=True)
        return x.mean(axis=1), sr
    output = []
    for row in candidates.itertuples(index=False):
        a, sr_a = read(row.path_a)
        b, sr_b = read(row.path_b)
        if sr_a != sr_b:
            raise ValueError('Different sampling rates: manual review required')
        score, overlap, lag = best_waveform_match(a, b, 5 * sr_a)
        ma, mb = manifest.loc[row.path_a], manifest.loc[row.path_b]
        dates = [re.search(r'F_(\d{14})', p) for p in (row.path_a, row.path_b)]
        output.append(dict(path_a=row.path_a, path_b=row.path_b, split_a=ma['split'], split_b=mb['split'],
                           group_a=ma.group_id, group_b=mb.group_id, label_a=int(ma.label), label_b=int(mb.label),
                           filename_time_a=dates[0].group(1) if dates[0] else '', filename_time_b=dates[1].group(1) if dates[1] else '',
                           same_pcm_hash=ma.pcm_sha256 == mb.pcm_sha256, fingerprint_distance=row.distance,
                           max_abs_waveform_correlation=score, overlap_seconds=overlap/sr_a if overlap else None,
                           lag_seconds=lag/sr_a if lag is not None else None,
                           decision='UNREVIEWED', evidence='Automated waveform comparison; ancestry and transforms require review'))
    out = ROOT/'outputs/acoustic-validity-audit/cross_split_pair_evidence.csv'
    pd.DataFrame(output).to_csv(out, index=False)
    print('Pairs:', len(output), 'unique recordings:', read.cache_info().misses, 'output:', out)
    print(pd.DataFrame(output).max_abs_waveform_correlation.describe().to_string())


if __name__ == '__main__':
    main()
