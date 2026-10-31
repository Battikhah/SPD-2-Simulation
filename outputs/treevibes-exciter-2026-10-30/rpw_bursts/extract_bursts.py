"""Extract high-contrast transients under the user's RPW assumption; no identity classifier."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
from scipy import ndimage, signal

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent
RATE = 8000
WINDOW = 80
HOP = 40
PARAMS = dict(band_hz=[100, 2500], butterworth_n=4, frame_ms=10, hop_ms=5,
              high_db=12, low_db=6, peak_db=18, merge_gap_ms=40,
              context_ms=30, minimum_active_ms=10, maximum_event_ms=1500,
              noise_frame_quantile=.3, spectral_oversubtraction=2., maximum_per_source=3,
              maximum_exports=2000, edge_exclusion_ms=100)
SOS = signal.butter(4, [100, 2500], btype='bandpass', fs=RATE, output='sos')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(x):
    """Return denoised full signal, padded event intervals, and exploratory quiet-frame RMS."""
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all() or len(x) < RATE:
        raise ValueError('Expected finite mono audio of at least one second')
    bp = signal.sosfiltfilt(SOS, x - np.mean(x))
    frames = np.lib.stride_tricks.sliding_window_view(bp, WINDOW)[::HOP]
    powers = np.mean(frames * frames, axis=1)
    baseline = max(float(np.quantile(powers, .3)), 1e-14)
    high = powers > baseline * 10 ** (PARAMS['high_db'] / 10)
    low = powers > baseline * 10 ** (PARAMS['low_db'] / 10)
    # Boundary filter transients must not become events.
    edge = int(.1 * RATE / HOP)
    low[:edge] = low[-edge:] = False
    high[:edge] = high[-edge:] = False
    labels, count = ndimage.label(low)
    intervals = []
    for label in range(1, count + 1):
        idx = np.flatnonzero(labels == label)
        if high[idx].any():
            intervals.append((int(idx[0] * HOP), int(idx[-1] * HOP + WINDOW)))
    merged = []
    for start, end in intervals:
        if merged and start - merged[-1][1] <= int(.04 * RATE):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    events = []
    for start, end in merged:
        active_ms = (end - start) / RATE * 1000
        local = powers[start // HOP:min(len(powers), end // HOP + 1)]
        contrast = float(10 * np.log10(max(float(np.max(local)), 1e-20) / baseline))
        if 10 <= active_ms <= 1500 and contrast >= PARAMS['peak_db']:
            events.append(dict(start_sample=max(0, start - 240), end_sample=min(len(x), end + 240),
                               active_start_sample=start, active_end_sample=end, peak_contrast_db=contrast))
    # Noise PSD is recording-specific: quiet short-time frames, never another log's raw amplitude.
    _, _, z = signal.stft(bp, RATE, nperseg=256, noverlap=192)
    p = np.abs(z) ** 2
    energy = np.sum(p, axis=0)
    quiet = energy <= np.quantile(energy, .3)
    noise = np.median(p[:, quiet], axis=1) / np.log(2)
    mask = np.sqrt(np.maximum(1 - PARAMS['spectral_oversubtraction'] * noise[:, None] / (p + 1e-20), 0))
    mask = ndimage.uniform_filter(mask, size=(3, 3), mode='nearest')
    y = signal.istft(z * mask, RATE, nperseg=256, noverlap=192)[1][:len(x)]
    quiet_samples = np.zeros(len(x), dtype=bool)
    for index in np.flatnonzero(powers <= baseline):
        quiet_samples[index * HOP:index * HOP + WINDOW] = True
    stats = dict(background_input_rms=float(np.sqrt(np.mean(bp[quiet_samples] ** 2))),
                 background_output_rms=float(np.sqrt(np.mean(y[quiet_samples] ** 2))),
                 baseline_frame_power=baseline)
    return y, events, stats


def write_audio(path, x, rate):
    if not np.isfinite(x).all() or np.max(np.abs(x)) >= 1:
        raise ValueError('Invalid waveform')
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, rate, subtype='PCM_16')
    decoded, fs = sf.read(path)
    if fs != rate or decoded.shape != x.shape or np.max(np.abs(decoded)) >= 1 - 1 / 32768:
        raise ValueError('PCM output validation failed')
    return sha(path)


def run(probe=False):
    rows = list(csv.DictReader((SOURCE / 'manifest.csv').open()))
    clean_records, clean_counts = [], []
    for path in sorted((SOURCE / 'original_8k/clean').glob('*.wav')):
        x, fs = sf.read(path)
        assert fs == RATE and x.shape == (80000,)
        _, events, stats = extract(x)
        clean_counts.append(len(events))
        clean_records.append(dict(source=path.name, sha256=sha(path), detected_events=len(events), **stats))
    candidates, source_stats = [], []
    for row in rows:
        path = SOURCE / row['original_path']
        assert sha(path) == row['source_sha256']
        x, fs = sf.read(path)
        assert fs == RATE and x.shape == (80000,)
        y, events, stats = extract(x)
        ranked = sorted(events, key=lambda e: (-e['peak_contrast_db'], e['start_sample']))[:3]
        source_stats.append(dict(source=path.name, sha256=row['source_sha256'], detected_events=len(events),
                                 eligible_events=len(ranked), **stats))
        for rank, event in enumerate(ranked):
            event.update(source=row['source_path'], source_relative_path=row['original_path'],
                         source_sha256=row['source_sha256'], rank_in_source=rank + 1, **stats)
            candidates.append(event)
    # Favor one event per recording before a second/third; never duplicate to reach the cap.
    candidates.sort(key=lambda e: (e['rank_in_source'], -e['peak_contrast_db'], e['source']))
    selected = candidates[:2000]
    summary = dict(parameters=PARAMS, scanned_infested=len(rows), eligible_events=len(candidates),
                   sources_with_events=sum(r['eligible_events'] > 0 for r in source_stats),
                   selected_events=len(selected), clean_control_recordings=len(clean_counts),
                   clean_control_detected_events=sum(clean_counts),
                   clean_control_recordings_with_events=sum(n > 0 for n in clean_counts),
                   identity='User assumption: insect sounds are RPW; event detector does not distinguish insects from artifacts',
                   method='Recording-adaptive spectral subtraction plus high-contrast transient cropping; no generative audio',
                   caveats=['Residual in-burst noise and transient artifacts may remain',
                            'No listening annotations or isolated source ground truth',
                            'Clean control result is a diagnostic, not an independently tuned/tested false-positive estimate',
                            'No calibrated natural vibration level or physical exciter test'],
                   environment=dict(numpy=np.__version__, scipy=scipy.__version__, soundfile=sf.__version__))
    ROOT.mkdir(exist_ok=True)
    for filename, records in [('source_scan.csv', source_stats), ('clean_control_scan.csv', clean_records)]:
        with (ROOT / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (ROOT / 'probe.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    if probe:
        return
    manifest, seen, playlist = [], set(), []
    cached_source, y = None, None
    for event in selected:
        path = SOURCE / event['source_relative_path']
        if cached_source != path:
            y = extract(sf.read(path)[0])[0]
            cached_source = path
        a, b = event['start_sample'], event['end_sample']
        clip = y[a:b].copy()
        fade = min(40, len(clip) // 4)
        ramp = np.linspace(0, 1, fade)
        clip[:fade] *= ramp
        clip[-fade:] *= ramp[::-1]
        if np.max(np.abs(clip)) < 2 / 32768:
            continue
        floating_hash = hashlib.sha256(clip.astype('<f8').tobytes()).hexdigest()
        if floating_hash in seen:
            continue
        seen.add(floating_hash)
        index = len(manifest) + 1
        name = f'rpw_assumed_{index:04d}.wav'
        # Keep source digital scaling (apart from explicitly recorded overload protection).
        gain8 = min(1., .98 / max(np.max(np.abs(clip)), 1e-20))
        clip *= gain8
        file8 = ROOT / 'candidate_bursts_8k' / name
        hash8 = write_audio(file8, clip, RATE)
        play = signal.resample_poly(clip, 6, 1)
        play_gain = 10 ** (-12 / 20) / max(np.max(np.abs(play)), 1e-20)
        play *= play_gain
        play[0] = play[-1] = 0
        file48 = ROOT / 'playback_48k' / name
        hash48 = write_audio(file48, play, 48000)
        manifest.append(dict(event_id=index, **event, source_start_s=a / RATE, source_end_s=b / RATE,
                             duration_s=(b - a) / RATE, gain_8k=gain8, playback_gain=play_gain,
                             event_path=str(file8.relative_to(ROOT)), event_sha256=hash8,
                             playback_path=str(file48.relative_to(ROOT)), playback_sha256=hash48))
        if index <= 20:
            playlist.extend([play, np.zeros(24000)])
    if manifest:
        with (ROOT / 'manifest.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
        write_audio(ROOT / 'playback_48k/playlist.wav', np.concatenate(playlist), 48000)
    summary.update(exported_events=len(manifest), unique_exported_event_float_hashes=len(seen),
                   exported_source_recordings=len(set(e['source'] for e in manifest)),
                   total_event_seconds=sum(e['duration_s'] for e in manifest), playlist_events=min(20, len(manifest)),
                   playlist_gap_s=.5, selected_but_rejected_after_processing=len(selected) - len(manifest),
                   stage='complete')
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    run(parser.parse_args().probe)
