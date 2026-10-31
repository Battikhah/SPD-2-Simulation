"""Build an auditable lab-borer playback set; filtering does not identify RPW."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import platform
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent
AUDIT = Path('/Users/khalidhamdan/Documents/Codex/2026-10-23/files-mentioned-by-the-user-codex/outputs/rpw-detection/data/manifests/full/all_audio_audit.csv')
RATE = 8000
SOS = signal.butter(4, [100, 2500], btype='bandpass', fs=RATE, output='sos')
LOCAL = threading.local()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pcm_hash(x):
    return digest(b'8000:1:' + np.asarray(x, dtype='<f8').tobytes())


def bandpass(x):
    return signal.sosfiltfilt(SOS, x)


def spectrum(x):
    return signal.stft(x, fs=RATE, nperseg=512, noverlap=384)[2]


def attenuate(x, noise_power):
    z = spectrum(x)
    # A 0.5 amplitude floor limits spectral attenuation to about 6 dB.
    mask = np.sqrt(np.maximum(1 - noise_power[:, None] / (np.abs(z) ** 2 + 1e-20), 0.25))
    y = signal.istft(z * mask, fs=RATE, nperseg=512, noverlap=384)[1]
    return y[:len(x)]


def choose_sources():
    saved = ROOT / 'selection.json'
    if saved.exists():
        previous = json.loads(saved.read_text())
        return previous['selected'], previous['stats']
    rows = list(csv.DictReader(AUDIT.open()))
    chosen, stats = {}, {}
    for label, count in [('infested', 2000), ('clean', 64)]:
        eligible = [r for r in rows if r['path'].startswith(f'lab/lab/{label}/')
                    and float(r['clipping_fraction']) == 0 and float(r['rms']) > 0
                    and r['sample_rate'] == '8000' and r['channels'] == '1'
                    and r['frames'] == '80000']
        eligible.sort(key=lambda r: r['path'])
        random.Random(42).shuffle(eligible)
        seen, unique = set(), []
        for row in eligible:
            if row['pcm_sha256'] not in seen:
                seen.add(row['pcm_sha256'])
                unique.append(row)
        if len(unique) < count:
            raise ValueError(f'Only {len(unique)} eligible unique {label} clips')
        chosen[label] = unique[:count]
        stats[label] = dict(eligible=len(eligible), unique_eligible=len(unique), selected=count)
    return chosen, stats


def download(row):
    name = row['path'].split('/')[-1]
    label = row['path'].split('/')[-2]
    path = ROOT / 'original_8k' / label / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and digest(path.read_bytes()) == row['file_sha256']:
        return row
    from kagglehub.clients import build_kaggle_client
    from kagglehub.handle import DatasetHandle
    from kagglehub.http_resolver import _build_dataset_download_request

    if not hasattr(LOCAL, 'client'):
        LOCAL.context = build_kaggle_client()
        LOCAL.client = LOCAL.context.__enter__()
    last = None
    for attempt in range(2):
        try:
            response = LOCAL.client.datasets.dataset_api_client.download_dataset(
                _build_dataset_download_request(DatasetHandle('potamitis', 'treevibes', 1), row['path']))
            response.raise_for_status()
            with response:
                data = b''.join(response.iter_content(256 * 1024))
            if digest(data) != row['file_sha256']:
                raise ValueError('Source SHA256 differs from version-1 archive audit')
            x, rate = sf.read(io.BytesIO(data), dtype='float64', always_2d=True)
            if rate != RATE or x.shape != (80000, 1) or pcm_hash(x) != row['pcm_sha256']:
                raise ValueError('Decoded source differs from audited PCM')
            temporary = path.with_suffix('.part')
            temporary.write_bytes(data)
            temporary.replace(path)
            return row
        except Exception as error:
            last = f'{type(error).__name__}: {str(error).splitlines()[0][:100]}'
            if attempt == 0:
                time.sleep(1)
    raise RuntimeError(f"{row['path']}: {last}")


def save_wav(path, x, rate=RATE):
    if not np.isfinite(x).all() or np.max(np.abs(x)) >= 1:
        raise ValueError(f'Nonfinite or overloaded derivative: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, rate, subtype='PCM_16')
    decoded, actual_rate = sf.read(path, dtype='int16', always_2d=True)
    if actual_rate != rate or decoded.shape != (len(x), 1) or sf.info(path).subtype != 'PCM_16':
        raise ValueError(f'Written WAV metadata differs: {path}')
    if np.any(np.abs(decoded.astype(np.int32)) >= 32767):
        raise ValueError(f'Written PCM reaches full scale: {path}')
    return digest(path.read_bytes())


def selfcheck():
    t = np.arange(RATE * 2) / RATE
    x = 0.1 * np.sin(2 * np.pi * 500 * t)
    y = bandpass(x)
    assert len(y) == len(x) and np.isfinite(y).all()
    assert np.sqrt(np.mean(y[1000:-1000] ** 2)) > .06
    z = bandpass(0.1 * np.sin(2 * np.pi * 20 * t))
    assert np.sqrt(np.mean(z[1000:-1000] ** 2)) < .001
    quiet = attenuate(np.zeros(RATE), np.ones(257))
    assert len(quiet) == RATE and np.max(np.abs(quiet)) == 0
    restored = attenuate(y, np.zeros(257))
    assert np.max(np.abs(restored - y)) < 1e-10
    assert len(signal.resample_poly(y, 6, 1)) == len(y) * 6
    print('DSP selfchecks passed', flush=True)


def run():
    started = time.monotonic()
    selected, selection_stats = choose_sources()
    infested_hashes = {r['pcm_sha256'] for r in selected['infested']}
    clean_hashes = {r['pcm_sha256'] for r in selected['clean']}
    assert len(infested_hashes) == 2000 and len(clean_hashes) == 64
    assert not infested_hashes.intersection(clean_hashes)
    selection_path = ROOT / 'selection.json'
    audit_hash = (json.loads(selection_path.read_text())['source_audit_sha256'] if selection_path.exists()
                  else digest(AUDIT.read_bytes()))
    selection_path.write_text(json.dumps(dict(seed=42, source_audit_sha256=audit_hash,
        stats=selection_stats, selected=selected), indent=2) + '\n')
    failures, done = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(download, row) for label in selected for row in selected[label]]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(str(error))
            done += 1
            if done % 100 == 0 or done == len(futures):
                print(json.dumps(dict(downloaded=done-len(failures), attempted=done, failures=len(failures))), flush=True)
                (ROOT / 'status.json').write_text(json.dumps(dict(stage='download', attempted=done,
                    selected=len(futures), failures=failures), indent=2) + '\n')
    if failures:
        raise RuntimeError('Source acquisition failed; see status.json; rerun resumes verified files')

    noise = []
    for row in selected['clean']:
        x, _ = sf.read(ROOT / 'original_8k/clean' / Path(row['path']).name)
        noise.append(np.median(np.abs(spectrum(bandpass(x))) ** 2, axis=1))
    noise_power = np.median(noise, axis=0)
    np.save(ROOT / 'clean_reference_noise_power.npy', noise_power)
    manifest, audition = [], []
    for i, row in enumerate(selected['infested'], 1):
        name = Path(row['path']).name
        original = ROOT / 'original_8k/infested' / name
        x, _ = sf.read(original)
        y = bandpass(x)
        z = attenuate(y, noise_power)
        # Usually unity; only reduce level if zero-phase filtering overshoots.
        gain = min(1.0, .98 / max(np.max(np.abs(y)), np.max(np.abs(z)), 1e-20))
        y, z = y * gain, z * gain
        bp_path, dn_path = ROOT / 'bandpassed_8k' / name, ROOT / 'attenuated_8k' / name
        bp_hash, dn_hash = save_wav(bp_path, y), save_wav(dn_path, z)
        manifest.append(dict(index=i, source_path=row['path'], source_sha256=row['file_sha256'],
            source_pcm_sha256=row['pcm_sha256'], source_rate_hz=RATE, duration_s=len(x)/RATE,
            source_rms=float(np.sqrt(np.mean(x*x))), source_peak=float(np.max(np.abs(x))),
            original_path=str(original.relative_to(ROOT)), bandpassed_path=str(bp_path.relative_to(ROOT)),
            bandpassed_sha256=bp_hash, attenuated_path=str(dn_path.relative_to(ROOT)), attenuated_sha256=dn_hash,
            processing_gain=gain, bandpassed_peak=float(np.max(np.abs(y))), attenuated_peak=float(np.max(np.abs(z))),
            bandpassed_rms=float(np.sqrt(np.mean(y*y))), attenuated_rms=float(np.sqrt(np.mean(z*z))),
            label='TreeVibes lab infested; species and audible insect activity unverified'))
        if i <= 5:
            v = signal.resample_poly(y, 6, 1)
            fade = np.linspace(0, 1, 480)
            v[:480] *= fade
            v[-480:] *= fade[::-1]
            playback_gain = 10 ** (-12 / 20) / max(np.max(np.abs(v)), 1e-20)
            v *= playback_gain
            path = ROOT / 'playback_48k' / f'candidate_{i:02d}.wav'
            checksum = save_wav(path, v, 48000)
            audition.append(dict(file=str(path.relative_to(ROOT)), source=name, source_index=i,
                derivative='bandpassed_8k', playback_gain=playback_gain,
                peak_dbfs=-12, fade_ms=10, file_sha256=checksum))
        if i % 500 == 0:
            print(f'Processed {i}/2000', flush=True)
    with (ROOT / 'manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    clips = [sf.read(ROOT / entry['file'])[0] for entry in audition]
    playlist = np.concatenate([part for i, clip in enumerate(clips)
        for part in ([np.zeros(96000), clip] if i else [clip])])
    save_wav(ROOT / 'playback_48k/playlist.wav', playlist, 48000)
    save_wav(ROOT / 'playback_48k/silence_10s.wav', np.zeros(480000), 48000)
    (ROOT / 'playback_48k/manifest.json').write_text(json.dumps(audition, indent=2) + '\n')
    directory_counts = {name: len(list((ROOT / name).glob('*.wav'))) for name in
                        ['original_8k/infested', 'original_8k/clean', 'bandpassed_8k', 'attenuated_8k', 'playback_48k']}
    assert list(directory_counts.values()) == [2000, 64, 2000, 2000, 7], directory_counts
    report = dict(stage='complete', source='potamitis/treevibes version 1', seed=42,
        infested_clips=len(manifest), clean_reference_clips=64, seconds_per_source=10,
        unique_infested_pcm=len(set(r['source_pcm_sha256'] for r in manifest)),
        directory_wav_counts=directory_counts,
        written_wav_validation='Every derivative re-decoded as mono PCM16; rate/frame count verified; zero full-scale samples',
        cross_label_selected_pcm_duplicates=0,
        source_selection=selection_stats,
        filter='Butterworth N=4 bandpass 100–2500 Hz (8th-order one-pass), forward/backward zero phase',
        attenuation='STFT 512/128 hop; median clean-reference spectral power; gain mask floor 0.5; no model',
        source_audit_sha256=audit_hash, runtime_seconds=round(time.monotonic()-started, 2),
        environment=dict(python=sys.version.split()[0], platform=platform.platform(), numpy=np.__version__,
            scipy=scipy.__version__, soundfile=sf.__version__),
        limits=['Not species-confirmed RPW', 'Filtering does not isolate insects or prove audible activity',
            'Split laboratory recordings may be adjacent; exact-PCM uniqueness is not biological independence',
            'Clean-reference clips are unpaired, not simultaneous noise measurements',
            'Playback normalized to -12 dBFS is a bench stimulus, not natural amplitude calibration',
            'No listening annotation or physical exciter measurement performed'])
    (ROOT / 'qc_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    (ROOT / 'status.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selfcheck', action='store_true')
    args = parser.parse_args()
    selfcheck()
    if not args.selfcheck:
        run()
