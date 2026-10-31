# Assumed-RPW activity bursts for the exciter

This is the **activity-burst version**, replacing whole ten-second recordings for the user's current playback goal. The user explicitly assumes that the insect sounds in the lab recordings are RPW. The processing suppresses stationary background and crops around strong transient activity; it does not prove that every transient is an insect or that no background remains within a burst.

## Start here

- `playback_48k/playlist.wav`: first 20 selected events, with 0.5 seconds of digital silence after each. Start amplifier volume at minimum. These are contrast-ranked examples, not listening-approved examples.
- `playback_48k/rpw_assumed_*.wav`: individual burst clips, mono PCM16 at 48 kHz, each normalized to a −12 dBFS peak for bench playback. Original pitch and duration are preserved; the normalized level is not natural insect vibration amplitude or a safe electrical power limit.
- `candidate_bursts_8k/rpw_assumed_*.wav`: source-scale, noise-suppressed excerpts at the original 8 kHz rate. Five-millisecond boundary fades avoid switching clicks. The small included context protects the beginning and decay of the event.
- `manifest.csv`: source file/hash, sample-accurate start/end times, detection contrast, background RMS before/after suppression, digital gains and both output hashes.
- `source_scan.csv`: all 2,000 source recordings, including those with no qualifying event. `clean_control_scan.csv` records the 64 clean reference recordings through the identical detector. `summary.json` gives actual exported counts.

Use the parent `EXCITER_GUIDE.md` for the Dayton DAEX25FHE-4 mounting and amplifier checks. The pictured bass-only amplifier's passband remains unresolved.

## Method and limits

1. Read the existing 2,000 unchanged 10-second, 8 kHz lab `infested` originals; verify their source hashes against the parent manifest.
2. Remove DC and apply a SciPy Butterworth `N=4` 100–2,500 Hz bandpass (eighth-order bandpass per pass), forward/backward offline. This is the project's selected sensing band; it can remove real insect energy outside that range.
3. Detect transient activity with 10 ms energy windows, 5 ms hops, a recording-specific quiet-frame baseline (30th percentile), +12 dB onset, +6 dB extent and at least +18 dB peak contrast. Merge gaps up to 40 ms, retain active durations 10–1,500 ms and add 30 ms context on either side. Exclude the first/last 100 ms to avoid boundary-filter transients. These are signal-processing thresholds, not a trained or validated RPW detector.
4. Estimate the noise spectrum separately for each recording from its quietest 30% of STFT frames. Use 256-sample Hann STFT / 64-sample hop and spectral subtraction with factor 2, with a zero gain floor and a small 3×3 mask smoothing. No synthetic insect sound is generated. Strong attenuation can distort or suppress desired activity; residual nonstationary noise can survive.
5. Keep at most three events per source, favoring one event per source before second/third events, then higher peak contrast. Export at most 2,000, never duplicate or select quiet windows to fill a quota. Reject near-silent postprocessing outputs and exact floating-waveform duplicates.
6. The same unchanged detector found one candidate in one of 64 clean-label clips. This is a useful contamination check, not a population false-positive estimate. No thresholds were adjusted after inspecting that result. Taps, rubbing and other transient artifacts cannot be ruled out by this detector.

The probe found **1,933 qualifying events across 1,335 source recordings**. All 1,933 were exported after postprocessing checks, totaling 170.23 seconds of event audio. `summary.json` records the final counts. These are **assumed-RPW candidate bursts**, not verified isolated stems. Original recordings remain untouched. Human listening and the physical exciter/piezo test are still needed. Separate source clips may be adjacent sections from one long recording, so source count is not biological independence.

The reported background RMS values compare the same quiet samples before and after spectral processing at source digital scale. They are not true source-separation SNR, an independently measured noise floor, or proof of improvement inside each event.

## Reproduce

From this folder, using the existing audio runtime:

```sh
'/Users/khalidhamdan/Documents/Codex/2026-10-23/files-mentioned-by-the-user-codex/work/venv/bin/python' -m unittest discover -s . -p 'test_extract_bursts.py'
'/Users/khalidhamdan/Documents/Codex/2026-10-23/files-mentioned-by-the-user-codex/work/venv/bin/python' extract_bursts.py
```

No downloads, model training or new dependencies are required. Inputs are addressed relative to this folder's parent. Synthetic tests cover stationary-noise/silence/tone rejection, injected-burst timing, waveform preservation and quiet-background attenuation. They do not validate biological identity or real-audio perceptual quality.
