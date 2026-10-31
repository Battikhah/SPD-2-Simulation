# TreeVibes laboratory playback material

## Current requested output: RPW-assumed activity bursts

Following the user's clarification, treat the insect activity as RPW **by assumption**. The new `rpw_bursts/` package extracts short activity intervals and suppresses stationary background; it is the intended starting point for activity-only exciter playback. See `rpw_bursts/README.md`, `rpw_bursts/manifest.csv` and `rpw_bursts/playback_48k/playlist.wav`. The original full-length versions below remain source references. This assumption does not establish that every detected transient is an insect or that all residual noise was removed. The playlist's inserted gaps are artificial; do not use it to measure natural activity rates.

This package enhances **2,000 laboratory `infested` recordings** from
`potamitis/treevibes`, version 1 (Kaggle metadata: CC0). They are candidate
wood-borer sounds for a mechanical replay experiment. They are **not confirmed
red palm weevil recordings**, and filtering does not establish which transient
was produced by an insect. Source acquisition and processing status are recorded
in `status.json`; only `stage: complete` indicates the full package was produced.

## Audio files

- `original_8k/infested/`: 2,000 unchanged downloaded WAVs, each 10 s, mono,
  8 kHz, PCM16. Keep these as the reference.
- `original_8k/clean/`: 64 separate clean-label laboratory WAVs used only to
  estimate a background spectrum. They are not paired background measurements.
- `bandpassed_8k/`: full-duration 100–2,500 Hz versions, with no time stretching.
- `attenuated_8k/`: optional conservative spectral attenuation of the bandpassed
  audio. Retain the bandpassed version when attenuation makes a pulse less clear.
- `playback_48k/candidate_01.wav` through `candidate_05.wav`: first five seeded
  clips, resampled from the bandpassed audio to 48 kHz. Each has 10 ms fades and
  a normalized peak of −12 dBFS. They are examples, **not ranked or listening
  approved**. The digital gain is in `playback_48k/manifest.json`.
- `playback_48k/playlist.wav`: those five clips in order, with 2 s digital
  silence between them; total duration 58 s.
- `playback_48k/silence_10s.wav`: zero-valued digital control. An amplifier can
  still produce noise while playing digital silence.

Resampling does not recover frequencies above the original 4 kHz Nyquist limit.
The chosen band also removes real signal outside 100–2,500 Hz. These are bench
stimuli, not a calibrated reconstruction of the animal's vibration amplitude.

## Reproducible preparation

1. Read the previous full version-1 archive audit, which inventories 53,676 lab
   infested and 967 lab clean WAVs. The original lab README says longer RolandR4
   recordings of logs were split into small WAV files.
2. Require 8 kHz mono, exactly 80,000 frames, nonzero RMS, and no source samples
   at the audit's clipping threshold. Sort names, shuffle with Python seed 42,
   remove exact decoded-PCM duplicates, then select 2,000 infested and 64 clean.
   `selection.json` preserves the choices, source hashes, and audit-file hash.
   Selection is deliberately a reproducible QC-filtered subset, not an unbiased
   prevalence sample or an independent train/test split.
3. Download only those WAVs through Kaggle's public version-1 file API. Verify
   raw-file SHA256 and decoded-PCM SHA256 against the previous audit before
   accepting any file. Existing verified files are reused when resuming.
4. Apply a Butterworth bandpass (`N=4`, eighth-order one-pass bandpass) with
   forward/backward filtering.
   This zero-phase offline operation has a different response from a single-pass
   live filter. Keep all 10 seconds rather than relabeling the loudest tap as RPW.
5. For optional attenuation, use a 512-sample Hann STFT with a 128-sample hop.
   Estimate median spectral power within each clean clip, then take the median
   across the 64 clips. Apply `sqrt(max(1 − noise_power / signal_power, 0.25))`
   to the complex spectrum: the amplitude floor is 0.5 (about −6 dB).
   This can remove desired energy or retain unwanted energy; it is not source
   separation and has no measured SNR-improvement claim.
6. Preserve source digital amplitude scaling in the 8 kHz derivatives unless filtering
   overshoots: any common attenuation needed to keep both derivatives below
   0.98 peak is recorded as `processing_gain` in `manifest.csv`. Only the small
   audition set is individually normalized.
7. Check counts, unique decoded-source hashes, finite outputs, duration, sample
   rate, channels, PCM subtype, hashes, and clipping. `qc_summary.json` records
   processing details, runtime versions, and remaining evidence limits.

Run `prepare_audio.py --selfcheck` for the small DSP regression check, or
`prepare_audio.py` to resume acquisition and rebuild the derivatives. The
runtime requires NumPy, SciPy, SoundFile, and KaggleHub; its exact versions are
recorded in `qc_summary.json`. The current machine's verified command is:

```sh
'/Users/khalidhamdan/Documents/Codex/2026-10-23/files-mentioned-by-the-user-codex/work/venv/bin/python' '/Users/khalidhamdan/Documents/ChatGPT/SPD 2 Simulation/outputs/treevibes-exciter-2026-10-30/prepare_audio.py'
```

Append `--selfcheck` for a quick check without downloads or reprocessing.
Subsequent runs use the bundled `selection.json`; they do not need the external
audit file. To choose a new subset from scratch, the script's explicit original
audit path is required. Missing selected WAVs are downloaded; existing WAVs with
matching hashes are reused. The public source is
[TreeVibes on Kaggle](https://www.kaggle.com/datasets/potamitis/treevibes).

Do not train and evaluate by randomly splitting these files: source sessions,
individual logs, species, and adjacent segments are unresolved. Exact-PCM
uniqueness does not establish biological independence. Human listening review,
confirmed-species material, and physical exciter/piezo measurements are separate
next steps.
