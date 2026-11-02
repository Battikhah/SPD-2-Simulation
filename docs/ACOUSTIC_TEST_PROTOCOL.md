# TreeVibes exciter-to-piezo bench replay

## What this experiment can answer

This experiment checks how a source WAV is changed when played through the audio
player, amplifier, exciter, palm trunk, piezo sensor and acquisition chain. It is a
bench signal-chain/fidelity test. It cannot establish biological specificity,
live-palm detection range, infestation status or early-detection performance.

## Source audio and provenance

The local `outputs/treevibes-exciter-2026-10-30/AUDIO_README.md` documents 2,000
`infested`-labelled 10-second mono WAVs at 8 kHz and 64 separately clean-labelled
background WAVs. The clean clips are not paired background recordings. Five
band-passed clips have 48 kHz playback copies, and a 58-second playlist includes
those five with two-second gaps. Use the manifest to trace any playback file back
to its source and preserve source hashes and labels. Keep the original WAVs
unchanged; record processing variant and digital gain separately.

`infested` is a dataset label, not confirmation of red palm weevil in every file.
The source paper describes *X. chinensis* in mulberry, so use these stimuli as
candidate wood-borer proxy audio unless a species-confirmed RPW subset is separately
obtained and documented. Do not rename a file or transient as RPW based only on
the folder label, filtering or listening. Dataset/session independence is also
unresolved.

## Bench procedure

1. Follow the setup, amplifier compatibility, wiring, and stop-condition checks in
   `outputs/treevibes-exciter-2026-10-30/EXCITER_GUIDE.md`. In particular, verify
   the actual amplifier output path before driving the exciter. The purchased
   TPA3116 board is described as a bass/subwoofer board; its passband is unknown.
   Do not assume it reproduces the project's 100–2,500 Hz band. If the required
   passband cannot be verified, treat replay as exploratory and record that limit.
2. Keep the original audio and source hash; note the exact source file, source
   label, species-status evidence, processing version, playback sample rate, and
   digital gain. Disable player EQ, normalization and automatic gain. Record all
   player/amplifier settings. Do not use digital peak normalization as a physical
   amplitude calibration.
3. Fix the trunk, exciter attachment, piezo position, acquisition gain and cable
   routing. Record those positions, the exciter-to-piezo distance, trunk/session
   ID, room noise and replay level. Start with low amplifier gain. Avoid clipping
   and mechanical rattling; stop if either appears. The source recording already
   contains a tree/sensor response; the exciter, trunk and second piezo add another
   mechanical transfer response.
4. Capture controls in the same setup: (a) amplifier off/background, (b) amplifier
   on with digital silence, and (c) any available clean-labelled TreeVibes source.
   The clean clips are unmatched controls, not the same session's background.
   Then play a small seeded set of infested-labelled candidates. Repeat each at
   least three times without moving the rig.
5. Vary one bench setting at a time (gain, exciter-to-piezo distance, or mounting).
   Include repeated low-level trials and record failures, clipping and noise. Keep
   the raw recapture; do not overwrite it with filtered or normalized output.
6. Compare source and recapture only after time-aligning and resampling both to a
   common rate. First inspect raw gain/attenuation and clipping. Separately report
   gain-normalized waveform correlation, log-spectral difference, MFCC similarity,
   and/or burst-timing preservation if calculated. Compare the same source variant
   and fixed analysis window. Do not add an algorithm or claim a result before
   paired source/recapture files exist.

These comparisons describe bench replay fidelity only. They do not prove that a
replayed transient is biological, that a source label is correct, or that the
sensor can detect live infestation. Speaker replay is not a substitute for a
confirmed-RPW field study.

## What the 10/30/60/120-second scenarios mean

The current candidate playback files are 10 seconds long. Repeating or
concatenating a 10-second clip to make a 30-, 60-, or 120-second trial tests
processing/replay robustness only. It does not represent natural longer activity
or show that a longer sensing window improves detection chance. Verify that a
continuous, longer source recording exists before treating those as natural
recording-duration trials. The app's battery and capacity comparison holds packet
size and trigger probability fixed and scales processing linearly; detection
performance remains **not measured**.

The underlying TreeVibes paper concerns *X. chinensis* and mulberry rather than a
confirmed RPW corpus: [TreeVibes paper](https://www.mdpi.com/2624-6511/4/1/17/xml).

## Sensor-chain and confound controls (added 2 November 2026)

The development analysis in `docs/ESP32_DEV_DAY_REPORT.md` found that the current TreeVibes models
decide almost entirely from the 3–4 kHz band. That band sits just below the 8 kHz recordings' Nyquist
limit and outside the project's 100–2,500 Hz band. The models also depend on whether the near-Nyquist
spectrum is flat (recorder noise floor) or rolls off (real vibration energy). Any replay or field test
must control for this:

1. **Noise floor.** Record the ESP32 sensor chain with no trunk contact and with the sensor on a silent,
   damped block, using the exact gain and sample rate. Report the spectrum up to Nyquist and its flatness.
   A model may call any recording "infested" that differs from this floor.
2. **Passband.** Measure the exciter and amplifier passband up to the acquisition Nyquist (4 kHz at
   8 kHz). If the replay chain cannot reproduce 3–4 kHz, TreeVibes-trained models are not being tested
   on the information they use, and replay results say nothing about them.
3. **Matched negatives.** For every infested palm, record a clean palm at the same site, on the same
   day and with the same device and mounting. Include urban vibration (traffic, footsteps, children),
   rain and wind on clean palms. Record which trees were inspected and how.
4. **Report within-site, within-device metrics** (as `scripts/within_device_eval.py` and
   `scripts/esp32_report.py` do) next to any pooled score. A pooled score can be won by recognising
   the recorder or site.
5. **Hold out whole recorders.** Folder-grouped validation leaks recorder identity. When a recorder's
   infested folders are in training, its held-out clean folder (TreeVibes folder 24) is scored as
   infested. When the whole recorder is held out, it is mostly separated. Evaluate a new device with that
   device's recordings held out.
6. **Folder 24/25 check.** The TreeVibes "forest, healthy trees" folders 24 and 25 contain longer,
   more regular transients than the infested recordings. Listen to them before using similar recordings
   as negatives, and keep their inspection evidence with the recordings.

## Blank recording log

Fill `docs/acoustic_test_template.csv` with one row per source/recapture trial. Put
long source session duration and analysis-window duration in separate columns.
Record `front_end_gain` and `control_condition` explicitly, including amplifier-off,
powered silence, clean-source replay, candidate replay, or mechanically decoupled.
`source_species_status` should say `confirmed`, `unconfirmed`, or `unknown` and
include the evidence in notes. Never infer confirmed healthy or RPW from a quiet
waveform. Keep the source WAV immutable and save each rerecording under a unique
session/trial path.
