# ESP32 dev day: what the models actually detect (2 November 2026)

This follows `docs/ESP32_OVERNIGHT_REPORT.md` and supersedes its recommendation. All figures use
development data. Sections 1–4 use folder-grouped 5-fold cross-validation and section 5 holds out whole
recorders. The sealed test split was never read, and nothing was measured on an ESP32. Raw outputs are
in `outputs/folder-diagnostics/` (git-ignored); the scripts are listed at the end.

## Recommendation

- **Model:** `stream1s_w8_d4`, pos_boost 2, trained with folder balancing (p = 0.5), deployed as one int8 model.
  - About 35 KB on flash, with a 37 KiB arena that fits the 64 KiB in the firmware example.
  - 34 M multiply-adds per 10 s decision.
- **Threshold:** calibrated per device. At installation, record known-healthy trees and set the alarm threshold at the 90th percentile of their 10 s scores. A threshold copied from other devices flags 100% of some devices' healthy trees.
- **Expected performance on an unseen device:** about 86–90% recall with 14–15% false alarms on healthy trees, across two independent seed sets (section 5).
  - The figures are rough: two of the three test recorders have only 27 healthy recordings each.
  - Healthy and infested trees still come from different sites, even within a recorder.
- **Not for this:** pooled F1 numbers from folder-grouped folds. They mostly measure recorder, site and "activity vs silence" differences.

## TL;DR

- **The models mostly detect high-frequency vibration activity and recording conditions, not RPW.**
  - **The 3–4 kHz band carries everything.** All the CNN's skill sits in that band, just below the 4 kHz Nyquist limit. Trained on those 8 mel bands alone it scores AUC 0.94; without them, 0.60.
  - **One untrained number beats the CNN.** The level drop in the last 500 Hz below 4 kHz gives F1 0.965 with no training. The best CNN gets 0.932.
  - **Half the clean data is silence.** More than half of the clean recordings (folder 9) are the recorder's own white noise floor. One recorder captured only clean trees, and every infested tree is an urban palm.
- **Without recorder 8409 there is little skill left.** That recorder captured only clean trees (683 of the 975 clean recordings). Trained without them, the models fall from AUC 0.93 to about 0.71. Within the one same-month recorder, they then rank clean above infested (0.15).
- **Folder-grouped cross-validation leaks recorder identity.** Under folder folds, every model ranks the clean forest folder 24 above the infested trees from the same recorder (AUC 0.02–0.39). With leave-one-recorder-out folds, where each recorder is scored only by models that never heard it, w8 separates the classes within each recorder at 0.94 / 0.86 / 0.81. Recorder-held-out evaluation is the right protocol for a new ESP32 device. Site and activity level remain confounded within each recorder.
- **Your "noise, not RPW" idea is half right.** Folders 24/25 are quiet, not noisy, but they hold sharp high-frequency transients. The models treat any such activity as infestation.
- **Engineering answers:**
  - A 97%-training-recall threshold rule is steadier than argmax-F1.
  - A single model is almost as good as the 3-model ensemble at a third of the compute.
  - stream1s_w16 costs about 3.8× the compute of w8.
  - Real wind or rain barely hurts recall; white noise destroys it.
- **What to do:** record with the actual ESP32 sensor chain, on clean palms at the same urban sites and on verified infested palms. Then re-evaluate within each site and device. More training on TreeVibes cannot fix this.

## 1. Folders 24/25: not noisy, but impulsive

Source: `scripts/folder_diagnostics.py`, `click_feature_cv.py`.

| Group | Recordings | RMS (dBFS) | Noise floor >500 Hz | Transients per s | Kurtosis >500 Hz | Click length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean, Feb–Mar (7, 8, 9) | 921 | −43.4 | −45.3 | 0.10 | 0.5 | 2.9 ms |
| clean, May (24, 25) | 54 | −40.9 | −45.0 | 1.39 | 33 | 6.8 ms |
| infested, Feb–Mar | 332 | −42.0 | −44.8 | 0.34 | 6.0 | 2.1 ms |
| infested, May–Jun | 1,235 | −37.8 | −40.8 | 0.88 | 5.4 | 2.4 ms |

- **24/25 are as impulsive as infested recordings.** Their clicks are longer and more regular (inter-click CV 1.27 vs 1.6–2.4).
- **Simple click rules fail.** The recording-level overlap is too large: a click-length rule tuned on folder 25 rejects only 7% of folder 24 at 95% infested recall.
- **Hand features don't fix it.** Under folder CV they don't help on their own or fused with the CNN.
- **Folder 7 is impulsive too** (kurtosis 18), and it is the clean folder with the most false alarms.

## 2. The 3–4 kHz band carries the decision

Sources: `scripts/band_occlusion.py`, plus training with `--max-band` and `--min-band`.

| stream1s_w16 pb2 trained on | AUC (per seed, float) |
| --- | --- |
| all 64 mel bands | 0.939 / 0.935 / 0.925 |
| **only bands 56–63 (3.0–4.0 kHz)** | **0.936 / 0.938 / 0.939** |
| everything except bands 56–63 | 0.595 / 0.593 / 0.605 (int8 ensemble: F1 0.627, every clean recording flagged) |
| all bands, gain-normalised per window | 0.852 / 0.837 / 0.859 |

Masking the 3.0–3.9 kHz block on the trained model lowers the folder 24/25 logit by 2.70, from 2.65 to about 0. On real infested windows it lowers it by only 0.97; no other block moves either group by more than 0.73.

## 3. One spectral number explains most of it

Source: `scripts/rolloff_check.py` and `spectral_shape.csv`.

- **Clean folders 7, 8 and 9 are flat up to 4 kHz.** Folder 9 is pure white noise (spectral flatness 0.994) at the lowest level of any folder.
- **Every infested folder rolls off near Nyquist,** as an anti-aliasing filter does when there is real vibration energy. Folders 24 and 25 roll off too.
- **The roll-off alone does better than the CNN.** The roll-off is level(3.8–4.0 kHz) − level(3.5–3.8 kHz). Used directly as the score, with a cross-fitted F1 threshold, it gives AUC 0.951 and F1 0.965 (98.2% recall, 94.8% precision, 8.7% FPR). It flags 100% of 24/25 and 5/8/1% of folders 7/8/9.
- **The CNN follows the roll-off:** Spearman correlation 0.75.

## 4. Site and recorder confounds (TreeVibes.xlsx)

- **Sites:** every infested folder is an urban palm (city centre, alleys, children playing, pavement). The clean folders are orange, fig and apricot trees in wind or rain, plus "forest, healthy trees" (24, 25). Infested folders 3, 4, 18, 22 and 23 are marked CONFIRMATION = NO.
- **Recorders** (IMEI tail):

  | Recorder | Infested folders | Clean folders |
  | --- | --- | --- |
  | 8409 | none | 7 and 9 (683 of 975 clean recordings) |
  | 7260 | 2, 4, 6 | 8 |
  | 2538 | 11 folders | 24 |
  | 8391 | 7 folders | 25 |

- **Within-recorder AUC** (`scripts/within_device_eval.py`; also printed by `esp32_report.py`):

  | Model | All data | Recorder 7260 | Recorder 2538 | Recorder 8391 |
  | --- | ---: | ---: | ---: | ---: |
  | stream1s_w16 pb2, fresh seeds | 0.939 | 0.997 | 0.266 | 0.793 |
  | stream1s_w8_d4 pb2, distilled, fresh seeds | 0.936 | 0.991 | 0.082 | 0.808 |
  | stream1s_w8 pb1, folder-balanced | 0.934 | 0.998 | 0.393 | 0.938 |
  | roll-off number, no training | 0.951 | 0.978 | 0.106 | 0.606 |

- **Without silent folder 9, AUC drops:** w16 0.872 and roll-off 0.906.
- **Against the forest folders 24+25, all models score below chance:** 0.04–0.34.

**Decisive test: never train on recorder 8409.** Folders 7 and 9 were dropped from every training fold. They are still scored when their fold is held out.

| Model, not trained on recorder 8409 | All data | Recorder 7260 | Recorder 2538 | Recorder 8391 |
| --- | ---: | ---: | ---: | ---: |
| stream1s_w8 pb1 (float) | 0.706 | 0.141 | 0.218 | 0.627 |
| stream1s_w16 pb2 (float) | 0.715 | 0.153 | 0.225 | 0.709 |

Recorder 7260 is the only one with infested and clean trees from the same month. On it, AUC falls from 0.997 to about 0.15: the model now ranks clean folder 8 above infested folders 2, 4 and 6. So most of the apparent skill came from learning what recorder 8409's quiet and wind-or-rain clean recordings look like.

**Leave-one-recorder-out** (`esp32_train.py --folds recorder`, 7 folds). Every recording is scored by models that never heard its recorder:

| Model | Pooled AUC (per seed, float) | Recorder 7260 | Recorder 2538 | Recorder 8391 |
| --- | --- | ---: | ---: | ---: |
| stream1s_w8 pb1, folder-grouped folds | 0.934 / 0.921 / 0.920 | 0.995 | 0.287 | 0.817 |
| stream1s_w8 pb1, recorder-held-out folds | 0.938 / 0.948 / 0.917 | 0.936 | **0.859** | 0.806 |

The folder-24 inversion under folder folds is largely recorder-fingerprint leakage. Folder folds keep the same recorder's infested folders in training, so the model learns "recorder 2538's sound means infested", and the held-out clean folder 24 inherits that. With the recorder unseen, folder 24 is mostly separated. Two cautions remain:

- Within each recorder, the infested trees are urban palms and the clean trees are one forest or fig-tree site.
- The pooled number is still made easy by folder 9's silence.

## 5. New-device estimates and the model to deploy

Everything below uses leave-one-recorder-out folds. Thresholds are chosen on the other recorders, and scores come from the int8 3-seed ensemble unless marked float.

| Model | AUC | argmax-F1 rule: recall / precision / FPR, F1 | 97%-recall rule: recall / precision / FPR, F1 |
| --- | ---: | --- | --- |
| stream1s_w16 pb2 | 0.953 | 91 / 90 / 16%, 0.905 | 97 / 92 / 13%, 0.948 |
| stream1s_w8 pb1 | 0.941 | 88 / 83 / 30%, 0.854 | 94 / 88 / 21%, 0.906 |

The architecture comparison under recorder hold-out (float AUC per seed):

| Model | AUC per seed | Arena | MACs per 10 s, single model |
| --- | --- | ---: | ---: |
| stream1s_w16 pb2 | 0.948 / 0.956 / 0.949 | 74 KiB | 90 M |
| **stream1s_w8_d4 pb2** | 0.945 / 0.952 / 0.949 (fresh seeds: 0.945 / 0.953 / 0.952) | **37 KiB** | **34 M** |
| stream1s_w8 pb1 | 0.938 / 0.948 / 0.917 | 37 KiB | 24 M |
| stream1s_ds pb1 | 0.936 / 0.941 / 0.920 | 87 KiB | 39 M |
| stream1s_w8 pb1, folder-balanced | 0.933 / 0.934 / 0.920 | 37 KiB | 24 M |
| stream1s_w4 pb2 | 0.856 / 0.935 / 0.862 | 18 KiB | 7 M |

Training for 40 epochs gives no consistent gain: w8_d4 scores 0.959 / 0.946 / 0.939, against 0.945 / 0.952 / 0.949 at 20 epochs. Distillation from a 32-channel teacher gives no gain either: 0.944 / 0.944 / 0.951.

**int8 results with the threshold chosen on the other recorders** (97%-recall rule), shown as F1 (recall / precision / FPR %), plus within-recorder AUC:

| Model | Pooled F1 | Recorder 7260 | Recorder 2538 | Recorder 8391 |
| --- | --- | ---: | ---: | ---: |
| stream1s_w16 pb2 | 0.948 (97/92/13) | 0.947 | 0.840 | 0.830 |
| stream1s_ds pb1 | 0.920 (98/87/23) | 0.951 | 0.744 | 0.749 |
| stream1s_w8_d4 pb2 | 0.903 (90/90/16) | 0.982 | 0.770 | 0.740 |
| stream1s_w8 pb1 | 0.906 (94/88/21) | 0.933 | 0.854 | 0.813 |
| stream1s_w8 pb1, folder-balanced | 0.851 (88/83/30) | 0.940 | 0.925 | 0.888 |
| stream1s_w8_d4 pb2, folder-balanced | 0.848 (86/83/28) | 0.985 | 0.900 | 0.869 |
| stream1s_w16, 3–4 kHz only | 0.948 (95/95/9) | 0.999 | 0.583 | 0.389 |

- **Pooled F1 is not a good model selector here.** A w16 that sees only 3–4 kHz has excellent pooled F1 but barely separates the classes inside two recorders.
- **Transferred thresholds fail.** At a threshold taken from other recorders, every model flags 100% of folders 24 and 25. The ranking inside those recorders is good, but each device shifts all its scores.

**Per-device calibration** (`scripts/device_calibration.py`). For each held-out recorder, the threshold is the 90% quantile of a random half of its healthy-tree scores. It is tested on the other half and that recorder's infested recordings, over 200 splits. No infested labels from the device are used.

| Model | Recorder 7260 | Recorder 2538 | Recorder 8391 | Mean recall / FPR |
| --- | --- | --- | --- | --- |
| **stream1s_w8_d4, folder-balanced** | 96 / 11 | 85 / 15 | 78 / 14 | **86 / 14** |
| **the same, fresh seeds 3–5** | 99 / 11 | 93 / 16 | 78 / 17 | **90 / 15** |
| stream1s_w8, folder-balanced | 72 / 11 | 89 / 14 | 80 / 14 | 80 / 13 |
| the same, fresh seeds 3–5 | 56 / 11 | 93 / 16 | 88 / 17 | 79 / 15 |
| stream1s_w16 | 78 / 10 | 79 / 16 | 73 / 16 | 77 / 14 |
| stream1s_w8_d4 | 97 / 11 | 73 / 16 | 56 / 18 | 75 / 15 |
| stream1s_w8 | 69 / 10 | 77 / 13 | 73 / 17 | 73 / 13 |
| sensitivity: stream1s_w8_d4, folder-balanced p = 1 | 91 / 10 | 91 / 15 | 84 / 16 | 89 / 14 |

Stronger balancing (p = 1) is not clearly better than p = 0.5, so the recommendation keeps p = 0.5.

Folder-balanced training looked useless under folder-grouped folds. Under recorder hold-out with per-device calibration, it is what makes models robust across devices.

**How many healthy recordings does calibration need?** Recommended model, recorder 7260 (238 clean recordings), 90% quantile, 200 random draws:

| Healthy recordings used | Mean recall | Mean FPR | FPR range (10th–90th percentile) |
| ---: | ---: | ---: | --- |
| 5 | 97% | 21% | 6–39% |
| 10 | 96% | 15% | 5–28% |
| 20 | 96% | 13% | 5–22% |
| 50 | 96% | 11% | 5–16% |
| 100 | 96% | 11% | 7–17% |

Plan on about 50 healthy-tree recordings per installed device. These TreeVibes recordings are about 20 s each, so that is roughly 17 minutes of audio, ideally from several known-healthy trees.

## 6. Engineering answers

- **Threshold rule** (`scripts/threshold_rules_study.py`, 27 window ensembles). The threshold where the training folds reach 97% recall scores best:

  | Measure | 97%-recall rule | argmax-F1 rule |
  | --- | --- | --- |
  | Mean F1 | 0.894 | 0.885 |
  | Worst F1 | 0.838 | 0.812 |
  | F1 change between seed sets | 0.014 | 0.034 |

  For stream1s_w16 on fresh seeds it gives 91.4% recall and 91.4% precision (F1 0.914). It was chosen among five rules, so it carries a small selection optimism.
- **Ensemble vs single model:** across 27 configurations the 3-model ensemble adds only +0.002 AUC and +0.005 F1 over a single model, for three times the compute.
- **Compute per 10 s decision** (3-model ensemble; `scripts/esp32_compute.py`): w4 20 M, w8 71 M, w8_d4 102 M, ds 118 M, w16 269 M MACs. A single w16 model needs 90 M. Latency is unmeasured.
- **Noise stress** (`scripts/noise_stress.py`; held-out folds only). Recall at the F1 threshold for each noise level:

  | Noise added | None | 20 dB | 10 dB | 5 dB | 0 dB |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Rain | 100% | 99% | 94% | 84% | 65% |
  | Wind | 100% | 100% | 91% | 80% | 69% |
  | White | 98% | 96% | 45% | 5% | 1% |

  White noise added to clean recordings causes no false alarms. The model fails safe.
- **Distilled stream1s_w8_d4 on fresh seeds:** AUC 0.936 (it holds), but F1 0.904 (it does not).
- **Why the 10 s models are worse:** a 1 s model with the 10 s models' stride-2 stem loses only about 0.01 AUC (int8 0.916 vs 0.926). Downsampling is a small part of the gap.

## 7. What this means

- **The F1 and AUC figures** in the overnight report and the deck are true for this dataset, but they mostly measure "vibration activity at 3–4 kHz vs a silent sensor", plus device and site. They are not evidence of RPW-specific detection.
- **For the ESP32,** the cue the models use is shaped by the recorder's noise floor and anti-aliasing filter. The ESP32 microphone or accelerometer and its ADC chain will differ, so a TreeVibes-trained model may not transfer at all.
- **Needed:**
  1. Record with the actual ESP32 sensor chain.
  2. Record clean palms at the same urban sites, on the same days, with the same devices as verified infested palms (inspection-confirmed).
  3. Characterise the sensor noise floor with no tree attached.
  4. Report within-site, within-device metrics alongside overall ones.

## Scripts added today

`folder_diagnostics.py`, `click_feature_cv.py`, `band_occlusion.py`, `rolloff_check.py`,
`within_device_eval.py`, `threshold_rules_study.py`, `noise_stress.py`, `esp32_compute.py`,
`device_calibration.py`. New training options in `esp32_train.py`, honoured by `esp32_export.py`:
`--max-band`, `--min-band`, `--gain-norm`, `--exclude-train-folders`, `--folds recorder`.
`gen_esp32_example.py --model` embeds another model and requantises the fixture; the firmware README
shows the command for the recommended model. New architecture `stream1s_w8_s2` (stride-2 ablation). `esp32_report.py` now
prints within-recorder AUC.
