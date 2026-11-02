# ESP32 overnight sweep: report (1–2 November 2026)

> **Update, 2 November afternoon:** see `docs/ESP32_DEV_DAY_REPORT.md`. The figures below use folder-grouped
> folds, which leak recorder identity. On this dataset the models mostly use the 3–4 kHz band, loudness and
> recorder or site differences. Use the recorder-held-out results and the 97%-recall threshold rule from
> that report instead. The recommendation below is superseded.

Read this first. Details are in `docs/ESP32_TFLM_PLAN.md`; raw numbers are in `outputs/esp32-tflm/sweep/leaderboard.csv` and `report_tables.md` (git-ignored, on the training PC).

## TL;DR

- **Recommended model:** `stream1s_w16` with pos_boost 2. It is a 1 s log-mel CNN that runs on the ten 1 s windows of each 10 s capture and averages them on the MCU. It is 34.7 KB as int8 TFLite, with an arena lower bound of 74 KiB, and uses the same 7 TFLM ops and the same input quantization as the existing firmware example.
- **Recommended operating point:** the **FPR≤10% rule**, with the threshold chosen on the other folds. With fresh seeds 3–5 it gives **82.5% recall, 91.2% precision and 12.7% FPR**; the seeds used for selection gave 79.1% / 90.8% / 12.9%. This point was fixed before looking at the confirmation seeds, and no further tuning was done.
- **If memory is tight (37 KiB arena):** use `stream1s_w8` with pos_boost 1, trained **folder-balanced (p=0.5)**. It is 13.8 KB with a 37 KiB arena, and gives **80.3% / 91.1% / 12.6%** at the same rule. It gave 80.3% / 91.0% / 12.8% again with fresh seeds. It uses the same network as the model already wired into `firmware/esp32_stream1s/`; only the weights differ. Its F2 point is worse (about 30% FPR), so use it only with the FPR≤10% rule.
- **Nothing helped as much as hoped.** About 50 configurations were tried. The top dozen differ by under 0.015 AUC, and none moved the same-device clean folders 24 and 25. Every model still flags all of their recordings.
- **Nothing has run on an ESP32 yet.** Arena, latency and energy are unmeasured. All figures come from development cross-validation, and the sealed test split was never read.

## What was done overnight

All runs used the same folder-grouped 5-fold cross-validation on the 2,542 development recordings (1,567 infested and 975 clean). Each held-out fold is scored with the **int8 TFLite** model, so the figures reflect device arithmetic. The threshold is chosen on the other four folds, never on the fold it is reported on.

| Phase | What | Jobs | Time |
| --- | --- | ---: | --- |
| 1 | 9 architectures × pos_boost {0.5, 1, 2}, 3 seeds each | 27 | 21:10–02:50 |
| 2 | Top 3 by AUC retrained with **new seeds 3–5** (is the ranking luck?) | 3 | 02:50–04:25 |
| 3 | Improvements on models that fit a 64 KiB arena: folder balancing, distillation (KD), KD + balancing, KD with α=0.2 | 8 | 04:26–07:50 |
| Extra | Seed confirmation of the best phase-3 result | 1 | 07:52–08:07 |

Every job finished; none failed. There were two fixes during the night:

1. The report script's independent recount was out by one false alarm. A float64 versus float32 mean broke a score tie differently. Trained models and metrics were unaffected.
2. I limited phase 3 to models with an arena of at most 64 KiB (the example firmware's arena). Without this it would have spent the morning on the 74 KiB and 87 KiB models, which train 2–3× slower.

Status emails went out every 2 hours in the same thread.

## Results

### Best configurations

All rows use int8, a 3-seed ensemble and cross-fitted thresholds. "F2" weights recall twice as much as precision. "FPR≤10%" gives the highest recall whose FPR on the other folds is at most 10%.

| Configuration | .tflite | Arena (lower bound) | AUC | F2: recall / precision / FPR | FPR≤10%: recall / precision / FPR |
| --- | ---: | ---: | ---: | --- | --- |
| **stream1s_w16 pb2, seeds 3–5** | 34.7 KB | 74 KiB | **0.939** | 99.3% / 90.0% / 17.7% | **82.5% / 91.2% / 12.7%** |
| stream1s_w16 pb2, seeds 0–2 | 34.7 KB | 74 KiB | 0.932 | 93.4% / 86.9% / 22.6% | 79.1% / 90.8% / 12.9% |
| stream1s_ds pb2, seeds 3–5 | 28.5 KB | 87 KiB | 0.933 | 99.4% / 89.5% / 18.7% | 81.5% / 90.7% / 13.4% |
| stream1s_w8 pb1, folder-balanced p=0.5 | 13.8 KB | 37 KiB | 0.934 | 89.2% / 83.1% / 29.2% | 80.3% / 91.1% / 12.6% |
| the same, seeds 3–5 (confirmation) | 13.8 KB | 37 KiB | 0.922 | 87.6% / 81.0% / 33.0% | 80.3% / 91.0% / 12.8% |
| stream1s_w8_d4 pb2, KD | 35.3 KB | 37 KiB | 0.931 | 99.2% / 87.3% / 23.3% | 79.2% / 90.3% / 13.7% |
| stream1s_w8_d4 pb2 | 35.3 KB | 37 KiB | 0.928 | 98.4% / 87.1% / 23.4% | 77.1% / 90.0% / 13.7% |
| stream1s_w8 pb1 | 13.8 KB | 37 KiB | 0.926 | 93.5% / 87.9% / 20.6% | 74.2% / 89.7% / 13.6% |
| stream1s_w4 pb2 | 7.8 KB | 18 KiB | 0.925 | 94.4% / 87.4% / 21.8% | 75.3% / 89.5% / 14.2% |
| dscnn10s_w16 pb0.5 (best 10 s input) | 20.4 KB | 61 KiB | 0.908 | 98.0% / 76.2% / 49.1% | 72.7% / 90.5% / 12.2% |

### Trade-off by pos_boost

pos_boost is the extra weight on infested examples in the loss.

| Model | pos_boost | AUC | F2: recall / precision / FPR | FPR≤10%: recall / precision / FPR |
| --- | ---: | ---: | --- | --- |
| stream1s_w8 | 0.5 | 0.916 | 93.9% / 83.8% / 29.1% | 69.3% / 89.2% / 13.5% |
| stream1s_w8 | 1 | 0.926 | 93.5% / 87.9% / 20.6% | 74.2% / 89.7% / 13.6% |
| stream1s_w8 | 2 | 0.922 | 98.9% / 87.0% / 23.7% | 72.7% / 89.6% / 13.5% |
| stream1s_w16 | 0.5 | 0.923 | 91.4% / 83.6% / 28.8% | 76.1% / 90.1% / 13.4% |
| stream1s_w16 | 1 | 0.930 | 98.1% / 85.8% / 26.2% | 79.2% / 90.5% / 13.3% |
| stream1s_w16 | 2 | 0.932 | 93.4% / 86.9% / 22.6% | 79.1% / 90.8% / 12.9% |
| stream1s_w4 | 0.5 / 1 / 2 | 0.899 / 0.904 / 0.925 | 87.6–94.4% recall, 21.8–42.9% FPR | 60.9% / 63.6% / 75.3% recall at ~14% FPR |

pos_boost 0.5 was never best. Down-weighting infested examples does not reduce false alarms; it only costs recall. pos_boost 1 and 2 are close.

### Trade-off by threshold rule

For the recommended model (stream1s_w16 pb2, seeds 0–2, int8):

| Rule | Recall | Precision | FPR | Precision if 20% of trees infested | if 5% infested |
| --- | ---: | ---: | ---: | ---: | ---: |
| F2 | 93.4% | 86.9% | 22.6% | 50.9% | 17.9% |
| F1 | 92.5% | 88.1% | 20.0% | 53.6% | 19.6% |
| F0.5 | 87.4% | 88.7% | 17.8% | 55.0% | 20.5% |
| **FPR≤10%** | **79.1%** | **90.8%** | **12.9%** | **60.5%** | **24.4%** |

**Why FPR≤10% is recommended:** it is the only rule whose realised FPR is predictable. Across every 1 s configuration it landed at 12.6–14.5%. The F2 point swung between 17.7% and 43% FPR from one configuration to the next, because folder-to-folder score shifts move the threshold. The target of 10% becomes about 13% on unseen folders; plan for 13%.

### What helped and what did not

- **Capacity helped a little.** Going from 8 to 16 channels added about 0.006–0.013 AUC and about 5–8 points of recall at the FPR≤10% point, and the gain held on fresh seeds. This is the only improvement that replicated on every metric.
- **Distillation from a 32-channel teacher gave a small or inconsistent gain.** stream1s_w8_d4 rose from 0.928 to 0.931, but stream1s_w8 fell from 0.926 to 0.923 with KD. Treat this as noise.
- **Folder-balanced training** (down-weighting the large clean folder 9 and up-weighting small folders) was mixed:
  - stream1s_w8: AUC rose from 0.926 to 0.934.
  - stream1s_w8_d4: no change.
  - 10 s dscnn: AUC fell from 0.908 to 0.851.
  - Seed confirmation for stream1s_w8 (seeds 3–5): the AUC gain did **not** replicate (0.922). The recall gain at the FPR≤10% point did replicate: 80.3% / 91.0% / 12.8%, the same as seeds 0–2, versus 74.2% unbalanced. Its F2 point got worse on both seed sets (29–33% FPR). Balancing seems to separate the clearest infested recordings better without improving overall ranking. This is useful only at a strict-FPR operating point.
- **10 s-input models are clearly worse** (AUC 0.84–0.91) than 1 s windows averaged over 10 s (0.89–0.94). They also give the same-device folder 24 a score of about 0.99.
- **Top-3-of-10 window aggregation** was always worse than plain averaging.
- **int8 conversion costs almost nothing:** at most 0.003 AUC.

### Where the errors are

For the recommended model at the F2 point:

| Clean folder | Recordings | Flagged as infested |
| --- | ---: | ---: |
| folder_8 (wind/rain) | 238 | 36% |
| folder_7 (wind/rain) | 155 | 47% |
| folder_24 (same device as infested recordings) | 27 | **100%** |
| folder_25 (same device as infested recordings) | 27 | **100%** |
| folder_9 | 528 | 1.5% |

At the FPR≤10% point, folder 8 drops to near zero. The remaining false alarms are folder 7 and folders 24/25. Missed infested recordings come mostly from folder 33 (13% missed at F2).

**The key finding remains:** every model flags every recording of the same-device clean folders 24/25 as infested. Per your assumptions (other-species clean recordings are valid negatives, and all recordings are acoustically comparable), this is reported as a check, not treated as disqualifying. However, it is the strongest sign that part of what the models learn is recording or device conditions rather than infestation. No training trick changed it; only new same-device, same-site clean recordings can test it.

## What this means for the ESP32

- **Memory:** the plan's arena target is at most 96 KiB. `stream1s_w16` (74 KiB lower bound, before planner overhead) fits that target, but **not** the 64 KiB `kArenaSize` in `firmware/esp32_stream1s/main/main.cc`. Raise the arena to about 96 KiB before using it. The real arena is only known after `interpreter.arena_used_bytes()` on a board.
- **Three-model ensemble:** run the three seed models sequentially on each 1 s window, reusing one arena. That means 3 × 34.7 KB of flash, and three times the latency of one model. Latency is unmeasured.
- **Front end:** the C log-mel (`firmware/logmel/`) is unchanged. The w16 models use the same input scale and zero point (0.09447, 116) as w8, and real-window int8 parity with Python was exact on the host.
- **Switching the example to w16:** regenerate `main/generated/` from the w16 directory with `scripts/gen_esp32_example.py` (after adding a selector for the model directory) and enlarge the arena.

## Honesty notes

- Development cross-validation only. The test split was never read, and normalization and quantization were fitted on training folds only.
- About 50 configurations were compared, so the best rows are optimistic. The seeds 3–5 replication is the least biased number here.
- Labels are weak, folder-level labels, not individually verified infestations.
- Precision depends on prevalence. At realistic field prevalence (5–20% of trees infested), the recommended point gives about 60% precision at 20% prevalence (3 true alarms for every 2 false ones). At 5% prevalence it gives about 24% (1 true alarm for every 3 false ones).
- Nothing was measured on an ESP32: not arena, latency, energy, nor `logf` parity on the board.

## Next steps

1. Build `firmware/esp32_stream1s` on a real ESP32-WROOM32. Record arena used, latency per window and per 10 s decision, and current draw. Then try w16 with a 96 KiB arena.
2. Collect same-device, same-site clean recordings, especially from the folder 24/25 setup and windy or rainy conditions. This is the only thing that can fix the dominant error, which training cannot.
3. Once those exist, run the frozen recommendation through the notebook's one-time final-evaluation gate. Do not tune further on this development set.
