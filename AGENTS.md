# Agent handoff: palm sensing and acoustic detection

Last updated: 1 November 2026. Read this file before making changes. Claude Code may not automatically load `AGENTS.md`; give it this file explicitly or import it from its own instructions.

## Project and priorities

This repository contains a Streamlit digital twin for palm monitoring and a separate TreeVibes acoustic training/evaluation workflow. The user wants useful, autonomous progress, but **scientific validity, truthful labels and leakage-safe evaluation take priority over model scores**. Distinguish simulated assumptions, weak acoustic labels, validation results, biological verification and hardware measurements. Do not present the simulator or a training metric as proof of field detection or ESP32 capability.

Before acting, read `docs/ACOUSTIC_DATA_AUDIT.md`, the latest entries of `outputs/DEVICE_SETUP_STATUS.md` when available, and `docs/DEVICE_HANDOFF.md`. The status file is chronological: earlier entries saying downloads or training are active/blocked are historical and superseded by later entries. Inspect `git status` and running processes first; preserve existing uncommitted user work and do not start duplicate campaigns or downloads.

## Important source and data locations

- `app.py`: default full engineering Streamlit dashboard; `engineering_app.py` implements it. `meeting_app.py` opens the separate concise meeting view; `simulator/` contains the models.
- `notebooks/TreeVibes_Three_Track_Training.ipynb`: data curation, grouped split, training and final-evaluation gates. `scripts/acoustic_validity_audit.py`, `scripts/review_audio_pairs.py`, and `scripts/acoustic_noise_sensitivity.py` are development diagnostics.
- `docs/ACOUSTIC_DATA_AUDIT.md`: current evidence, detailed limitations, per-source results, and collection plan. `docs/ACOUSTIC_TEST_PROTOCOL.md` is the physical playback protocol; `docs/HARDWARE_BASELINE.md` describes the hardware.
- On the current Windows PC, the original TreeVibes v1 dataset is under `%USERPROFILE%\.cache\kagglehub\datasets\potamitis\treevibes\versions\1` (58,797 WAVs plus workbook). The existing real-data campaign is under `outputs/real-training-cuda/full/treevibes-984e68c4e99668c8/`; the run contains the locked manifest/split, predictions, candidates and review template. Additional audit outputs are under `outputs/acoustic-validity-audit/`. The prior public reference checkout is `outputs/reference-ml-for-senior/` if present.
- **Transfer warning:** `/outputs/`, datasets, model weights, feature caches and virtual environments are generally Git-ignored. The audit documentation, scripts, notebook modifications, meeting entry point, tests and selected small status/split-evidence files are included in this handoff commit. A friend cloning the repository will not receive uncommitted work or ignored artifacts automatically. Inspect `git status` and arrange an authorized transfer or commit of required files; do not assume a clone has the data, model, split lock or results. Never publish private data or credentials.

## Acoustic validity and final-holdout rules

- The completed CUDA full campaign recorded 950 successful trials and no trial failures. Its CNN validation result was 495/774 positive detections and 14/292 false positives (63.95% recall, 4.79% FPR). An exploratory validation-selected CNN/AST blend gave 554/774 and 13/292. These are heavily reused validation results, not independent test performance, field reliability or confirmed red palm weevil identification. Do not keep tuning against this validation set as though it were fresh.
- The original training split has three components, each containing just one weak label; only one training component is negative. Class and source are confounded. Labels originate from folder-level human listening, not individually verified species/infestation labels. The original paper describes a separate `CONFIRMATION` inspection field; it does not supply complete per-clip verification or recording ancestry. A folder is not automatically a verified tree or session.
- **Preserve the final holdout:** do not load its audio/features for model development, evaluate it, inspect model scores/errors, change the existing locked split or manufacture lineage sign-off. Review ancestry without holdout model outputs; freeze a justified selection and threshold and follow the notebook's one-time final-evaluation gate only after evidence-backed review. If corrected relationships require a different split, retire the old split and create a new manifest/campaign; do not overwrite the existing lock. No final holdout marker was present at last check.
- 89 development near-duplicate pairs remain unreviewed, 65 across train/validation. All 65 were checked with hashes, metadata and normalized waveform alignment for overlaps of at least five seconds: no direct/shifted/gain-scaled duplicate was confirmed, but shorter, transformed or shared-parent relationships remain unresolved. Do not declare them cleared. The earlier manifest incorrectly hashed `NA` coordinates as a site; the notebook now treats missing/non-numeric/out-of-range coordinates as unknown. The corrected notebook can group supplied tree, session, device, site, parent and reviewed duplicate-group IDs transitively; never invent IDs or equate missing metadata with independence.
- On a fixed 125-recording validation subset, 10 dB synthetic white noise changed 68 frozen-CNN decisions; positive detections fell from 69/110 to 4/110. This is a stress test, not a natural-noise field estimate. Historical ML-for-Senior 98.17% recall / 5.56% FPR used a different protocol and was not reproduced here.
- Do not retrain merely to chase better scores. Original collection/inspection and parent/session logs and independently verified recordings from multiple trees/sites/devices with both classes and balanced noise conditions are needed. The audit proposes at least 10 positive and 10 negative target-host trees across three sites as an initial pilot, not proof of deployment readiness; species-specific claims need species confirmation. Reserve independent sites for future evaluation.

## Setup and verification

PowerShell on this PC:

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check app.py engineering_app.py meeting_app.py simulator tests scripts
```

Use `meeting_app.py` instead of `app.py` for the meeting view. The last complete verification reported 95 tests passing; scoped application/scripts lint passed. Repository-wide Ruff also reports existing notebook style violations. Re-run appropriate checks after changes; do not assert they still pass without running them.

For GPU diagnostics on this specific computer, CUDA-enabled PyTorch is installed separately in `outputs/torch-cu121` and the GPU is an RTX 2070. Set `$env:PYTHONPATH = (Resolve-Path outputs/torch-cu121).Path` and check `torch.cuda.is_available()` before assuming GPU availability. That ignored folder may not exist on another PC. See the notebook and `docs/DEVICE_HANDOFF.md` for installing additional audio/training dependencies; a synthetic smoke run is a software check only.

## Session log

- **1 November 2026 (later session, "continue on the models"):** Re-read this file, `docs/ACOUSTIC_DATA_AUDIT.md`, `docs/DEVICE_HANDOFF.md` and the status file. Git tree was clean; the only running Python processes were Streamlit and VS Code helpers (no training or downloads); no `FINAL_TEST_CONSUMED.json`. No new labelled data, lineage records or independent sites have arrived, so **no new acoustic training was run** (it would only add validation overfitting, per the audit). Re-verified: `pytest -q` 95 passed; scoped Ruff passed. Next unblock: independently verified multi-tree/site recordings and original ancestry logs (see audit collection plan). Non-training model work that is still valid: simulator model changes, feature/LoRa budget work, or documenting the pilot data-collection template.
- **1 November 2026 (simulator models):** Reviewed `simulator/engine.py`, `energy.py`, `network.py` and `docs/MODELS.md`. Fuzzed 400 random valid configs (no NaN, no served > reachable) and checked directional behavior (range vs TX power, sensitivity vs SF, battery life vs trigger rate/capacity, reach vs gateways); all consistent, no model bug found, so no model equations changed. Added `tests/test_model_invariants.py` (6 tests). These are internal sanity checks, not calibration; Free-space "range" of tens to hundreds of km is the documented idealized baseline. Unmeasured currents/efficiency remain assumptions. Known leftover: `network.py:287` `~active & ~fixed` gives a Python 3.16 bool-inversion DeprecationWarning (harmless with numpy arrays; check dtype before changing).

- **1 November 2026 (ML-for-Senior "insanely high recall"):** Explained, not reproduced: that repo's default split groups each file alone (`file_and_duplicate_only`), i.e. a random recording-level split. `scripts/split_leakage_demo.py` (dev data only) shows a fixed MFCC random forest scores 100% recall / 1.85% FPR under random recording-level folds but 65.79% / 62.56% when whole folders are held out. Do not adopt its recall figures or split as a target; keep folder/component-grouped splits. Details in `docs/ACOUSTIC_DATA_AUDIT.md`.

- **1 November 2026 (local GPU three-track campaign):** User asked for maximal models on three tracks (unrestricted; 30x13 MFCC; ESP32-WROOM32), trained locally (not Kaggle). `scripts/grouped_cv_campaign.py` runs folder-grouped 5-fold CV on dev data only (test never read), 3 seeds per network, GPU-resident batches with AMP (GPU ~96% during CNN training; earlier version was CPU-copy bound). Best: log-mel CNN AUC ~0.95, recall@5%FPR 39-75% depending on seed; a 6.5k-param log-mel CNN matches the 101k one; 30x13 MFCC models near chance. Same-device negative folders 24/25 are scored as infested, so the signal is likely host-tree/environment confounding, not verified detection. Results table in `docs/ACOUSTIC_DATA_AUDIT.md`; raw outputs in `outputs/grouped-cv-campaign/full/`. Do not tune further against these five negative folders; the blocker is same-host, same-site, same-device negative recordings.

- **1 November 2026 (whole-recording inputs, no FPR cap):** User asked to drop the 5% FPR cap and use more than 1 s context. `scripts/grouped_cv_fullclip.py` (+ shared trainer/uncapped metrics in `scripts/grouped_cv_round2.py`) adds whole-clip 30x13 MFCC (frames split into 30 chunks, mean/max pooled) and full 640x64 log-mel; recordings are ~20.48 s here, not 10 s. Balanced-point (pooled OOF, optimistic) best: track C 1 s log-mel CNN w8 3-seed ensemble 97.5% recall / 13.5% FPR / 93.3% acc; track A 1 s log-mel w32 86.1% / 7.9%; track B whole-clip MFCC CNN AUC 0.76 (was 0.52) at 91.3% / 38.4%. Same-device negative folder 24 is still scored infested; CMVN hurts. Round 2 (window-level SpecAug/CMVN) was stopped by the user after one seed and has no results. Table in `docs/ACOUSTIC_DATA_AUDIT.md`.

- **1 November 2026 (10 s inputs):** User required 10 s inputs for tracks A/C. `scripts/grouped_cv_10s.py` uses two 10.24 s log-mel segments (320x64) per recording, mean-pooled. 3-seed ensembles: A w32 AUC 0.931, 85.4% recall / 9.1% FPR at the balanced point; C w8 (25.7k) AUC 0.932, 82.8% / 8.9%; C w4 (6.5k) AUC 0.919, 79.5% / 10.7%. Roughly equal to 1 s windows and full 20 s clips; folder 24 still scored infested. Use 10 s inputs for tracks A/C going forward per the user.

- **1 November 2026 (ESP32 TFLite/TFLM models):** Plan and results in `docs/ESP32_TFLM_PLAN.md`. Limiting factors measured by `scripts/esp32_limits.py`: pooled-OOF thresholds were optimistic (cross-fitted FPR 23-30% vs 8-11%); arena, not weights, blocks the old 10 s CNNs (100-800 KiB). Pipeline: `scripts/esp32_arch.py` (shared spec), `scripts/esp32_train.py` (GPU), `scripts/esp32_export.py` (TF 2.18.1 CPU in ignored `outputs/tf-venv`; use reference kernels, as XNNPACK crashes on Windows). Best int8 candidate stream1s_w8 (13.8 KB, ~38 KiB arena, 3-seed ensemble): cross-fitted F2 point 98.9% recall / 87.0% precision / 23.7% FPR, AUC 0.922; int8 loss <=0.003 AUC. No board or ESP-IDF on this PC; on-device results remain unmeasured. Always report cross-fitted thresholds from now on.

- **1 November 2026 (FPR reduction, C front end, overnight sweep):** User assumptions recorded: (a) clean recordings from other tree species are acceptable negatives (similar water content/density to palm); (b) all recordings are treated as acoustically comparable (user checked). Keep reporting the same-device folder-24 check, but host species is not treated as a confound. Added `--pos-boost/--tag/--seeds` (esp32_train.py), `--tag/--metrics-only` and cross-fitted F1/F0.5/FPR<=10% rules plus top-3-of-10 window aggregation (esp32_export.py). C99 streaming log-mel front end in `firmware/logmel/` (tables generated by `scripts/gen_logmel_tables.py`; MSVC parity via `firmware/logmel/test/build_msvc.cmd`: real windows bit-identical int8, 0.062% of values 1 LSB off overall, only on a synthetic full-scale tone; state 6,184 B). Unbuilt ESP-IDF example in `firmware/esp32_stream1s/` (generated model/fixture git-ignored). **Overnight sweep** `scripts/esp32_sweep.py` (9 archs x pos_boost 0.5/1/2, then new seeds 3-5 for the top 3) writes `outputs/esp32-tflm/sweep/` (sweep.log, leaderboard.csv); it is resumable: rerun the same command and finished configs are skipped. Never run two sweeps at once. The many configurations make every selected figure optimistic.

- **2 November 2026 (overnight sweep results):** About 50 int8 configurations were completed with no failures. Phase 3 (`--phase3`) only improves models whose arena is ≤64 KiB; the queue stops starting jobs at 07:15. Report: `docs/ESP32_OVERNIGHT_REPORT.md`.
  - Recommendation: `stream1s_w16` pb2 at the cross-fitted FPR≤10% rule. Fresh seeds 3–5 give 82.5% / 91.2% / 12.7% (recall / precision / FPR). It needs about a 96 KiB arena; the firmware example currently allocates 64 KiB.
  - Fallback: `stream1s_w8` pb1 folder-balanced p=0.5. Recall at FPR≤10% (80.3%) replicated on seeds 3–5, but its AUC gain did not.
  - Distillation gave no gain that replicated. Same-device folders 24/25 are still always flagged.
  - `esp32_report.py` averages seed scores in float32 to match `esp32_export.py` tie-breaking.
  - Do not keep tuning on this development set; the next steps are on-board measurement and new same-device clean recordings.

- **2 November 2026 (dev day: what the models detect).** Report: `docs/ESP32_DEV_DAY_REPORT.md`. Working notes: `outputs/dev-day/NOTES.md` (ignored). All development data; the test split was never read.
  - **What the CNNs use.** They decide from the 3–4 kHz band, which sits just under Nyquist. Training on mel bands 56–63 alone matches all bands; without them AUC is about 0.6. Loudness also matters.
  - **A trivial rule matches them.** The near-Nyquist roll-off (`rolloff_check.py`) alone gives F1 0.965.
  - **Confounds** (`TreeVibes.xlsx`):
    - Recorder IMEI …8409 recorded only clean folders 7 and 9.
    - Folder 9 is the recorder's white noise floor.
    - Infested folders are all urban palms.
  - **Folder-grouped folds leak recorder identity.** The clean folder 24 is "failed" because the same recorder's infested folders are in training.
  - **Evaluate with `esp32_train.py --folds recorder`.** The fold ids are saved in `train_info.json` because the TF venv has no openpyxl, and `esp32_export.py` reads them.
  - **New training options:** `--max-band`, `--min-band`, `--gain-norm`, `--exclude-train-folders`.
  - **New diagnostics:**
    - `folder_diagnostics.py`
    - `click_feature_cv.py`
    - `band_occlusion.py`
    - `rolloff_check.py`
    - `within_device_eval.py`
    - `threshold_rules_study.py`
    - `noise_stress.py`
    - `esp32_compute.py`
  - **Reporting.** `esp32_report.py` prints within-recorder AUC.
  - **Threshold rule.** Use 97% training recall.
  - **Single model.** Deploy one model, not an ensemble.
  - **Firmware.** `gen_esp32_example.py --model` can embed another model and requantises the fixture.
  - **Per-device calibration.** A threshold transferred from other recorders flags all healthy trees on some devices. Calibrate each device on known-healthy trees (`device_calibration.py`).
  - **Current recommendation:** single int8 `stream1s_w8_d4` pb2, folder-balanced (p=0.5), calibrated at the 90th percentile of healthy-tree scores. It reaches about 86–90% recall at 14–15% false alarms on unseen recorders. These figures are rough and site-confounded.

- **2 November 2026 (scoring WAVs on a PC):** `scripts/score_wavs.py` (tf-venv) scores WAV files, directories or `folder_N` names with an exported int8 model through the training front end and the reference interpreter; it refuses sealed-test recordings, marks recordings the exported seed-0/fold-0 model trained on, and supports per-device calibration (`--calibrate-folder`, `--calibration-share 0.5`). For the recommended run, fold 0 held out recorder 2538 (folders 1, 3, 5, 11, 13, 16, 18, 20, 22, 26, 34 and clean 24); its scores match `oof_int8.csv` exactly. One calibration split there: 386/501 recall, 4/13 false alarms (the single-seed average over 200 splits is 76% / 16%).

If the requested work is blocked by missing evidence or data, state exactly what is missing and the next action; do not fabricate labels, grouping, tests, measurements or outcomes. Record meaningful progress and recovery steps in the audit/status files.
