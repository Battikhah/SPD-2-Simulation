# Destination PC setup — 31 October 2026

## Ready

- Project-local `.venv` uses Python 3.12.16. Simulator, test, lint, audio, notebook execution, scikit-learn, matplotlib, openpyxl, joblib and PyTorch dependencies are installed. Optional TensorFlow was not installed; TFLite remains disabled as configured by the notebook.
- Simulator running at http://127.0.0.1:8501; its health endpoint returned `ok`. PID: `outputs/streamlit.pid`. Logs: `outputs/streamlit.stdout.log` and `outputs/streamlit.stderr.log`.
- All 90 simulator tests passed. Application lint passed. Audio DSP selfchecks passed. Both burst-extractor tests passed. `pip check` reported no broken requirements.
- Synthetic training smoke notebook completed: 9 successful trials, including PyTorch and portable int8 models, zero failed trials. Results are software checks, not evidence of real-data detection performance.
- Executed notebook: `outputs/device-setup-smoke/TreeVibes_Three_Track_Training.smoke.ipynb`. Artifacts: `outputs/device-setup-smoke/SYNTHETIC-8fc483c60467b409/`.
- Original training notebook unchanged. Lineage review and final-evaluation gates preserved; no final holdout marker was created.
- Installed package snapshot: `outputs/device-setup-requirements.txt`.

## Blocked by Kaggle source availability

370 of 2,064 selected playback source clips were downloaded and their SHA-256 hashes verified. Initial individual downloads worked, then Kaggle returned HTTP 429 rate limits. The downloader was stopped. A later missing-file retry returned HTTP 404. Both versioned and unversioned full-dataset downloads also returned HTTP 404. No downloader or training campaign remains running.

Playback derivatives and bursts have not been regenerated because their complete source inputs are unavailable. Real-data training has not run because the original dataset could not be downloaded. Do not use the partial playback inputs as a complete training dataset. Download diagnostics are in `outputs/audio-preparation.log` and `outputs/dataset-download.log`; `status.json` contains the interrupted downloader's last checkpoint and is not a completion receipt.

## Resume in PowerShell

Python UTF-8 mode is required on this Windows PC: the notebook's default text reads otherwise fail with a cp1252 UnicodeDecodeError.

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

The simulator is already running for this session; the command above is for a later restart.

Once Kaggle access is restored, resume playback regeneration (existing hash-verified sources are reused):

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe outputs/treevibes-exciter-2026-10-30/prepare_audio.py
if ($LASTEXITCODE -eq 0) {
    .\.venv\Scripts\python.exe outputs/treevibes-exciter-2026-10-30/rpw_bursts/extract_bursts.py
}
```

For real-data training, supply the full original TreeVibes dataset and set `TREEVIBES_DATA_ROOT` to its folder before launching the notebook kernel. Start with `TREEVIBES_PRESET=smoke` and `TREEVIBES_SYNTHETIC=0`; use a different `TREEVIBES_OUTPUT_ROOT` from the synthetic run. Run one campaign at a time and retain the notebook's review and evaluation gates.

Generated files and the environment are ignored by Git; tracked repository files remain unchanged.

## Continuation — 31 October 2026

Kaggle full-dataset download became available and was started through `kagglehub.dataset_download('potamitis/treevibes')`. It was still downloading when this entry was written; it is not yet verified or a completed dataset. A detached supervisor (`outputs/real-training-pipeline.pid`) waits for that download to exit, then runs `outputs/real_training_pipeline.py`. The pipeline retries acquisition if needed, checks for the workbook and at least 1,000 WAVs, runs real-data smoke and then full training sequentially, and stops if a notebook or any trial fails. The notebook's existing grouping, lineage, and final-test gates remain in force; no final review or holdout evaluation is fabricated.

Progress: `outputs/real-training-pipeline.log` and `outputs/real-training/pipeline-status.json` (the latter appears once the supervisor starts the pipeline). Executed notebooks and model artifacts will be under `outputs/real-training/`. The two-hour Codex heartbeat `Continue TreeVibes training` checks and continues the task; it must not launch duplicate campaigns. The PC must remain powered on, awake, and connected, and Codex scheduling must be available for heartbeat checks. If download or training fails, inspect the logs and status before resuming; the script intentionally halts on errors. Training completion alone is not final holdout validation or proof of species identification.

### Latest verified progress

The 6.08 GiB archive completed and extracted successfully to `C:\Users\DeskEB2117-3.ESB-0036-20\.cache\kagglehub\datasets\potamitis\treevibes\versions\1`. The pipeline found one `TreeVibes.xlsx` and 58,797 WAV files and started real-data smoke training at 22:17 Dubai time. Full training is queued, not yet started; the supervisor starts it only after smoke finishes with one validation leaderboard and no `failure.json` files. Monitor `outputs/real-training/pipeline-status.json` and `outputs/real-training-pipeline.log`. A temporary keep-awake process runs until the supervisor exits; it does not protect against shutdown, restart, lost power, or a closed scheduler. Playback generation remains outstanding and can reuse source clips from the extracted dataset after checking each selected SHA-256.

### GPU continuation

Real-data smoke completed at 22:22 on 31 October with zero failed trials; its notebook and leaderboard are under `outputs/real-training/smoke` and `outputs/real-training/TreeVibes.smoke.ipynb`. The initial CPU full run was intentionally stopped shortly after starting. The active full campaign restarted at 22:24 with CUDA-enabled PyTorch 2.5.1+cu121, installed separately at `outputs/torch-cu121`. CUDA availability, RTX 2070 detection, and a GPU matrix multiplication were verified. The notebook selects CUDA for neural trials; preprocessing and scikit-learn trials remain CPU work. Its early pipeline stages may therefore show zero GPU utilization.

The active status is `outputs/real-training-cuda/pipeline-status.json`, with log `outputs/real-training-cuda-pipeline.log`, artifacts under `outputs/real-training-cuda/`, and supervisor PID in `outputs/real-training-pipeline.pid`. The earlier `outputs/real-training/pipeline-status.json` reflects the stopped CPU run and is stale. Never start a duplicate campaign. To resume after a failure, inspect the CUDA log and artifacts first and set `PYTHONPATH` to the absolute `outputs/torch-cu121` path before running `outputs/real_training_cuda_pipeline.py`. Preserve final holdout and lineage gates.

### Full campaign completed — 1 November 2026

The full real-data campaign completed at 03:16 Dubai time: 950 successful trials, zero failed trials. The executed notebook confirms `torch 2.5.1+cu121` and `device: cuda`. Outputs: `outputs/real-training-cuda/TreeVibes.full.ipynb` and `outputs/real-training-cuda/full/treevibes-984e68c4e99668c8/`, including `leaderboard_validation.csv`, `track_winners_validation.csv`, `selection_draft.json`, and `lineage_review_template.json`. Results are validation-selected weak-borer metrics; none is ESP32 hardware-qualified or species-confirmed RPW. The final holdout has not been evaluated: no FINAL_TEST_CONSUMED.json exists. The notebook requires a documented lineage review and frozen candidate selection before final evaluation; follow its instructions and do not fabricate sign-off. Playback audio regeneration remains outstanding and is separate from the completed training campaign. The training supervisor and temporary keep-awake process have exited.

### Evaluation audit and report — 1 November 2026

Wrote `outputs/TREEVIBES_EVALUATION_REPORT.md` and emailed the report to the connected School account. Audit: 950 successful trials, no failure records, CUDA confirmed; the validation winner log-mel CNN detected 495/774 positives and flagged 14/292 negatives. All 89 diagnostic near-duplicate pairs remain unreviewed. Exact PCM hashes, recorded components, device IDs and site IDs do not cross splits, but the source has only seven independent components and available metadata cannot prove parent ancestry. The final holdout remains unconsumed pending evidence-backed lineage review and frozen selections. Do not mark the review complete without that evidence.

### Model-improvement experiments — 1 November 2026

In response to the request for stronger models, exploratory GPU experiments are running on train and validation only; the final test remains untouched. First: AudioSet-pretrained MIT AST (`f826b80d28226b62986cc218e5cec390b1096902`) frozen embeddings, 12 classifier configurations; progress `outputs/model-improvement/experiment.log`, results `validation_results.csv`, exit code `exit-code.txt`. Second: speech-pretrained `facebook/wav2vec2-base-960h` (`22aad52d435eb6dbaf354bdad9b0da84ce7d6156`), same protocol and classifier grid, queued after successful AST; logs under `outputs/model-improvement-wav2vec/`. Speech pretraining is an exploratory out-of-domain comparison. Supervisor PIDs and processes should be inspected before restarting; the second waits for the first, and a keep-awake process is active. GPU feature extraction uses two padded 10-second segments resampled from 8kHz to 16kHz, training and validation only, and checks input file hashes. Existing validation has been searched extensively, and seven total independent source groups and 89 unreviewed near-duplicate candidates remain serious limitations. Any improvements are exploratory and do not resolve lineage or establish RPW identification. Continuation heartbeat: `improve-treevibes-model`.

### Exploratory pretrained-model improvement

Both CUDA experiments completed with exit code 0. AST and Wav2Vec2 each used 1,476 train and 1,066 validation recordings, with test audio untouched; standalone best validation recalls were 21.45% and 8.40%, respectively, at 4.79% FPR. The exploratory best 80% baseline CNN / 20% AST logistic blend reached 554/774 (71.58%) validation recall and 13/292 (4.45%) FPR versus 495/774 (63.95%) and 14/292 (4.79%) for baseline. Weight, classifier and threshold were chosen on the already heavily reused validation set, so this is not independently validated and can be optimistic. Critically, all three training components are single-class; group and label are confounded. Full details and limitations: `outputs/model-improvement/REPORT.md`; blend specification and predictions are in the same directory. The test holdout remains untouched and lineage review outstanding. This ensemble is experimental and not integrated or hardware-qualified. The `improve-treevibes-model` heartbeat can continue exploring further techniques.

### ML-for-Senior comparison and reference architecture — 1 November 2026

Reference repository cloned at `outputs/reference-ml-for-senior/`. Its documentation reports 98.17% validation recall at 5.56% FPR using file/duplicate grouping (without verified tree/session grouping); it selects validation F1. Its generated historical model artifacts are not in this checkout. Reproduced its reference MFCC and temporal_w64 architecture on the existing grouped train/validation split, three seeds: best first-10s recall 7.88% at 4.79% FPR. Full-recording max-pool raw/gain-normalized experiment (six fits) achieved best 24.68% at 4.79% FPR. First raw MFCC window matched the reference extractor exactly in the installed environment. A validation-selected CNN+AST+reference temporal blend achieved 559/774 (72.22%) at 14/292 (4.79% FPR), versus prior CNN+AST 554/774 at 13/292. This heavily validation-selected difference is not an independent result. The active experiment completed successfully; no final test audio was used. Full report: `outputs/reference-reproduction/REPORT.md`; per-seed results and predictions are in the same folder. Continue with independent source groups and lineage review; three training components are individually single-class.

### Acoustic validity audit — 1 November 2026

See docs/ACOUSTIC_DATA_AUDIT.md and outputs/acoustic-validity-audit/. The notebook now merges supplied session and confirmed duplicate-group identifiers and no longer treats text NA coordinates as a known shared site. The old lock was preserved; previous site completeness claims are corrected because NA coordinates were incorrectly hashed. All 3 train components remain single-class in the original lock; 65 of 89 unreviewed development near-duplicate candidates cross train/validation. A train-only simple audio baseline and source shortcut probe are documented. No new training or final-test evaluation was justified. Independent labelled sources and original lineage evidence are the blocker. The former model-optimization heartbeat was paused to prevent further validation overfitting; resume only when relevant new evidence is available.

### Development-only duplicate and noise investigation

All 65 train/validation candidate pairs were compared using hash-verified waveforms and metadata. Their peak absolute normalized waveform correlations over overlaps of at least five seconds were 0.021–0.185; no direct/shifted/gain-scaled duplicate was confirmed. This is not a clearance of lossy/short or shared-parent relationships; candidates stay UNREVIEWED. See `outputs/acoustic-validity-audit/cross_split_pair_evidence.csv`. Original paper defines AUDIO as human listening and CONFIRMATION as trunk-slicing to locate larvae; this refines prior provenance notes but does not establish per-clip species labels or complete ancestry. A fixed 125-recording validation subset under synthetic 10 dB white-noise perturbation changed 68 CNN predictions; scores reproduced for the unperturbed model, and no threshold was adjusted. See `docs/ACOUSTIC_DATA_AUDIT.md` and `noise_sensitivity.csv` for limits. No final holdout audio or scores used; no new training justified. The meeting view has a separate `meeting_app.py`; app.py still opens the full dashboard.
