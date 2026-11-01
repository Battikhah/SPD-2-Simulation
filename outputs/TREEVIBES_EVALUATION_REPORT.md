# TreeVibes training and evaluation report

**Prepared:** 1 November 2026, Dubai time
**Project:** SPD-2 Simulation
**Scope:** Full real-data training, validation, and an audit of readiness for final evaluation.

## Findings

The full real-data campaign completed on 1 November at 03:16 Dubai time: **950 successful trials and zero recorded failed trials**. It used 3,806 curated field recordings. The executed notebook records PyTorch 2.5.1+cu121 and `device: cuda`; 42 neural trials, 900 classical trials, and eight portable int8 trials completed. A prior real-data smoke run also completed without failed trials.

The strongest unrestricted validation candidate was a log-mel CNN. At its validation-selected threshold it detected 495 of 774 positive-labelled recordings (63.95% recall) and incorrectly flagged 14 of 292 negative-labelled recordings (4.79% false-positive rate). Its precision was 97.25% on this validation set. These are validation results selected from many trials; they are not independent final-test results or proof of performance in the field.

| Candidate | Validation recall | Validation false-positive rate | Validation counts | Practical status |
|---|---:|---:|---|---|
| Log-mel CNN, unrestricted | 63.95% | 4.79% | 495/774 positives detected; 14/292 negatives flagged | Neural model; CUDA training confirmed |
| Polynomial SVM, MFCC features | 55.04% | 4.79% | 426/774; 14/292 | Server-side candidate |
| Portable int8 logistic model | 10.72% | 4.79% | 83/774; 14/292 | Export candidate; no measured board qualification |

The threshold was chosen on the validation data under a 5% false-positive-rate ceiling. Comparisons are on the same 1,066 validation recordings. The low recall of the export candidate means it missed about 89% of validation positives at that threshold. These metrics use weak folder-level acoustic labels for wood-borer activity; they do not verify red palm weevil species or tree health. Operational performance at other locations and prevalence levels is unknown.

## Evaluation audit

The grouped manifest contains 3,806 recordings: 1,476 training, 1,066 validation and 1,264 test. Checks found no grouped component, exact decoded-audio hash, recorded device ID, or recorded site ID shared across splits. This verifies the available recorded identifiers, not unrecorded ancestry. There are only seven independent grouped components in total (three training, two validation, two test), which materially limits confidence about generalization. Another 348 recordings were excluded for unknown or ambiguous parent lineage.

The near-duplicate diagnostic lists **89 candidate pairs, all marked UNREVIEWED**. It is a screening list, not proof that any pair is a duplicate or that all potential relationships were found. Source ancestry and near-duplicate relationships have not been established sufficiently to sign off the split. The final holdout has therefore **not been evaluated**; its consumption marker is absent. No final-test accuracy, recall or false-positive rate is claimed.

The draft selection includes the three distinct validation winners shown above, with their validation thresholds. These are draft selections, not a signed review or a final test report. No model is marked ESP32 hardware-qualified; model size and host timings do not establish on-device performance.

## Required next steps

1. Review source-recording ancestry and all flagged near-duplicate candidates against original evidence, without using holdout model scores. Establish whether any recordings across splits share a parent, device, site or other dependency that the manifest missed. The available file and audio hashes alone cannot prove independence.
2. If grouping must change, retire this split and create a new manifest and campaign. If the review supports the existing grouping, document the evidence and reviewer in a split-bound lineage review.
3. Freeze a small candidate selection and the existing validation thresholds before using the final holdout. Follow the notebook's one-time final-evaluation procedure and preserve its marker.
4. Independently measure the selected embedded model on the target board, including firmware build, memory, latency and energy. Obtain independently labelled recordings and external-site evaluation before drawing deployment or species-specific conclusions.
5. Playback-audio regeneration remains outstanding; the full source dataset is available locally for a hash-verified regeneration.

## Evidence and reproducibility

- Executed notebook: `outputs/real-training-cuda/TreeVibes.full.ipynb`.
- Run directory: `outputs/real-training-cuda/full/treevibes-984e68c4e99668c8/`.
- Results: `leaderboard_validation.csv`, `track_winners_validation.csv`, `selection_draft.json` and individual trial artifacts.
- Audit inputs: `manifest.csv`, `splits.csv`, `split_lock.json`, `exclusions.csv`, `near_duplicate_candidates.csv` and `lineage_review_template.json`.
- Manifest SHA-256: `b6315ec10cb2b201056bd5f44b99d20ed397baaac0a012b99b8aed8ab4037fe0`.
- Split SHA-256: `1fd44d140599d5ae539a217f2ac9cf0f5a3a28f943074fdb9bd3076833f9dfcf`.

Counts and split checks were independently computed from the saved CSVs. The notebook, trial artifacts and source files remain available locally. This report does not imply that the outstanding lineage review was completed.
