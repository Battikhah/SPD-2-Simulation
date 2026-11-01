# Comparison with ML-for-Senior and reference-model experiments

**1 November 2026**

## What the earlier project reported

The user's public repository `https://github.com/Battikhah/ML-for-Senior` reports a temporal_w64 validation recall of 98.17% at 5.56% false positives. Its full-recording review separately reports 95.83% test recall (92/96 positives) with 22 false positives; the test result has already been consumed. These reported numbers are from that project's own dataset and protocol and their generated evidence artifacts are not in its current checkout. They cannot be directly compared to the present grouped validation set. Its README states that the split isolates duplicate recordings but does not prove unseen tree/session generalization. Its data code defaults to individual file and duplicate grouping when explicit grouping is unavailable, whereas the current campaign groups recorded devices and sites as well as duplicates. The earlier workflow optimizes validation F1; the present comparison constrains validation FPR to at most 5%. Both projects use weak audio annotations and do not establish species-confirmed RPW identity.

## Experiments on the current grouped split

All experiments use the same 1,476 train and 1,066 validation recordings, a 5% validation false-positive ceiling, 774 positive and 292 negative validation labels, and no final test audio. These results are exploratory because the validation set has been used repeatedly.

| Approach | Recall | False-positive rate | Positive detections | False alarms |
|---|---:|---:|---:|---:|
| Existing log-mel CNN | 63.95% | 4.79% | 495/774 | 14/292 |
| Prior CNN + AST blend | 71.58% | 4.45% | 554/774 | 13/292 |
| Exploratory CNN + AST + reference blend | 72.22% | 4.79% | 559/774 | 14/292 |
| Best gain-normalized full-recording temporal model | 24.68% | 4.79% | 191/774 | 14/292 |
| Best first-10-second temporal model | 7.88% | 4.79% | 61/774 | 14/292 |

For the first-10-second experiment, the old reference 100–2500 Hz bandpass, 30×13 MFCC frontend and temporal_w64 architecture were reproduced in PyTorch CUDA; the old SNR exclusion was disabled so all existing validation recordings remain in scope. Three seeds were trained with train-only feature scaling and validation-loss checkpoint selection. At the F1-maximizing validation threshold each of the three models predicted every validation recording positive, giving 100% recall and 100% FPR. This result illustrates why recall alone does not characterize detection quality.

The full-recording experiment applies 10-second windows with 5-second spacing plus a unique final window, first using original gain and then per-window RMS gain normalization to 0.01 after bandpass. It trains a temporal_w64 network with recording-level max-logit pooling and validation-loss checkpoint selection, three seeds per variant. A raw first-window MFCC matrix was checked against the old repository's extractor and matched exactly in the installed environment. We did not reproduce the previous project's actual warm-start weights, exact split, pretrained checkpoints, historical package environment or SNR-based exclusions; its checkpoints are not present in the checkout. Therefore these are architecture/frontend adaptations, not a literal replay of its claimed metrics.

The three-model blend was selected from multiple seed/weight combinations on the same validation set. Its formula is 0.95 × (0.80 × CNN + 0.20 × AST logistic) + 0.05 × raw full-recording temporal seed 43, using validation threshold 0.728558 (full precision in the CSV). Its small observed gain over the two-model blend is five additional detected validation positives and one additional false alarm, and is particularly vulnerable to selection bias. It is not a deployable packaged model or independent proof of improvement.

## Key limitation and next steps

The current training split has three source components, and every component contains just one label class. This means class can be confounded with source-specific patterns. Only two components exist in validation, one containing both labels and one containing positives only. There are 89 unreviewed near-duplicate pairs. No parameter search can establish performance on new trees under these conditions. Collect independently labelled recordings spanning multiple devices, trees, sites and both classes within multiple groups, then evaluate a frozen candidate on independent groups. Review recording ancestry before considering the existing final holdout; do not fabricate a lineage sign-off.

## Evidence

Reference checkout: `outputs/reference-ml-for-senior/`; reference comparison inputs are its `docs/deployment-options.md`, `docs/full-recording-review.md`, `treevibes/data.py` and `treevibes/models.py`. Experiments, model files, logs and detailed validation CSVs: `outputs/reference-reproduction/`. `reference_blends_validation.csv` contains the blend grid. Previous CNN/AST results are in `outputs/model-improvement/`. The final holdout has not been used for these experiments.
