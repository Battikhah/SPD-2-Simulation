# TreeVibes model-improvement experiments

**1 November 2026 — exploratory validation only**

## Result

A blend of the previous log-mel CNN and a frozen AudioSet-pretrained Audio Spectrogram Transformer (AST) embedding classifier reached **554/774 positive-labelled validation recordings detected (71.58% recall)** with **13/292 negative-labelled recordings incorrectly flagged (4.45% FPR)**. The previous CNN recorded 495/774 (63.95%) at 14/292 (4.79%). The absolute recall difference is 7.62 percentage points on the same validation set. This is a validation-selected comparison, not a test-set or prospective result; the same validation data was already used to search the original 950 trials and to select both models, blend weight and threshold. The apparent gain can be optimistic.

The blend uses 80% CNN validation probability and 20% AST logistic-C100 validation probability, threshold 0.740951. It is an experimental ensemble requiring both models and AST preprocessing; it is not yet integrated into the simulator or deployable on the embedded target. The original model remains intact. Validation predictions and exact coefficients are saved under `outputs/model-improvement/`.

## Comparisons

- AudioSet-pretrained AST, frozen embeddings, 12 classifier configurations: best stand-alone validation result 21.45% recall at 4.79% FPR. The weaker individual model nonetheless contributed to the selected blend.
- Speech-pretrained Wav2Vec2, frozen embeddings, 12 classifier configurations: best stand-alone result 8.40% recall at 4.79% FPR. Its pretraining is out of domain for wood vibrations. The best sampled blend of Wav2Vec2 and the CNN had 69.51% recall at 4.79% FPR.
- The CNN/AST blend selection explored eight weights per encoder and selected the pretrained classifier itself on validation. No claim of independent improvement is justified until new independent data are assessed.

## Data and evaluation constraints

Both experiments use 1,476 training and 1,066 validation recordings only. The 1,264 test recordings were not read by these experiments and the prior final-test marker remains absent. Input clips were hash-checked against the existing manifest, sampled at 8 kHz, resampled to 16 kHz, split into two 10-second segments with zero padding where needed, and the segment embeddings averaged. The two pretrained models ran on the RTX 2070 through CUDA. Audio pretraining and speech pretraining may be out of domain; no frequencies above the original 4 kHz Nyquist limit are added by resampling.

There are only three independent training components, and they are confounded with class: two contain only positive-labelled examples and one only negative-labelled examples. The two validation components comprise one mixed group and one positive-only group. The training data therefore cannot separate class from all source-specific effects, and recording-level counts should not be treated as independent biological replicates. Labels are weak borer annotations, not confirmed red palm weevil or tree health. Eighty-nine near-duplicate pairs remain unreviewed, and source lineage remains unresolved. These limitations prevent a defensible final-holdout sign-off and materially limit generalization.

## Reproduction

- Existing full campaign: `outputs/real-training-cuda/full/treevibes-984e68c4e99668c8/`.
- AST model: `MIT/ast-finetuned-audioset-10-10-0.4593`, revision `f826b80d28226b62986cc218e5cec390b1096902`.
- Wav2Vec2 model: `facebook/wav2vec2-base-960h`, revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`.
- Experiments, model files, cached embeddings, validation results, input checks and tests: `outputs/model-improvement/` and `outputs/model-improvement-wav2vec/`.
- The saved notebook and run artifacts contain the original CNN architecture and weights; the blend specification does not by itself constitute a complete deployable inference package.
- `exploratory_blends.csv` records tested weights; `best_blend_exploratory.json` records the selected combination; `best_blend_validation_predictions.csv` records the matched per-record scores.

Before any final evaluation, independently review recording ancestry and the 89 diagnostic pairs, update grouping if required, freeze a candidate and threshold, and follow the existing one-time holdout gate. Most importantly, collect independently labelled recordings from multiple trees, devices and locations with both classes represented across groups; then evaluate on new independent groups. Measure hardware resource use separately if an embedded model is required.
