# Sealed-holdout test: pre-registration

Written 11 November 2026, before any holdout audio, feature or score was loaded. The user explicitly decided to test both frozen-AST models (K = 24 and K = 12), each with and without folder 8 in training. This is a one-time test: after it, the holdout is no longer an independent test set.

## What the holdout is

- **Sealed MIL groups** (`mil/groups.json`): `s07` = folder 10 (healthy), `s09` = folder 15 (infested), `s13` = folders 29 + 30 (infested, same tree), `s14` = folder 32 (infested).
- **This is the project's locked final test split.** Folders 10, 15, 29 and 32 (1,264 recordings) are `split = test` in `outputs/real-training-cuda/.../splits.csv`. Folder 30 is `split = val`, but it is the same tree as folder 29, so it was sealed with it. Testing here consumes that split for every pipeline.
- **Only one healthy group.** Folder 10 is a cut trunk on a balcony, not a standing tree, and has no coordinates or recorder in the workbook. Specificity and AUC therefore rest on one healthy recording stream and cannot show how often healthy palms are flagged.
- **Correction to the question asked of the user:** it said the healthy holdout trees were "two cut trunks". Folder 15 is labelled infested, so only folder 10 is healthy.
- **Duplicates:** holdout recordings that duplicate a development recording by PCM hash are dropped and counted.

## Frozen models

All four models use frozen AST AudioSet embeddings (768-d), gated attention and a linear classifier. Training settings for every model:

- contiguous windows
- 30 epochs x 64 steps x 6 bags
- Adam 3e-4, cosine schedule
- seed 42
- trained on all development trees
- config identical to E-AST-lin / E-AST-lin-K12 except the description text (checked in code)

| ID | Run | K | Folder 8 in training |
| --- | --- | ---: | --- |
| H1 | `FINAL-all` (already trained 10 November) | 24 | yes |
| H2 | `H-AST-K12` | 12 | yes |
| H3 | `H-AST-noF8` | 24 | no |
| H4 | `H-AST-noF8-K12` | 12 | no |

The E-AST-lin-K12 configuration is saved in `mil/configs/E-AST-lin-K12_fold0_config.json`.

## Holdout features

- AST embeddings are computed exactly as for development clips (`build_embeddings.ast_model`: first 20 s at 8 kHz, resampled to 16 kHz, two 10 s halves, patch tokens max-pooled).
- Windows are built per folder in time order, as in development.

## Decision rules (fixed now; no tuning on the holdout)

- **Window level:** probability > 0.5 means infested. Reported on non-overlapping (stride K) and overlapping (stride 1) windows.
- **Tree-day level:** the mean window score (stride 1) per folder and calendar day; > 0.5 means infested.
- **Tree level:** the mean window score per group; > 0.5 means infested.

## Metrics

- AUC, balanced accuracy, recall, specificity, precision, F1 and accuracy.
- Each metric is reported both unweighted (each window or day counts once) and tree-weighted (each group counts equally).
- Mean score and share of windows flagged, per folder.
- With one healthy group, AUC measures whether folder 10's windows rank below the infested windows. It is not a population estimate.
- No confidence interval over trees is meaningful with 1 healthy and 3 infested groups.

## Reporting

- All four models are reported, whatever the outcome. None is presented as "the best" on the strength of this test, because choosing among them after seeing holdout results would make that number optimistic.
- A consumption marker `outputs/mil/HOLDOUT_CONSUMED.json` is written before the first holdout score is computed.

## Known dependence on development data (added before scoring)

- **Folder 32 vs folder 33:** folder 32 (753 recordings) was recorded at the same time as development folder 33, on a second recorder at the same urban site. The data audit could not tell neighbouring trees from one tree with two probes, and they were kept as separate groups because they are not under 5 m apart.
- **Folders 29 and 30 vs folders 28 and 31:** they started within minutes of development folders 28 and 31.
- **Consequence:** the infested holdout groups share site, time and possibly recorder conditions with training trees, so infested recall here is likely optimistic.
- **Size:** the holdout has 1,284 recordings, and none duplicates a development recording by PCM hash.
