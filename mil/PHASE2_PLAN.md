# Phase 2 plan: device-robust MIL models

Written 11 November 2026, before any phase-2 result existed. Development data only; the sealed groups (folders 10, 15, 29+30, 32) stay sealed.

## Why this phase

Every MIL model so far fails the same way: it flags healthy trees that share a recorder or site with infested trees (folders 8 and 24), and per-window specificity is mostly 5-45%. Healthy folder 25, recorded at the same time and site as folder 24 on another recorder, is scored near 0. That points to the device and its coupling to the trunk, not the trees. So the most useful next step is to remove device colouring from the input, not to add capacity.

## What the literature suggests

| Method | Evidence | How it is implemented here |
| --- | --- | --- |
| Frequency-wise input normalisation | Device information sits mainly in per-frequency statistics. Relaxed instance frequency-wise normalisation won DCASE2021 Task 1A (Kim et al. 2022). Frequency centring of the input raised unseen-device accuracy by up to 18.2 points in AST models (Primus & Widmer 2023). | `front="fn"`: subtract each mel band's mean over the clip, then per-clip z-score (`mil/front.py`). |
| Freq-MixStyle | A standard DCASE device-generalisation method. It mixes per-frequency statistics between clips (Truchan et al. 2024; Morocutti et al. 2023). | `mixstyle=True`: p = 0.4, Beta(0.3, 0.3), on the input spectrogram, training only (`mil/model.py`). |
| Device impulse response augmentation | Performs about as well as Freq-MixStyle and complements it, giving the best unseen-device accuracy in 5 of 6 setups (Morocutti et al. 2023). No measured TreeVibes device responses exist, so a random smooth EQ is used as a stand-in. | `randeq=True`: 6-knot random EQ curve within ±6 dB per clip (`mil/data.py`). |
| PCEN instead of log | Adaptive gain control suppresses stationary background and reduced overfitting across recording conditions in bioacoustic sensor networks (Lostanlen et al. 2019). | `front="pcen"`: librosa PCEN, 0.4 s time constant, then per-clip z-score. |

There is no published work on augmentation or generalisation to unseen trees for borer vibrations. The TreeVibes papers report random 10-fold accuracies of 95-97%, which are not comparable with grouped evaluation.

Sources:
- Kim et al. 2022, arXiv 2206.12513
- Primus & Widmer 2023, arXiv 2306.11764
- Morocutti et al. 2023, arXiv 2305.07499
- Truchan et al., DCASE2024 technical report
- Lostanlen et al. 2019, arXiv 1905.08352
- Karar et al. 2022, Alexandria Engineering Journal

## Protocol, fixed before results

- **Folds:** tree-safe leave-one-recorder-out (6 folds), because the question is generalisation to an unseen device.
- **Model:** log-mel CNN encoder, gated attention and linear classifier, so only the front end or augmentation changes.
- **K:** K = 12 first (cheaper; earlier K = 12 augmentations helped more). The best one or two settings are then repeated at K = 24.
- **Seeds:** 3 seeds (42, 43, 44) for every run, including `P-base`.
- **Training windows:** taken from the XS campaign. If XS-cnn-K12 is not better than S-skip-K12 or B0-K12, phase 2 uses the default budget (11,520 bags per fold) and skip windows only if S-skip-style windows helped.
- **Metrics reported for every run:**
  - unweighted per-window ROC-AUC, balanced accuracy, recall and specificity on stride-K, skip and stride-1 windows
  - mean within-recorder AUC (recorders 2538, 7260 and 8391, the only ones with both classes)
  - tree-weighted metrics, for continuity

## Decision rule, fixed before results

A setting counts as better than `P-base` only if all of these hold:

1. Its 3-seed mean unweighted stride-K window AUC exceeds P-base's by more than twice the larger of the two runs' seed SDs.
2. Its mean within-recorder AUC is not lower than P-base's.
3. At least 2 of the 3 seeds beat the P-base mean.

Combinations are run only from settings that pass, plus the fixed combinations listed in `mil/experiments.py`, which were set before results. About 10 settings are compared on 5 healthy trees, so any winner is still optimistic. A clear confirmation needs new healthy recordings from the same recorders as infested trees.

## Queue

1. **Phase 2a (K = 12, 3 seeds each):** P-base, P-fn, P-pcen, P-mix, P-randeq, P-fn-mix, P-mix-randeq, P-pcen-mix.
2. **Phase 2b:** the passing settings at K = 24, 3 seeds, plus frozen AST (`R-AST-lin`, already 5 seeds) as the pretrained reference.
3. **Phase 2c, if anything passes:** the passing front end combined with skip windows and a larger training budget.
