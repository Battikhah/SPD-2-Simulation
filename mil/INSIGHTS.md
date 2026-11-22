# Patterns in the MIL results and the plan to improve them

Written 11 November 2026 from all 48 MIL runs (out-of-fold predictions, development trees only), the TreeVibes workbook, and simple acoustic statistics. The sealed holdout was not used.

## 1. Healthy false alarms follow the weather notes, not the model type

Mean window score per healthy folder (non-overlapping windows), averaged over runs:

| Healthy folder | Workbook note | Log-mel CNN, tree folds | Log-mel CNN, recorder held out | Frozen pretrained, tree folds | Frozen pretrained, recorder held out |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 | fig tree, **rainy day, heavy shower**, pruned | 0.99 | 1.00 | 0.98 | 0.93 |
| 9 | apricot tree, **wind**, pruned | 0.84 | 0.99 | 0.72 | 0.45 |
| 7 | orange tree, **rain, wind**, branches and leaves | 0.64 | 0.87 | 0.70 | 0.20 |
| 24 | forest, healthy trees | 0.98 | 0.49 | 0.93 | 0.93 |
| 25 | forest, healthy trees | 0.04 | 0.00 | 0.29 | 0.16 |

- **Folder 8 is scored about 1.0 by every model, at every hour of the day, and even when its recorder is held out.** Rain on a trunk makes short broadband impulses, which is also what larval feeding sounds like (Mankin et al.: trains of brief broadband impulses). Folder 8 is very likely a rain false alarm, not a device artefact.
- **The healthy training trees are weather-noise trees,** and every infested tree is an urban palm with traffic. So "impulsive or noisy" is a strong label shortcut in this dataset.
- **Folder 24 vs 25** (same forest site and time, different recorders) is the device effect seen before. Holding the recorder out fixes folder 24 for the CNN (0.98 to 0.49) but not for the frozen pretrained encoders.

## 2. Strong day-night pattern

- **Infested folder 33** (753 recordings, the largest folder) is missed only at night. Frozen AST scores it 0.03-0.06 between 0 and 8 h and 0.66-0.83 from 8 to 24 h; the log-mel CNNs score it 0.06-0.56 at night and 0.88-1.00 by day.
- **Its share of impulsive frames** (frames at least 6 dB above the clip median) is about 2% at night and 9-10% by day.
- **Healthy folder 9 also peaks in daytime** (wind), so daytime impulses are ambiguous.
- **Per-window AUC is below chance at night for every run checked** (0.31-0.39), against 0.44-0.68 by day.

The literature found here reports larval activity cycles but no day-night rhythm for red palm weevil, so whether the night lull is biological or just less noise is open.

## 3. Deciding per tree-day is far better than deciding per window

These use the same predictions; only the aggregation changes. The unit is one folder and calendar day (40-47 units).

| Run | Per-window AUC (stride 1) | Folder-day AUC, mean score | Folder-day AUC, median of 8-24 h |
| --- | ---: | ---: | ---: |
| E-AST-lin (frozen AST, K = 24) | 0.545 | 0.814 | 0.846 |
| E-AST-lin-K12 | 0.571 | 0.804 | 0.809 |
| S-skip-K12 (log-mel CNN, skip windows) | 0.497 | 0.833 | 0.721 |
| B0-K12 (log-mel CNN) | 0.388 | 0.804 | 0.585 |
| XS-mfcc15-K12 (MFCC 13x15 CNN, 10x skip) | 0.257 | 0.757 | 0.702 |
| R-AST-lin (frozen AST, recorder held out) | 0.445 | 0.756 | 0.764 |
| R-Awin-K12 (CNN + aug, recorder held out) | 0.144 | 0.480 | 0.346 |

- **Per-window AUC is dominated by a few long folders** (33, 9, 8) and their time of day.
- **A day-level mean is closer to what a field alarm would do.** Caveat: only 5 healthy trees give the healthy days, and these aggregation rules were chosen after seeing the data.
- **Frozen AST is the only family that holds up on unseen recorders** (0.76). The CNNs collapse there (0.48).

## 4. Training memorises the training trees

- Training loss reaches about 0.0001 by epoch 10-15 in every CNN run, and training-tree AUC is 1.000.
- 10x more skip windows (XS-mfcc15-K12) did not help: non-overlapping window AUC 0.330, against 0.536 for S-skip-K12 and 0.399 for B0-K12.
- With 15-20 training trees, the models separate them easily; more steps on the same trees add nothing.

## 5. Input type

- **MFCC models are at or below the log-mel CNN.** XS-mfcc15-K12 misses folder 33 almost entirely (0.01 on non-overlapping windows) and folder 5 in the MFCC average (0.04).
- **Frozen AST has the best day-level AUC and the most stable behaviour across recorders.**

## 6. Lab data (one fixed split; see `mil/LAB_RESULTS.md`)

- **Frozen AST:** tree-weighted AUC 0.902 with lab data against 0.779 without. Balanced accuracy at threshold 0.5 is 0.50 for both, so every test window falls on one side of 0.5. The ranking improved, but the threshold did not transfer.
- **The CNN and MFCC runs** are reported in `mil/LAB_RESULTS.md`.

## Recommendations, in order of expected value

1. **Change the decision unit, which needs no training.** Decide per tree per 24 h, not per window, and report day and night separately. On a device this means storing window scores and raising an alarm from a daily statistic. Choose the statistic before looking at new data.
2. **Make weather impossible to use as a shortcut.**
   - Mix rain and wind impulses into infested bags as well as healthy ones. A2-noise, which mixed in healthy-tree noise, was one of the few augmentations that passed.
   - Add a rain-and-wind label to each recording from weather records for the recording dates. TreeVibes gives coordinates and timestamps, so historical weather can be looked up.
   - Report false-alarm rates separately for rain, wind and calm.
3. **Make frozen AST the main encoder, and normalise its input by frequency.** Frequency-wise normalisation was proposed for this exact model (Primus & Widmer 2023). It means recomputing AST embeddings from frequency-centred spectrograms. Phase 2 tests the same idea on the CNN.
4. **Use the structure of larval sounds.** Larval impulses come in trains with gaps under 25 ms; rain drops arrive more randomly. Add burst or train statistics (train rate, spacing, impulses per train) as an interpretable branch or check feature, and test whether it separates folder 8 from infested trees.
5. **Stop scaling training steps.** Use a small validation group inside the training trees for early stopping, or a stronger mixing augmentation. Extra steps only memorise.
6. **Collect the data that removes the shortcuts.** That means:
   - healthy palms at the same urban sites, with the same recorders, recorded 24 h or more
   - logged rain and wind
   - several known-healthy trees per recorder, which per-device calibration also needs

   No model change can separate rain from larvae with 5 healthy trees.
7. **Lab data:** keep it in training only if the CNN and MFCC results agree with AST. Lab clean and lab infested come from a different setting than field trees, so check that the model does not simply learn "lab vs field".

Sources:
- Mankin and colleagues on red palm weevil acoustics: [ARS publication list](https://www.ars.usda.gov/ARSUserFiles/3559/publications/redpalmweevil16jee.pdf), [activity cycles, Jalinas et al. 2017](https://www.ars.usda.gov/ARSUserFiles/3559/publications/Jalinas-rpwcycles-aesa-17-sax061.pdf), [early-instar detection](https://bioone.org/journals/Florida-Entomologist/volume-95/issue-4/024.095.0425/Acoustical-Detection-of-Early-Instar-Rhynchophorus-ferrugineus-Coleoptera--Curculionidae/10.1653/024.095.0425.full)
- [Primus & Widmer 2023](https://arxiv.org/pdf/2306.11764)
- Phase-2 sources: `mil/PHASE2_PLAN.md`
