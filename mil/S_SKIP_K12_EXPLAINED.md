# S-skip-K12: what it is, how it was trained, and how its numbers were calculated

Facts are taken from `outputs/mil/experiments/S-skip-K12/seed42/` (config, logs and predictions) and from the code in `mil/`. All results are cross-validated on development trees only.

## 1. One-sentence summary

A small CNN turns each 20 s vibration recording into a 128-number fingerprint. An attention layer weighs 12 recordings from the same tree, taken within 48 hours, and combines them. A linear layer then outputs the probability that the tree is infested.

This is multiple-instance learning (MIL). The label belongs to the tree, not to each recording, because a larva is not audible in every clip.

## 2. Data

| Item | Value |
| --- | --- |
| Source | TreeVibes field recordings: 20 s, 8 kHz, one recording about every 5 minutes |
| Development set | 2,477 recordings, 29 folders, 21 tree groups (5 healthy, 16 infested) |
| Tree groups | folders merged only on same-tree evidence (under 5 m apart or shared recordings); `mil/groups.json` |
| Removed | duplicate recordings (PCM hash); folder 35 (a mixed pool); the sealed holdout groups |

## 3. Input features (per recording)

1. Take the first 20 s, zero-padding if shorter.
2. Band-pass filter 100-2500 Hz (4th-order Butterworth, zero-phase `filtfilt`).
3. Mel spectrogram: FFT 512, hop 256 samples (32 ms), 64 mel bands spanning 100-2500 Hz, so 64 x 626.
4. Convert to dB: `10 log10(power)`.
5. Per-recording z-score: subtract the mean and divide by the standard deviation over the whole 64 x 626 map. This removes overall loudness.
6. Training only: a random circular time shift of each recording.

## 4. Architecture (257,666 parameters)

**Clip encoder** (240,960 parameters, shared by all 12 recordings). Four blocks, each Conv 3x3 (padding 1), BatchNorm, ReLU and MaxPool 2x2:

| Block | Channels | Output size |
| --- | --- | --- |
| 1 | 1 -> 32 | 32 x 32 x 313 |
| 2 | 32 -> 64 | 64 x 16 x 156 |
| 3 | 64 -> 128 | 128 x 8 x 78 |
| 4 | 128 -> 128 | 128 x 4 x 39 |

Then a global max over frequency and time gives h_k (128 numbers) for recording k.

**Gated-attention pooling** (Ilse et al. 2018). V and U are 128 -> 64 linear maps and w is 64 -> 1:

    e_k = w^T [ tanh(V h_k) * sigmoid(U h_k) ]      (* = element-wise)
    a_k = exp(e_k) / sum_j exp(e_j)                 (attention weights over the 12 recordings, summing to 1)
    z   = sum_k a_k h_k                             (one 128-number summary of the window)

**Classifier:** dropout 0.3, then a linear 128 -> 1 layer, then a sigmoid:

    p = sigmoid(W z + b)                            (probability that the window is from an infested tree)

## 5. Training

| Setting | Value |
| --- | --- |
| Window (bag) | K = 12 recordings from one folder (about 1 h if contiguous) |
| Skip windows | pick a random start; keep the recordings within the next 48 h; sample 12 without replacement; sort them by time. Folders shorter than 12 repeat their recordings. |
| Batch | 3 healthy bags and 3 infested bags per step (always balanced) |
| Sampling a bag | a tree group uniformly within its class, then a folder of that tree in proportion to its recording count |
| Loss | binary cross-entropy, averaged over the 6 bags: `L = -mean[ y log p + (1-y) log(1-p) ]` |
| Optimiser | Adam, learning rate 3e-4, weight decay 1e-4, cosine annealing over 30 epochs |
| Length | 30 epochs x 64 steps = 1,920 steps = 11,520 bags per fold |
| Precision / hardware | mixed precision (fp16) on an RTX 2070; 2.3 min per fold, 1.5 GB of GPU memory |
| Model selection | none: the last epoch is kept, so the held-out fold never chooses anything |
| Seed | 42 (one seed only) |

**Training curve** (mean over folds):

| Epoch | Loss | Training-tree AUC |
| ---: | ---: | ---: |
| 1 | 0.33-0.37 | 0.976 |
| 10 | about 0.005 | 0.999 |
| 30 | about 0.0006 | 0.9997 |

The model memorises its training trees.

## 6. Evaluation protocol

- **10-fold cross-validation grouped by tree** (`StratifiedGroupKFold`, shuffle, seed 42; `mil/folds.json`). Every recording is scored by the one model whose training never contained its tree. The ten folds' predictions are then pooled ("out-of-fold") and scored together. With 5 healthy trees, 5 of the 10 folds contain no healthy tree.
- **Window types scored on held-out trees:**
  - **Stride 1:** every run of 12 consecutive recordings, shifting by one (2,187 windows; heavily overlapping).
  - **Stride K:** non-overlapping runs of 12 (199 windows; each recording used once).
  - **Skip:** 20 random time-ordered windows per folder (466 windows).
- **Tree-day:** each stride-1 window is dated by the midpoint of its first and last recording. Per folder and calendar day:

      daily score = mean of that day's window probabilities;  infested if daily score > 0.5

  This gives 48 tree-days: 9 healthy days from 5 trees and 39 infested days. A day can be partial; the median day has 7 windows.

## 7. Formulas

With true positives (TP), false negatives (FN), false positives (FP) and true negatives (TN) counted at threshold 0.5:

    recall (sensitivity)  = TP / (TP + FN)
    specificity           = TN / (TN + FP)
    precision             = TP / (TP + FP)
    F1                    = 2 * precision * recall / (precision + recall)
    balanced accuracy     = (recall + specificity) / 2
    accuracy              = (TP + TN) / (TP + FN + FP + TN)
    ROC-AUC               = P(score of a random infested unit > score of a random healthy unit), ties counting 1/2
                          = sum over infested i, healthy j of w_i w_j [ 1(s_i > s_j) + 0.5 * 1(s_i = s_j) ] / (W_inf * W_healthy)

- **Unweighted:** every unit (window or tree-day) has weight w = 1.
- **Tree-weighted:** a unit's weight is 1 / (number of units from its tree), so each tree contributes 1 in total. Weighted TP, FN, FP and TN are sums of weights, and the same formulas apply.
- **Confidence intervals** (where reported): stratified cluster bootstrap over trees. Healthy and infested trees are resampled separately with replacement, 1,000 draws, giving the 2.5th and 97.5th percentiles.

**The row asked about** (tree-day, unweighted): TP 37, FN 2, FP 6, TN 3.

    recall      = 37 / 39 = 0.949
    specificity = 3 / 9   = 0.333
    precision   = 37 / 43 = 0.860
    F1          = 2 x 0.860 x 0.949 / (0.860 + 0.949) = 0.902
    balanced accuracy = (0.949 + 0.333) / 2 = 0.641
    AUC         = 0.835 over the 9 x 39 = 351 healthy-infested day pairs

**Baseline to quote alongside it:** a "detector" that flags every day gets recall 1.0, precision 39/48 = 0.8125 and F1 0.897. F1 is almost the same, so the useful information is in the AUC and the specificity.

## 8. All headline results (seed 42)

| Level | Weighting | AUC | Bal. acc. | Recall | Specificity | Precision | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tree-day | unweighted | 0.835 | 0.641 | 0.949 | 0.333 | 0.860 | 0.902 |
| Tree-day | tree-weighted | 0.850 | 0.660 | 0.953 | 0.367 | 0.828 | 0.886 |
| Window, stride 1 | tree-weighted (pre-set primary metric) | 0.745 [0.57, 0.90] | 0.635 [0.48, 0.81] | | | | |
| Window, stride K | unweighted | 0.536 [0.41, 0.85] | 0.546 | 0.820 | 0.273 | 0.641 | 0.719 |
| Window, skip | unweighted | 0.730 [0.50, 0.94] | 0.639 | 0.918 | 0.360 | 0.840 | 0.877 |
| Tree-day, folder 8 removed from scoring | unweighted | 0.927 | 0.689 | 0.949 | 0.429 | 0.902 | 0.925 |

The pre-set improvement rule compared it with the baseline B0-K12 (tree-weighted AUC 0.677) and found **no detectable difference**: the gain was inside seed noise and the paired bootstrap interval. That is why skip windows were not adopted for FINAL.

## 9. Where it is right and wrong

**Healthy tree-days** (daily score):

| Folder | Day 1 | Day 2 | Day 3 | Note |
| --- | ---: | ---: | ---: | --- |
| 25 | 0.001 | | | forest |
| 7 | 0.61 | 0.25 | | rain, wind |
| 9 | 0.12 | 0.64 | 0.80 | wind |
| 24 | 0.90 | | | forest; same recorder as 11 infested folders |
| 8 | 1.00 | 0.96 | | rainy day |

- **Missed infested days:** folder 28 on 4 June (0.26, only 7 windows) and folder 33 on 13 June (0.12). Folder 33 is missed at night.
- **Pattern:** false alarms come from rain or wind days (folders 7, 8 and 9) and from the recorder shared with infested trees (folder 24).

## 10. Limitations to state up front

1. **Only 5 healthy trees (9 healthy days):** one day moves specificity by 11 points, and the intervals are wide.
2. **One seed:** earlier seed pairs differed by up to about 0.1 in AUC.
3. **Pooled out-of-fold scoring:** ten different models are scored together, so their score scales are mixed. A threshold that works for one model may not work for another.
4. **Confounds in the data:** every infested tree is an urban palm; the healthy trees are other species, with rain or wind or in a forest. Some recorders recorded only one class. The model can partly learn weather or device instead of larvae.
5. **The tree-day rule was chosen after seeing the results;** only the 0.5 threshold was fixed in advance.
6. **No sealed-holdout result exists for this model.** The holdout tested the frozen-AST models only and is now consumed.

## 11. How to explain it in one slide

> "Each 20 s vibration recording becomes a spectrogram, and a small CNN summarises it. Because a larva is not audible in every recording, the model looks at 12 recordings from the same tree (taken within 2 days) and learns which ones to pay attention to. Tested on trees it never saw in training (10-fold, grouped by tree), averaging its scores per tree and day caught 37 of 39 infested tree-days (95%), but also flagged 6 of 9 healthy days, mostly rainy or windy ones. With only 5 healthy trees, the next step is recording healthy palms at the same sites."
