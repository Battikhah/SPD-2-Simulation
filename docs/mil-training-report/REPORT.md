# From clip classification to tree-level MIL: TreeVibes training findings

Prepared 14 November 2026. Experiments and decisions: 9–11 November 2026; saved-artifact inventory checked 14 November. Companion: [complete model appendix](MODEL_APPENDIX.md).

## Executive summary

Multiple-instance learning (MIL) was pursued because the available labels describe a tree or folder, while audible activity can be intermittent within its recordings. The intended improvement was to let a model combine several recordings and learn which deserve attention, instead of treating every recording from an acoustically positive folder as an independently verified positive example. MIL made the learning unit more appropriate, but it could not repair uncertain biological labels, very few healthy trees, or recording conditions that correlate with class.

The strongest broadly supported model family was **frozen AudioSet AST embeddings with gated-attention MIL and a linear classifier**. In the initial tree-grouped K=24 experiment, E-AST-lin improved tree-weighted stride-1 ROC-AUC from **0.552 to 0.814**. Its balanced accuracy at the fixed 0.5 threshold was only **0.594**, however. In the more demanding recorder-held-out experiment, R-AST-lin reached **0.860 ± 0.041 tree-weighted pooled AUC** and **0.807 ± 0.073 mean within-recorder AUC**, across five seeds. These describe development performance and seed variation, not an independent field test. [1, 2]

The most valuable learning came from failures and changes in evaluation. Models memorised training trees, extra windows did not add independent trees, the apparent ranking changed with weighting and aggregation, and threshold transfer was unstable. A one-time test on four previously sealed groups showed the clearest failure: the AST models trained with healthy folder 8 called **all four groups infested**; the otherwise corresponding models without it called **all four healthy**. Some window-level AUCs were high despite these unhelpful group decisions. The holdout was consumed on 11 November and cannot serve as an independent test again. [3]

The defensible outcome is a model-development study identifying promising representations and major sources of failure. It does not establish reliable infestation detection, species identification, or deployment readiness. A new evaluation needs independently verified, balanced recordings from multiple trees, sites, devices and weather conditions, with thresholds and aggregation fixed before results are inspected.

## 1. Why we moved toward MIL

The preceding clip-level work gave reasons to question apparently excellent classification. A fixed MFCC random forest reached 100% recall and 1.85% false positives under random recording-level folds, but 65.79% recall and 62.56% false positives when whole folders were held out. Nearby recordings from the same source were much easier to recognise than a new recording source. Separately, the early ESP32 models depended strongly on the near-Nyquist 3–4 kHz region and recording characteristics; holding out recorders exposed threshold-transfer problems. These experiments motivated more careful grouping and a decision unit closer to a monitored tree. They are background evidence, not numerically comparable MIL baselines. [4]

The MIL hypothesis was that a healthy or infested-tree monitoring decision should combine repeated recordings. A short clip from an acoustically positive folder can contain little relevant activity, so copying the folder label to every clip creates weak instance supervision. MIL assigns one bag label to several recordings and uses learned attention to combine their representations. It can emphasise informative clips without requiring a verified label for each clip.

This is an inductive assumption, not new ground truth. Positive bags are not guaranteed to contain audible larvae, and the workbook's AUDIO field is a listening judgment rather than complete per-clip biological verification. An attention weight is also not proof that a clip contains feeding sounds. The report uses “healthy” and “infested” as the experiment's assigned class names throughout. [5]

## 2. The audit changed both the dataset and the meaning of a window

Before training, the 9 November audit corrected several assumptions. Recordings were approximately **five minutes apart**, not hourly. Consequently, K=24 usually meant about two hours and K=12 about one hour, with substantial gaps and unequal folder lengths. Neither means a complete day. Many positive folders were too short to supply 24 distinct recordings. [5]

Folder 35 was described as “negative from infested trees,” spanned many collection days, and included a duplicate of a positive recording. It was a pool of quiet clips from infested trees, not an independently healthy tree. The decision was to exclude it entirely. Its earlier high scores therefore could not be treated as demonstrated false alarms on healthy trees.

Duplicate detection also changed grouping. Folders 3 and 4 shared 15 PCM-identical recordings despite different file bytes; folders 27 and 34 shared ten recordings, and a nested folder contained 20 copies. The applied policy merged folders on the user-approved same-tree evidence rule—under 5 m or shared recordings—and removed duplicates. Distance is an operational grouping proxy, not independently verified identity. A broader 25 m grouping was retained as a sensitivity analysis rather than silently replacing the main split.

The user initially retained the sealed groups as whole groups: folders 10, 15, 29+30 and 32. Folder 30 joined folder 29 because of their same-tree evidence, even though earlier pipelines had used it in development. After exclusions and deduplication, MIL development contained **2,477 recordings in 29 folders and 21 groups: 955 recordings from five healthy groups and 1,522 from 16 infested groups**. Ten stratified group folds were fixed; five folds contained no healthy group. We pooled out-of-fold predictions to obtain both-class metrics, instead of interpreting a single-class fold's apparent accuracy as full classification performance. [1, 5]

These decisions reduced leakage, but did not establish independent sites or complete recording ancestry. Simultaneous neighbouring deployments could remain related beyond the 5 m rule. The same recorder could also appear across tree folds.

## 3. The baseline established the problem before architecture search

**B0** used a trainable four-block CNN to turn each 20 s spectrogram into a 128-dimensional vector, then gated attention over K=24 recordings and a linear classifier. It was trained for 30 epochs with balanced six-bag batches, retaining the last epoch. B0-s2 repeated the procedure with seed 43 instead of 42. [1, 6]

| Baseline | Primary tree-weighted stride-1 AUC | Primary balanced accuracy | Unweighted stride-1 AUC |
| --- | ---: | ---: | ---: |
| B0, K24, seed 42 | 0.552 [0.30, 0.84] | 0.611 [0.47, 0.80] | 0.258 |
| B0-s2, K24, seed 43 | 0.619 [0.38, 0.87] | 0.619 [0.48, 0.81] | 0.258 |
| B0-K12, seed 42 | 0.677 | 0.615 | 0.388 |
| B0-s2-K12, seed 43 | 0.688 | 0.629 | 0.428 |

The K24 baseline tree decisions flagged all 16 positive groups but also four of five healthy groups. Training losses fell to very low values, with training-tree AUC approaching 1.0. High training performance accompanied weak held-out discrimination. That is consistent with memorising a small number of training sources rather than learning a robust activity detector. It does not identify the exact acoustic feature responsible.

The second seed mattered: K24 primary AUC moved by about 0.067 without changing the architecture. The experiment therefore used a stricter decision rule than selecting the highest point estimate. A candidate had to improve at least one primary metric—tree-weighted stride-1 AUC or balanced accuracy—with a paired tree-bootstrap 95% interval above zero, exceed the corresponding B0/B0-s2 seed difference, and show no significant deterioration in the other primary metric. Only one baseline seed pair estimated that noise, so this remained a rough safeguard. Reusing the same trees across many comparisons still made selection optimistic. [1, 7]

## 4. Representation, augmentation and pooling experiments

The full initial run list was executed at K24 and K12. It compared frozen PANNs CNN14, BEATs and AST embeddings; linear versus MLP classifiers; individual and combined augmentations; skip-window training; four attention heads; and a transformer before attention. Full per-run values and architectures are in the appendix. The following comparisons all use the original **tree-weighted stride-1** primary metrics. [1, 6]

| Change | K24 AUC / balanced accuracy | K12 AUC / balanced accuracy | What it changed in the decision |
| --- | --- | --- | --- |
| Baseline CNN | 0.552 / 0.611 | 0.677 / 0.615 | Reference |
| Frozen AST + linear | 0.814 / 0.594 | 0.805 / 0.665 | K24 AUC passed; K12 paired intervals included no gain |
| Frozen AST + MLP | 0.797 / 0.507 | 0.825 / 0.573 | Extra head capacity was not a reliable improvement |
| Frozen BEATs + linear | 0.511 / 0.500 | 0.707 / 0.621 | Did not pass |
| Frozen PANNs + linear | 0.663 / 0.560 | 0.657 / 0.676 | Did not pass |
| CNN + MLP | 0.599 / 0.605 | 0.668 / 0.627 | Did not pass |
| A1: synthetic lab pulses | 0.645 / 0.587 | 0.730 / 0.576 | Did not pass |
| A2: healthy-tree noise mixing | 0.610 / 0.595 | 0.680 / 0.660 | K12 balanced accuracy passed |
| A3: hard negatives | 0.590 / 0.612 | 0.655 / 0.606 | Did not pass |
| A4: spectral tilt | 0.649 / 0.632 | 0.691 / 0.653 | K12 balanced accuracy passed |
| A5: SpecAugment | 0.569 / 0.613 | 0.661 / 0.615 | Did not pass |
| A-all: all five augmentations | 0.687 / 0.600 | 0.733 / 0.599 | K24 AUC passed; K12 did not |
| S-skip: sample recordings within 48 h | 0.649 / 0.610 | 0.745 / 0.635 | Did not pass the original rule |
| Four attention heads | 0.600 / 0.626 | 0.650 / 0.631 | Did not pass |
| Transformer + gated attention | 0.629 / 0.622 | 0.681 / 0.634 | Did not pass |

**Why AST was favoured.** Frozen AST brought a representation learned from a much larger acoustic corpus and required only the attention/classification head to fit this small field dataset. Reducing the number of task-specific parameters that must learn from 21 groups is a plausible explanation for its better ranking. This is an interpretation, not an isolated causal test of pretraining: AST also used raw resampled audio with its own front end, whereas the CNN used the 100–2500 Hz filtered spectrogram. We did not fine-tune the AST backbone, and we did not establish what its features identify biologically.

**Why larger heads and attention mechanisms did not solve the problem.** More expressive MLPs, multiple heads and a transformer added capacity to a task with very few independent healthy examples. They could fit training sources without improving transfer. In particular, AST-MLP K12 achieved a higher primary AUC than AST-linear K12 but lower fixed-threshold balanced accuracy. Optimising rank and getting a usable alarm threshold are separate problems.

**Why noise mixing and tilt were retained at K12.** Their balanced-accuracy gains were about 0.045 and 0.038, with paired intervals just above zero, larger than the K12 baseline seed difference. Combining them in A-win-K12 gave **0.680 AUC and 0.666 balanced accuracy**, a roughly 0.050 balanced-accuracy gain. Mixing healthy-tree backgrounds into other bags and changing spectral response plausibly made some shortcuts less useful. It did not prove removal of recorder or weather dependence. A-all's K24 AUC gain did not show that every augmentation helped individually.

**Why lab pulses, hard negatives and masking were not adopted individually.** None met the predeclared rule. Pulse insertion produced synthetic positive bags, whose realism and lab parent ancestry were not established. Hard-negative mining can repeatedly focus on a few atypical healthy sources, and masking can remove a useful signal as well as a shortcut. Those are possible explanations; the experiment only establishes the absence of a convincing gain here. At K24, A1-lab had unweighted stride-1 specificity of only about **0.005**, illustrating that an AUC increase can accompany worse false alarms. [1]

**Why skip training was not initially chosen.** K12 S-skip improved primary AUC from 0.677 to 0.745, but the paired AUC-difference interval was approximately −0.031 to 0.183. The balanced-accuracy interval also crossed zero. “Did not pass” means the available evidence did not distinguish the improvement from uncertainty, not that skip training was conclusively harmful. The later daily and skip-window analyses asked different questions.

## 5. What FINAL meant, and what the second seeds showed

The selection code assembled the final recipes from the passing components. At **K24**, it chose frozen AST, gated attention, a linear classifier, contiguous bags and no augmentation. FINAL seed42 reproduced E-AST-lin's configuration and predictions; it was not a new independent improvement. At **K12**, no pretrained candidate passed, so it chose the CNN with noise mixing and spectral tilt, ordinary gated attention and contiguous bags. FINAL-K12 reproduced A-win-K12. [7]

The K12 AST result was not rejected because its point estimate was poor: its AUC-difference interval versus B0-K12 crossed zero. Selection followed the evidence rule rather than replacing it after seeing a favourable number.

| Selected recipe | Seed 42 AUC / balanced accuracy | Seed 43 AUC / balanced accuracy |
| --- | --- | --- |
| FINAL K24: frozen AST + linear | 0.814 / 0.594 | 0.789 / 0.639 |
| FINAL K12: CNN + noise + tilt | 0.680 / 0.666 | 0.721 / 0.629 |

All values above are tree-weighted stride-1 development metrics. The K12 second seed's balanced accuracy fell back close to the baseline; the seed42 gain was therefore not strong replication across seeds. The AST ranking remained relatively good, while its fixed-threshold behaviour varied. FINAL-all and FINAL-all-K12 were also fit on all development groups. They have no out-of-fold metric of their own; an all-data training fit must not inherit a CV estimate as its measured performance. FINAL-all later became holdout model H1. [1, 3]

## 6. Stricter grouping exposed device and site effects

Folders 24 and 25 were healthy recordings at the same forest site and time on different recorders. Many CNN models scored folder 24 near 1 and folder 25 near 0. The tree-fold baseline scored the healthy folder 8 near 1 as well. These repeated failures motivated holding out the recorder, while removing every linked test-tree group from training. Six recorder folds were used; within-recorder AUC was measurable for only three recorders with both classes. [2]

| Model | Seeds | Pooled TW stride-1 AUC | Pooled TW balanced accuracy | Mean within-recorder AUC | UW stride-K AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| R-B0, CNN K24 | 1 | 0.668 | 0.647 | 0.807 | 0.286 |
| R-Awin-K12, CNN + noise/tilt | 5 | 0.620 ± 0.053 | 0.615 ± 0.038 | 0.781 ± 0.033 | 0.255 ± 0.067 |
| R-AST-lin, frozen AST K24 | 5 | 0.860 ± 0.041 | 0.675 ± 0.037 | 0.807 ± 0.073 | 0.709 ± 0.094 |

Here ± is **sample SD across seed-level metrics**, not a confidence interval or an ensemble result. R-B0 seed43 was stopped before a completed fold; it contributes no scientific result. [2, 8]

The CNN's folder-24 errors improved dramatically when its recorder was absent from training: within-recorder AUC on recorder 2538 rose from B0's 0.189 to R-B0's 0.998, and the healthy-folder mean score fell from about 1.0 to 0.093. That supports recorder dependence as a major explanation for this particular failure. It does not remove the influence of sites, mounting or tree differences.

Frozen AST retained the most promising overall ranking on unseen recorders, but it still flagged essentially all folder-24 windows and most folder-8 windows at 0.5. R-AST-lin's mean within-recorder AUC was **0.888 on 2538, 0.574 on 7260, and 0.960 on 8391**. One recorder was much harder than the others, and each had just one healthy folder. A good three-recorder mean should not hide that limitation.

The 25 m site sensitivity run provided another warning: B0-site25 had primary AUC **0.500** and balanced accuracy **0.484**. Whole-site grouping removed much of the apparent discrimination. A different grouping also changes the learning problem and available training composition, so this does not isolate site as the only cause. It shows that the tree-fold headline was insufficient evidence of transfer to independent sites. [1]

## 7. Longer training, MFCC inputs, and incomplete campaigns

On 11 November the request shifted toward much more skip-window training, both K values, CNN and pretrained representations, and whole-clip 13×15/13×30 MFCC. No new field recordings had arrived. “More data” therefore meant **ten times as many sampled bags from the same 21 groups**: 115,200 instead of 11,520 bags per fold. [9]

Whole-clip MFCC was computed from the cached dB mel via an orthonormal DCT, retaining 13 coefficients and mean-pooling 626 frames into 15 or 30 time chunks. This is not the earlier one-second MFCC representation. A custom CNN was used; no suitable pretrained encoder was actually applied to these MFCC maps.

| Completed K12 run | UW stride-K AUC / balanced accuracy | UW random-skip AUC | UW stride-1 AUC | TW stride-1 AUC |
| --- | --- | ---: | ---: | ---: |
| B0-K12, contiguous, default budget | 0.399 / 0.473 | n/a | 0.388 | 0.677 |
| S-skip-K12, log-mel, default budget | 0.536 / 0.546 | 0.730 | 0.497 | 0.745 |
| XS-mfcc15-K12, tenfold budget | 0.330 / 0.284 | 0.819 | 0.257 | 0.779 |
| XS-mfcc30-K12, tenfold budget | 0.346 / 0.315 | 0.789 | 0.274 | 0.766 |

The MFCC runs scored relatively well on randomly sampled skip bags yet poorly on ordinary contiguous held-out windows. Random skip evaluation spreads observations over a longer interval and changes the distribution and weighting of units; XS also used 100 skip draws per eligible folder instead of S-skip's 20. These numbers are not interchangeable evidence of a superior detector. Temporal compression may discard short impulsive structure, while more steps may strengthen source memorisation. The available comparison changes both input and training budget, so it cannot causally establish that tenfold training itself made performance worse. [9]

Training losses and training-tree AUC nevertheless showed that lack of fit to training sources was not the main problem. Repeatedly sampling the same few trees does not create new biological or environmental variation.

The original XS run also crashed with Windows error 1455 after worker shared-memory mappings accumulated. The loader was changed to restart workers every 1,920 steps, with segment-specific sampling seeds, and the first run was restarted. This was an execution repair, not evidence of an accuracy gain.

**Artifact status matters:** only XS-mfcc15-K12 and XS-mfcc30-K12 have complete production CV results. XS-mfcc15 K24 has **three of ten folds** completed. The remaining XS pretrained/log-mel queue entries have no production run directories. A proposed device-robustness phase implemented frequency centring, PCEN, Freq-MixStyle and random EQ, but **no P-* production checkpoints were found**. PHASE2_RESULTS.md currently contains R-* reference models only. These approaches must be described as planned/implemented, not as completed negative or positive experiments. [8, 10; artifact inventory]

## 8. Two uses of laboratory data gave different lessons

The initial **A1-lab** experiment inserted selected short lab pulses into a few clips of a healthy bag and assigned a synthetic positive label. Its pulse bank contained 2,571 accepted clips from the first 15,000 candidate lab-infested clips after an impulse threshold. It did not convincingly improve the primary metrics. [1]

The later **L-* experiment** instead joined pairs of 10 s lab recordings into 20 s pseudo-clips: **483 clean and 967 infested pseudo-clips**. These were training-only additions, not extra field test trees. Six K12 models compared AST, log-mel CNN and MFCC30 CNN, with and without lab data, using 115,200 skip-training bags. This used **one deliberately selected fixed development split and seed**, not full CV: healthy test folders 8 and 24, and positive test groups from folders 5, 11, 19+23 and 28. The other 15 field groups were training data. [11]

| Encoder | UW stride-K AUC, no lab → lab | UW stride-K balanced accuracy, no lab → lab |
| --- | --- | --- |
| Frozen AST | 0.746 → 0.877 | 0.500 → 0.500 |
| Log-mel CNN | 0.542 → 0.436 | 0.505 → 0.407 |
| MFCC30 CNN | 0.515 → 0.504 | 0.450 → 0.368 |

AST's corresponding TW stride-1 AUC rose from **0.779 to 0.902**, but at the non-overlapping-window 0.5 threshold it still flagged every test window. CNN and MFCC results deteriorated on the stated measures. Lab data therefore offered an AST ranking hypothesis, not a general solution. Frozen representations may accommodate added acoustic diversity differently from encoders trained from scratch; a lab/field shortcut or mismatched positive examples could also hurt transfer. This experiment cannot distinguish those mechanisms, and missing lab parent ancestry prevents counting pseudo-clips as independent biological replicates.

## 9. Aggregation and error analysis changed the interpretation

### Weighting changed the question

Long folders dominated unweighted window metrics, especially folders 8, 9 and 33. A short positive group contributes only a few windows, while one long negative sequence can contribute hundreds. B0's stride-1 AUC was **0.258 unweighted but 0.552 tree-weighted**; E-AST-lin's was **0.545 unweighted but 0.814 tree-weighted**. Neither is an arithmetic error: the first describes a randomly selected saved window; the second gives each assigned tree group equal influence. Neither equals an AUC calculated from one final binary decision per tree.

Pooled OOF AUC also compares scores produced by different fold models. Score-scale differences across folds/recorders can change pooled ranking. Within-recorder AUC compares windows from one held-out recorder scored by the same model, making it useful alongside—not a replacement for—pooled metrics. [1, 2]

### Daily averaging made S-skip-K12 look useful, with important limits

The later tree-day analysis averaged stride-1 probabilities per **folder and calendar day**, dating a window by the midpoint of its first/last timestamps. These are deployment-folder days, not necessarily complete 24 h intervals or unique merged-tree days. For S-skip-K12 this gave 48 units: nine healthy days from five healthy groups and 39 positive days. [12]

It detected **37/39 positive days** and flagged **6/9 healthy days**: recall 0.949, specificity 0.333, precision 0.860, F1 0.902, balanced accuracy 0.641, and AUC about **0.835**. The TW daily AUC was **0.850**, with balanced accuracy 0.660. This is a useful aggregation finding, but the daily rule was selected after examining results, so it is exploratory.

An always-positive rule on those same 48 days has recall 1.0, precision 39/48 = 0.8125 and F1 **0.897**. S-skip's F1 of 0.902 therefore sounds more impressive than the extra discrimination it provides. Its AUC and specificity are more informative than recall or F1 alone. Folder 8's removal from *scoring* raised daily AUC to 0.927, but that deletes a known difficult healthy source after seeing its errors. It is an error analysis, not a fair improved headline result or a retrained model.

The earlier INSIGHTS.md snapshot reports a slightly different S-skip daily AUC (0.833). The more detailed S_SKIP_K12_EXPLAINED.md and the saved prediction reconstruction support the approximately 0.835 figure used here. Definitions and source are kept explicit rather than mixing the two snapshots.

### Weather and time of day remain plausible explanations, not verified causes

The healthy folder-8 workbook note describes a rainy day/heavy shower; it scored high across models, including recorder-held-out models. Healthy folders 7 and 9 also have rain/wind notes. Folder 33, the largest positive folder, often had low scores at night and high scores by day. Exploratory analyses found lower impulse activity at night and poor night-window discrimination. This supports testing sensitivity to weather/background activity and time of day. It does not prove that rain caused each false positive or that the day/night pattern represents larval biology. No experiment independently labelled the relevant acoustic events. [13]

The user had accepted other host species as usable negatives and treated the recordings as acoustically comparable for this work. Those choices define the working scope; they do not demonstrate that site, weather, mounting or device response is irrelevant. We cannot identify one unique reason for every model failure from these data.

### Persistent memory did not remove persistent false positives

MEM-alarm reused existing scores with `state = max(decay × state, probability)`. Both a per-recording and a real-time per-day decay were examined because the five-minute cadence changed the meaning of the originally proposed rule. It was not a newly trained model. A leaky maximum can retain isolated false alarms as well as true events; for B0, end-state decisions had recall 1.0 but specificity 0.2. Thus the blanket claim that every memory setting flagged every tree is too broad. The useful conclusion is that persistence did not solve the specificity problem. [1, 14]

## 10. The one-time holdout showed unstable thresholds

On 11 November the user explicitly authorised consuming the sealed groups. Four frozen AST models and their rules were written down before loading holdout audio/features/scores. H1 reused the K24 FINAL-all model; H2 was K12 with folder 8 included; H3/H4 were K24/K12 without folder 8. All used the same frozen AST/gated-linear recipe and seed42, trained on all applicable development groups. [3, 15]

The holdout contained **1,284 recordings from four groups: one healthy group (folder 10, a cut trunk on a balcony) and three positive groups (15, 29+30, 32)**. It was not a pristine independent-site test: folder 32 shared time/site conditions with development folder 33, and 29+30 were linked in time/site to development neighbours. No duplicate holdout PCM was found against development in the saved evaluation record. One healthy source cannot establish population specificity.

| Frozen model | K | Folder 8 in training | UW stride-K AUC | TW stride-K AUC | Mean-score group decisions at 0.5 |
| --- | ---: | --- | ---: | ---: | --- |
| H1 = FINAL-all | 24 | yes | 0.768 | 0.915 | all four infested |
| H2 = H-AST-K12 | 12 | yes | 0.719 | 0.894 | all four infested |
| H3 = H-AST-noF8 | 24 | no | 0.498 | 0.782 | all four healthy |
| H4 = H-AST-noF8-K12 | 12 | no | 0.613 | 0.821 | all four healthy |

The statements in the last column refer to **group means**. H2 did produce both positive and negative window decisions; it did not flag literally every window. Folder 10 had the lowest mean score for H1 and H2 but still exceeded 0.5 (0.805 and 0.598). Removing folder 8 shifted every group's mean below 0.5. Thus ordering some groups correctly did not provide a stable alarm scale.

This is strong evidence of sensitivity to the healthy training distribution. It supports investigating per-device or deployment calibration, but calibration was **not validated by this holdout experiment**. Selecting a new threshold from these holdout errors would be additional development. All four models must be reported; the high H1 AUC does not justify selecting it as a new independent-test winner.

The consumption marker was written at **2026-11-11 21:51:48** as recorded. Neither this MIL holdout nor the overlapping locked final split of the earlier pipeline is an independent test any longer. This report uses existing saved results and does not rerun holdout evaluation. A later NF8-AST-lin-K12 retraining attempt stopped with six of ten folds complete; its partial checkpoints do not supply a new CV conclusion. [artifact inventory]

## 11. Which models are best, and exactly how were they trained?

There is no single comparable leaderboard across tree folds, recorder folds, fixed lab splits, window weighting and daily aggregation. The appropriate choices depend on the claim:

| Purpose | Most informative candidate | Supported result | Qualification |
| --- | --- | --- | --- |
| Strongest repeated unseen-recorder ranking | R-AST-lin, K24 | Five-seed TW S1 AUC 0.860 ± 0.041; within-recorder AUC 0.807 ± 0.073 | Three mixed-class recorders; poor 0.5 false-alarm control |
| Original rule-selected K24 model | E-AST-lin / FINAL | TW S1 AUC 0.814 [0.64, 0.97] | Same configuration/seed repeated; selected CV estimate |
| Original rule-selected K12 CNN | A-win-K12 / FINAL-K12 | TW S1 balanced accuracy 0.666, seed42 | Fell to 0.629 in seed43; unseen-recorder results weak |
| Most instructive low-budget aggregation result | S-skip-K12 | 37/39 positive days detected; 6/9 healthy days flagged | Post-hoc daily analysis; no holdout result for this CNN |
| Highest lab-assisted AST development point estimate cited here | L-AST-lab-K12 | TW S1 AUC 0.902 on one fixed split | Not comparable to full CV; UW SK balanced accuracy 0.5 |

### Exact common recipe

Training sampled three healthy and three positive bags each step. Within each class it chose a tree group uniformly, then a folder proportional to its recording count. Default runs used **30 epochs × 64 steps × 6 bags = 11,520 sampled bags/fold**, Adam at **3×10⁻⁴**, weight decay **10⁻⁴**, a cosine learning-rate schedule, binary cross-entropy on logits, and mixed precision on the RTX 2070. The **last epoch** was retained. Watching held-out curves did not implement best-epoch selection, although human review of development performance still influenced later experiment design. [6]

Training bags were contiguous unless skip was enabled. Contiguous starts were retried up to 20 times to prefer spans ≤36 h; that is not an absolute span guarantee because the last attempted window can be accepted. Skip bags sampled without replacement inside a candidate 48 h span and were sorted by time, with fallback to contiguous selection. A short training folder repeated clips to fill K; evaluation used its available clips once. Bags were formed within a folder even when folds grouped multiple folders as one tree.

### Frozen AST models

AST used `MIT/ast-finetuned-audioset-10-10-0.4593`. The raw first 20 s at 8 kHz, padded if needed, was resampled to 16 kHz and split into two 10 s halves. Each half went through the AST feature extractor/model. CLS and distillation tokens were excluded; patch tokens were max-pooled and the halves were max-pooled into one **768-dimensional clip embedding**. Backbone weights stayed frozen. Each fold standardised embeddings with **training-fold** mean/SD.

The trainable MIL head used two 768→64 linear maps and a 64→1 gated attention score, followed by a weighted mean of the 768-d embeddings, dropout 0.3 and a 768→1 output. It had **99,266 trainable parameters**; this excludes the large frozen AST backbone and feature-extraction cost. Reporting that head size as the whole AST model size would be misleading.

E-AST-lin/FINAL used K24, tree-grouped ten-fold CV, seed42, contiguous bags and no augmentation. FINAL-s2 used seed43. R-AST-lin used the same architecture, K and budget with **tree-safe leave-one-recorder-out**, six folds and seeds **42–46**. A held-out recorder's folders were test sequences, and any tree group appearing on that recorder was wholly removed from training. These are five independent training seeds of a procedure, not a deployed five-model ensemble. H1–H4 used all applicable development groups as preregistered; H3/H4 removed folder 8. [2, 6, 15]

### The two informative K12 CNN recipes

The CNN front end took the first 20 s at 8 kHz, applied a fourth-order zero-phase Butterworth bandpass at **100–2500 Hz**, and produced **64×626** dB mel features with FFT512 and hop256. Each clip was z-scored. Four 3×3 conv/BatchNorm/ReLU/2×2-max-pool blocks with channels **32,64,128,128** and global frequency/time max pooling produced a 128-d clip vector. Gated attention used hidden size64; dropout0.3 and a linear output completed the **257,666-parameter** model.

S-skip-K12 used this CNN, 12 recordings selected within 48 h, a random circular shift of each training clip, the default budget and seed42. A-win-K12/FINAL-K12 used contiguous K12 bags plus **noise mixing** (another healthy training source, probability0.3/clip, relative gain −12 to0 dB in the power domain) and **spectral tilt** (random linear mel-band ramp within ±3 dB), then per-clip z-score. It used the same budget, with seeds42 and43 for the final repeats. The two recipes differ in sampling and augmentation; their daily and primary-window results should not be spliced into one model claim. [6, 12]

No MIL model here was benchmarked on the ESP32. The earlier small clip-level int8 model's measured on-board timing/memory is a different pipeline and cannot be transferred to AST or the 257k-parameter MIL CNN.

## 12. How the reported results were calculated

**Out-of-fold predictions.** Every development window was scored by a model that excluded its assigned tree group under tree folds, or its recorder and linked tree groups under recorder folds. Fold predictions were concatenated. These are pooled OOF metrics, not an unweighted mean of fold AUCs. L-* uses a single fixed split instead. A threshold of **probability >0.5** was used for MIL classification; this differs from earlier ESP32 threshold policies.

**Evaluation windows.** Stride1 uses consecutive overlapping windows; strideK uses disjoint full K-windows where possible, with one short window for a folder shorter than K. Long-folder incomplete tails are omitted in strideK. Skip evaluation samples ordered recordings within 48 h, using a fixed evaluation RNG; short folders give one available-clip bag. Overlapping and repeated random bags are not independent trees or independent biological observations.

**Weights.** Unweighted results assign each window/day weight1. Tree-weighted results give each unit weight `1 / number of units from its tree group` for the metric being calculated, making each group's total weight1. Class weighting, where present in the source tables, gives each class equal total weight; it is a third definition. Tree-weighted window results are not one-score-per-tree results. Actual tree verdicts use mean score, any positive window, or >25% positive-window share, as separately labelled in RESULTS.md.

**Metrics.** With weighted or ordinary TP, FN, FP, TN as appropriate:

```text
Recall       = TP / (TP + FN)
Specificity  = TN / (TN + FP)
FPR          = 1 − specificity
Precision    = TP / (TP + FP)
F1           = 2 × TP / (2 × TP + FP + FN), when defined
Balanced acc = (recall + specificity) / 2
Accuracy     = (TP + TN) / (TP + FN + FP + TN)
ROC-AUC      = weighted fraction of positive/negative score pairs
               correctly ordered, counting a tie as one half
```

Undefined class metrics stay undefined rather than becoming zero. The implementation's `pr_auc` is stepwise **average precision**, not a trapezoidal PR area. Precision depends on the dataset prevalence. Its optional `precision_at_10pct` is an algebraic projection to assumed 10% prevalence using the measured recall/FPR, not observed field precision. [16]

**Uncertainty and comparisons.** Saved 95% intervals use 1,000 stratified cluster bootstrap draws of healthy and positive **tree groups**, separately sampled with replacement; repeated copies retain their within-tree units. Paired model comparisons use the same tree draws. This respects within-tree clustering more than resampling windows, but does not fully account for shared site/device dependence, adaptive selection or training randomness. With only five healthy development groups, intervals are broad and unstable. Five-seed mean±SD values are calculated from each seed's already-pooled metrics; no prediction ensembling is implied.

**Within-recorder AUC.** For each seed, stride1 predictions were restricted to recorder2538,7260 or8391. Windows were tree-weighted within that recorder, then AUCs were averaged across those three recorders. Finally, those seed-level means were summarised across seeds. Thus the reported 0.807 is neither an AUC over all recorder windows pooled together nor an AUC from a mean prediction ensemble. [2]

**Daily results.** The S-skip daily example grouped by folder/calendar day at the window midpoint, averaged probabilities, then applied0.5. Its TP37/FN2/FP6/TN3 give recall37/39 and specificity3/9. The nine healthy days come from only five groups and do not constitute nine independent healthy trees.

**Holdout.** Only the four preregistered AST models were evaluated. Window, daily and group rules were fixed in the preregistration, and all four outcomes were retained. No meaningful population confidence interval can be supported by one healthy holdout group. The results show failure of threshold stability, despite some favourable rankings.

## 13. The findings that taught us the most

1. **The independent unit determines the apparent success.** Random files, tree-grouped windows, held-out recorders and held-out sites answer different questions. Near-perfect clip results disappeared under stricter splits. MIL grouping improved the protocol without creating a large independent dataset.
2. **A better representation helped more than extra task-specific capacity.** Frozen AST gave the most consistent ranking across recorders; larger heads, multihead pooling and a transformer did not convincingly solve the problem. That makes frozen AST a useful reference for future data, rather than a validated detector.
3. **Ranking, recall and useful alarms must be separated.** High AUC can coexist with all-positive decisions; high recall/F1 can be achieved by mostly flagging everything in a positive-heavy sample. Specificity, prevalence, weighting and decision unit must accompany every headline.
4. **Thresholds depended on the healthy training distribution.** Including/excluding folder8 flipped all holdout group decisions. That was more informative than selecting the highest AUC: the score scale was not stable enough for a transferable alarm threshold.
5. **More bags were not more trees.** The XS campaign expanded sampling tenfold while retaining five healthy sources. Good skip-bag results and excellent training fit did not imply reliable contiguous-window performance.
6. **Aggregation is promising but must be fixed prospectively.** Daily averaging gave more informative decisions than short windows, yet six of nine healthy days still alarmed. The next study should preregister the aggregation interval, missing-data rules and calibration procedure.
7. **The remaining uncertainty needs new evidence.** Balanced healthy/positive recordings on the same devices/sites, verified status, recorded weather and mounting, multiple healthy trees per device, and new reserved sites would directly challenge the observed shortcuts. Further selection on this dataset cannot provide that test.

## Sources, reconciliation and reproducibility

The visible chat and supplied AGENTS.md handoff were used to establish intent and decisions. Saved experiment configurations, metrics, predictions and scripts were used to check what actually ran. A full verbatim Claude conversation was not available here; this is an evidence-backed reconstruction, not a claim to have reviewed every historical chat message. No new model training, audio decoding, feature extraction or holdout scoring was performed for this report.

1. [MIL results](../../mil/RESULTS.md), plus saved `summary.csv`, `oof_predictions.csv` and `paired_vs_*.csv` under `outputs/mil/experiments/`. Historical text saying the holdout is sealed is superseded by source3 and its consumption marker.
2. [Recorder results](../../mil/RECORDER_RESULTS.md) and [recorder calculation](../../mil/recorder_report.py).
3. [Consumed-holdout results](../../mil/HOLDOUT_RESULTS.md) and `outputs/mil/HOLDOUT_CONSUMED.json`.
4. [Acoustic audit and preceding clip-level results](../ACOUSTIC_DATA_AUDIT.md), [ESP32 development report](../ESP32_DEV_DAY_REPORT.md), and [prior handoff summary](../../khalid_chat_with_cluade_summary.md). Earlier “do not open the test” descriptions are historical; the later consumption remains binding.
5. [MIL audit, especially applied decisions in section8](../../mil/DATA_REPORT.md), [groups](../../mil/groups.json), and [folds](../../mil/folds.json).
6. [Experiment definitions](../../mil/experiments.py), [trainer](../../mil/train.py), [sampling/augmentation](../../mil/data.py), [architectures](../../mil/model.py), [CNN features](../../mil/build_features.py), [frozen embeddings](../../mil/build_embeddings.py), and actual fold configurations snapshotted in [model_evidence.json](model_evidence.json).
7. [Selection rule and composition of FINAL](../../mil/decide.py), plus saved `_plans/*.json` included in the evidence snapshot.
8. [Phase2 reference results](../../mil/PHASE2_RESULTS.md) and `outputs/mil/phase2_per_seed.csv`.
9. [Skip-window results](../../mil/SKIP_RESULTS.md) and [MFCC construction](../../mil/build_mfcc.py).
10. [Phase2 plan](../../mil/PHASE2_PLAN.md). Literature proposals listed there are reported as historical motivation; this report makes no new literature-comparison claim.
11. [Lab experiment results](../../mil/LAB_RESULTS.md) and [pseudo-clip construction](../../mil/build_lab_features.py).
12. [Detailed S-skip-K12 explanation](../../mil/S_SKIP_K12_EXPLAINED.md), checked against its saved predictions.
13. [Exploratory insights](../../mil/INSIGHTS.md). Its weather/biology interpretations are hypotheses, not established event-level ground truth.
14. [Memory alarm calculation](../../mil/mem_alarm.py) and saved MEM-alarm rows in source1. The initial handoff's broad “every tree” description is qualified by the actual B0 specificity0.2 result.
15. [Holdout preregistration](../../mil/HOLDOUT_PREREG.md).
16. [Metric implementation](../../mil/metrics.py).

The appendix enumerates all 60 saved non-smoke MIL run names, 69 run/seed attempts and 526 checkpoints, rather than treating a planned queue as completed work. It includes complete, partial and all-data fits and preserves their exact fold configurations in the companion evidence snapshot. [build_inventory.py](build_inventory.py) regenerates that appendix using saved metadata only. Original artifacts remain in ignored `outputs/`; these report files alone do not transfer the weights or recordings.
