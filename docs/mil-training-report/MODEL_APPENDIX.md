# Appendix: complete saved MIL model register

Prepared 14 November 2026.

This register covers **every non-smoke run directory currently saved under `outputs/mil/experiments/`**, including each seed and partial attempt. Earlier clip-level campaigns are background to the MIL study, not enumerated as MIL models. A run name, a seed, a fold checkpoint and a distinct architecture are different counts.

Inventory: **60 run names, 69 run/seed attempts, 526 saved checkpoints**. Status counts: 55 complete CV; 5 trained on all development groups; 6 complete fixed split; 2 partial; no complete CV result; 1 started/stopped; no saved completed fold.

## How to use this appendix

The tables below report saved **pooled out-of-fold** values for each seed separately. They are not an ensemble. All threshold metrics use strict probability > 0.5. `TW S1` means stride-1 windows weighted so each tree group contributes total weight 1; `UW SK` means non-overlapping windows with one vote per window. These answer different questions. Brackets are saved 95% tree-cluster bootstrap intervals, not seed SDs. n/a means unavailable.

Complete per-fold configuration snapshots, split membership, every saved summary metric and metadata hashes are in [model_evidence.json](model_evidence.json). Original artifact paths below are relative to the repository. `outputs/` is generally ignored by Git; copying this appendix does not copy weights or predictions.

## Common training and architecture definitions

- **CNN:** first 20 s of 8 kHz audio; 100–2500 Hz Butterworth bandpass; 64×626 log-mel (FFT 512, hop 256); per-clip z-score. Four 3×3 conv/BN/ReLU blocks, channels 32/64/128/128, 2×2 pooling after each block, global max to 128 dimensions. 257,666 trainable parameters including MIL head.
- **MFCC:** DCT-II of the same 64-band dB mel, first 13 coefficients; 626 frames mean-pooled into 15 or 30 chunks; training-fold coefficient normalisation. Four conv/BN/ReLU blocks with only two intermediate pools, global max to 128 dimensions.
- **Frozen encoders:** raw first 20 s, resampled to each encoder's rate; AST takes two 10 s halves at 16 kHz and max-pools patch tokens (excluding CLS/distillation tokens) and halves. BEATs max-pools tokens at 16 kHz; PANNs CNN14 uses 32 kHz, final-conv frequency mean/time max. The frozen backbones are **not** counted in saved head parameter counts. Embedding normalisation uses training-fold mean/SD. These do not use the CNN's 100–2500 Hz filtered front end.
- **Gated pooling:** softmax attention of `w[tanh(Vh) * sigmoid(Uh)]`, attention hidden size 64; weighted sum of clip vectors. Linear classifier is dropout 0.3 then one output; MLP is pooled dimension→128→1 with ReLU/dropout. `heads4` concatenates four gated summaries. Transformer adds one 4-head, feedforward-256 layer and learned positions before gated pooling.
- **Default budget:** Adam 3e-4, weight decay 1e-4, cosine schedule, 30 epochs × 64 steps × 6 bags = 11,520 bags/fold; 3 bags/class each step, class-balanced BCE, tree group uniform within class, folder proportional to recording count. Last epoch retained; no best-epoch selection. Seed 42 unless listed otherwise; GPU AMP on RTX 2070.
- **K** counts recordings, not seconds: K=24 roughly two hours and K=12 roughly one hour at five-minute cadence. Default training is contiguous with a 36 h span preference/retry limit; skip samples K clips within 48 h, with contiguous fallback. Short training folders repeat recordings to fill K; short evaluation folders are scored once with their available recordings.
- **A1-lab:** insert selected lab pulse power into 1–3 clips of a healthy bag and assign a synthetic positive label. This is distinct from L-* lab pseudo-clip training. A2 mixes healthy training-tree noise (p=0.3/clip, −12 to 0 dB); A3 mines healthy training windows scoring >0.2; A4 applies ±3 dB spectral tilt; A5 masks one 1–8-band and one 1–60-frame span.
- **XS-*:** 30×640×6 = 115,200 bags/fold, skip training, 100 skip draws/evaluation folder. L-* uses this budget on one fixed development split, optionally adding 483 clean and 967 infested lab pseudo-clips. No additional independent field trees are created by either procedure.

## Every saved run and seed

| Run | Seed | K | Split | Completed / required folds | Status | Trainable params |
| --- | ---: | ---: | --- | --- | --- | ---: |
| A-all | 42 | 24 | tree | 10/10 | complete CV | 257666 |
| A-all-K12 | 42 | 12 | tree | 10/10 | complete CV | 257666 |
| A-win-K12 | 42 | 12 | tree | 10/10 | complete CV | 257666 |
| A1-lab | 42 | 24 | tree | 10/10 | complete CV | 257666 |
| A1-lab-K12 | 42 | 12 | tree | 10/10 | complete CV | 257666 |
| A2-noise | 42 | 24 | tree | 10/10 | complete CV | 257666 |
| A2-noise-K12 | 42 | 12 | tree | 10/10 | complete CV | 257666 |
| A3-hardneg | 42 | 24 | tree | 10/10 | complete CV | 257666 |
| A3-hardneg-K12 | 42 | 12 | tree | 10/10 | complete CV | 257666 |
| A4-tilt | 42 | 24 | tree | 10/10 | complete CV | 257666 |
| A4-tilt-K12 | 42 | 12 | tree | 10/10 | complete CV | 257666 |
| A5-specaug | 42 | 24 | tree | 10/10 | complete CV | 257666 |
| A5-specaug-K12 | 42 | 12 | tree | 10/10 | complete CV | 257666 |
| B0 | 42 | 24 | tree | 10/10 | complete CV | 257666 |
| B0-K12 | 42 | 12 | tree | 10/10 | complete CV | 257666 |
| B0-s2 | 43 | 24 | tree | 10/10 | complete CV | 257666 |
| B0-s2-K12 | 43 | 12 | tree | 10/10 | complete CV | 257666 |
| B0-site25 | 42 | 24 | site25 | 10/10 | complete CV | 257666 |
| E-AST-lin | 42 | 24 | tree | 10/10 | complete CV | 99266 |
| E-AST-lin-K12 | 42 | 12 | tree | 10/10 | complete CV | 99266 |
| E-AST-mlp | 42 | 24 | tree | 10/10 | complete CV | 197058 |
| E-AST-mlp-K12 | 42 | 12 | tree | 10/10 | complete CV | 197058 |
| E-BEATs-lin | 42 | 24 | tree | 10/10 | complete CV | 99266 |
| E-BEATs-lin-K12 | 42 | 12 | tree | 10/10 | complete CV | 99266 |
| E-BEATs-mlp | 42 | 24 | tree | 10/10 | complete CV | 197058 |
| E-BEATs-mlp-K12 | 42 | 12 | tree | 10/10 | complete CV | 197058 |
| E-PANNs-lin | 42 | 24 | tree | 10/10 | complete CV | 264386 |
| E-PANNs-lin-K12 | 42 | 12 | tree | 10/10 | complete CV | 264386 |
| E-PANNs-mlp | 42 | 24 | tree | 10/10 | complete CV | 524738 |
| E-PANNs-mlp-K12 | 42 | 12 | tree | 10/10 | complete CV | 524738 |
| FINAL | 42 | 24 | tree | 10/10 | complete CV | 99266 |
| FINAL-all | 42 | 24 | tree | 0/1 (all-data fit) | trained on all development groups | 99266 |
| FINAL-all-K12 | 42 | 12 | tree | 0/1 (all-data fit) | trained on all development groups | 257666 |
| FINAL-K12 | 42 | 12 | tree | 10/10 | complete CV | 257666 |
| FINAL-s2 | 43 | 24 | tree | 10/10 | complete CV | 99266 |
| FINAL-s2-K12 | 43 | 12 | tree | 10/10 | complete CV | 257666 |
| H-AST-K12 | 42 | 12 | tree | 0/1 (all-data fit) | trained on all development groups | 99266 |
| H-AST-noF8 | 42 | 24 | tree | 0/1 (all-data fit) | trained on all development groups | 99266 |
| H-AST-noF8-K12 | 42 | 12 | tree | 0/1 (all-data fit) | trained on all development groups | 99266 |
| H-mlp | 42 | 24 | tree | 10/10 | complete CV | 274178 |
| H-mlp-K12 | 42 | 12 | tree | 10/10 | complete CV | 274178 |
| L-AST-lab-K12 | 42 | 12 | single | 1/1 | complete fixed split | 99266 |
| L-AST-nolab-K12 | 42 | 12 | single | 1/1 | complete fixed split | 99266 |
| L-cnn-lab-K12 | 42 | 12 | single | 1/1 | complete fixed split | 257666 |
| L-cnn-nolab-K12 | 42 | 12 | single | 1/1 | complete fixed split | 257666 |
| L-mfcc30-lab-K12 | 42 | 12 | single | 1/1 | complete fixed split | 257666 |
| L-mfcc30-nolab-K12 | 42 | 12 | single | 1/1 | complete fixed split | 257666 |
| M-heads4 | 42 | 24 | tree | 10/10 | complete CV | 307781 |
| M-heads4-K12 | 42 | 12 | tree | 10/10 | complete CV | 307781 |
| NF8-AST-lin-K12 | 42 | 12 | tree | 6/10 | partial; no complete CV result | 99266 |
| R-AST-lin | 42 | 24 | recorder | 6/6 | complete CV | 99266 |
| R-AST-lin | 43 | 24 | recorder | 6/6 | complete CV | 99266 |
| R-AST-lin | 44 | 24 | recorder | 6/6 | complete CV | 99266 |
| R-AST-lin | 45 | 24 | recorder | 6/6 | complete CV | 99266 |
| R-AST-lin | 46 | 24 | recorder | 6/6 | complete CV | 99266 |
| R-Awin-K12 | 42 | 12 | recorder | 6/6 | complete CV | 257666 |
| R-Awin-K12 | 43 | 12 | recorder | 6/6 | complete CV | 257666 |
| R-Awin-K12 | 44 | 12 | recorder | 6/6 | complete CV | 257666 |
| R-Awin-K12 | 45 | 12 | recorder | 6/6 | complete CV | 257666 |
| R-Awin-K12 | 46 | 12 | recorder | 6/6 | complete CV | 257666 |
| R-B0 | 42 | 24 | recorder | 6/6 | complete CV | 257666 |
| R-B0 | 43 | n/a | n/a | 0/6 | started/stopped; no saved completed fold | n/a |
| S-skip | 42 | 24 | tree | 10/10 | complete CV | 257666 |
| S-skip-K12 | 42 | 12 | tree | 10/10 | complete CV | 257666 |
| T-transformer | 42 | 24 | tree | 10/10 | complete CV | 393218 |
| T-transformer-K12 | 42 | 12 | tree | 10/10 | complete CV | 391682 |
| XS-mfcc15 | 42 | 24 | tree | 3/10 | partial; no complete CV result | 257666 |
| XS-mfcc15-K12 | 42 | 12 | tree | 10/10 | complete CV | 257666 |
| XS-mfcc30-K12 | 42 | 12 | tree | 10/10 | complete CV | 257666 |

## Comparable results for every completed run/seed

L-* rows are a single fixed split and must not be ranked as though they were 10-fold CV. R-* rows hold out recorders, with linked tree groups removed from training.

| Run / seed | TW S1 AUC [95% CI] | TW S1 balanced accuracy | TW S1 recall | TW S1 specificity | UW SK AUC | UW SK balanced accuracy | UW SK recall | UW SK specificity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A-all / 42 | 0.687 [0.463, 0.909] | 0.600 | 0.963 | 0.236 | 0.499 | 0.481 | 0.882 | 0.079 |
| A-all-K12 / 42 | 0.733 [0.532, 0.915] | 0.599 | 0.888 | 0.311 | 0.511 | 0.427 | 0.738 | 0.117 |
| A-win-K12 / 42 | 0.680 [0.452, 0.906] | 0.666 | 0.965 | 0.367 | 0.471 | 0.574 | 0.902 | 0.247 |
| A1-lab / 42 | 0.645 [0.366, 0.910] | 0.587 | 0.973 | 0.200 | 0.414 | 0.418 | 0.809 | 0.026 |
| A1-lab-K12 / 42 | 0.730 [0.523, 0.906] | 0.576 | 0.919 | 0.232 | 0.467 | 0.417 | 0.795 | 0.039 |
| A2-noise / 42 | 0.610 [0.380, 0.840] | 0.595 | 0.944 | 0.245 | 0.326 | 0.429 | 0.779 | 0.079 |
| A2-noise-K12 / 42 | 0.680 [0.465, 0.901] | 0.660 | 0.980 | 0.340 | 0.485 | 0.559 | 0.910 | 0.208 |
| A3-hardneg / 42 | 0.590 [0.326, 0.859] | 0.612 | 0.927 | 0.296 | 0.291 | 0.392 | 0.706 | 0.079 |
| A3-hardneg-K12 / 42 | 0.655 [0.453, 0.863] | 0.606 | 0.917 | 0.295 | 0.366 | 0.464 | 0.811 | 0.117 |
| A4-tilt / 42 | 0.649 [0.433, 0.875] | 0.632 | 0.934 | 0.331 | 0.411 | 0.451 | 0.691 | 0.211 |
| A4-tilt-K12 / 42 | 0.691 [0.513, 0.878] | 0.653 | 0.971 | 0.335 | 0.488 | 0.534 | 0.861 | 0.208 |
| A5-specaug / 42 | 0.569 [0.282, 0.876] | 0.613 | 0.922 | 0.303 | 0.261 | 0.378 | 0.676 | 0.079 |
| A5-specaug-K12 / 42 | 0.662 [0.455, 0.870] | 0.615 | 0.902 | 0.328 | 0.419 | 0.463 | 0.770 | 0.156 |
| B0 / 42 | 0.552 [0.299, 0.837] | 0.611 | 0.938 | 0.284 | 0.352 | 0.470 | 0.809 | 0.132 |
| B0-K12 / 42 | 0.677 [0.473, 0.885] | 0.615 | 0.914 | 0.317 | 0.399 | 0.473 | 0.803 | 0.143 |
| B0-s2 / 43 | 0.619 [0.378, 0.868] | 0.619 | 0.966 | 0.273 | 0.343 | 0.451 | 0.824 | 0.079 |
| B0-s2-K12 / 43 | 0.688 [0.480, 0.884] | 0.629 | 0.985 | 0.273 | 0.434 | 0.482 | 0.885 | 0.078 |
| B0-site25 / 42 | 0.500 [0.249, 0.773] | 0.484 | 0.713 | 0.255 | 0.241 | 0.363 | 0.647 | 0.079 |
| E-AST-lin / 42 | 0.814 [0.636, 0.968] | 0.594 | 0.935 | 0.253 | 0.655 | 0.599 | 0.750 | 0.447 |
| E-AST-lin-K12 / 42 | 0.805 [0.591, 0.970] | 0.665 | 0.950 | 0.379 | 0.603 | 0.521 | 0.705 | 0.338 |
| E-AST-mlp / 42 | 0.797 [0.618, 0.942] | 0.507 | 0.970 | 0.045 | 0.659 | 0.479 | 0.853 | 0.105 |
| E-AST-mlp-K12 / 42 | 0.825 [0.666, 0.943] | 0.573 | 0.966 | 0.181 | 0.631 | 0.488 | 0.795 | 0.182 |
| E-BEATs-lin / 42 | 0.511 [0.211, 0.842] | 0.500 | 0.928 | 0.071 | 0.577 | 0.489 | 0.794 | 0.184 |
| E-BEATs-lin-K12 / 42 | 0.707 [0.502, 0.906] | 0.621 | 0.954 | 0.288 | 0.553 | 0.477 | 0.746 | 0.208 |
| E-BEATs-mlp / 42 | 0.622 [0.369, 0.873] | 0.609 | 0.977 | 0.241 | 0.557 | 0.500 | 0.868 | 0.132 |
| E-BEATs-mlp-K12 / 42 | 0.682 [0.448, 0.909] | 0.628 | 0.971 | 0.284 | 0.508 | 0.461 | 0.754 | 0.169 |
| E-PANNs-lin / 42 | 0.663 [0.445, 0.887] | 0.560 | 0.881 | 0.240 | 0.461 | 0.437 | 0.794 | 0.079 |
| E-PANNs-lin-K12 / 42 | 0.657 [0.415, 0.898] | 0.676 | 0.842 | 0.510 | 0.464 | 0.453 | 0.672 | 0.234 |
| E-PANNs-mlp / 42 | 0.580 [0.307, 0.850] | 0.602 | 0.903 | 0.301 | 0.490 | 0.482 | 0.779 | 0.184 |
| E-PANNs-mlp-K12 / 42 | 0.646 [0.437, 0.856] | 0.575 | 0.849 | 0.302 | 0.489 | 0.460 | 0.713 | 0.208 |
| FINAL / 42 | 0.814 [0.636, 0.968] | 0.594 | 0.935 | 0.253 | 0.655 | 0.599 | 0.750 | 0.447 |
| FINAL-K12 / 42 | 0.680 [0.452, 0.906] | 0.666 | 0.965 | 0.367 | 0.471 | 0.574 | 0.902 | 0.247 |
| FINAL-s2 / 43 | 0.789 [0.597, 0.935] | 0.639 | 0.933 | 0.345 | 0.615 | 0.599 | 0.750 | 0.447 |
| FINAL-s2-K12 / 43 | 0.721 [0.487, 0.919] | 0.629 | 0.985 | 0.274 | 0.496 | 0.475 | 0.885 | 0.065 |
| H-mlp / 42 | 0.599 [0.375, 0.833] | 0.605 | 0.954 | 0.256 | 0.342 | 0.455 | 0.779 | 0.132 |
| H-mlp-K12 / 42 | 0.668 [0.467, 0.856] | 0.627 | 0.958 | 0.296 | 0.489 | 0.515 | 0.836 | 0.195 |
| L-AST-lab-K12 / 42 | 0.902 [0.754, 1.000] | 0.502 | 1.000 | 0.004 | 0.877 | 0.500 | 1.000 | 0.000 |
| L-AST-nolab-K12 / 42 | 0.779 [0.419, 1.000] | 0.500 | 1.000 | 0.000 | 0.746 | 0.500 | 1.000 | 0.000 |
| L-cnn-lab-K12 / 42 | 0.554 [0.174, 0.883] | 0.399 | 0.732 | 0.065 | 0.436 | 0.407 | 0.815 | 0.000 |
| L-cnn-nolab-K12 / 42 | 0.684 [0.300, 0.970] | 0.619 | 0.989 | 0.250 | 0.542 | 0.505 | 0.963 | 0.048 |
| L-mfcc30-lab-K12 / 42 | 0.536 [0.194, 0.876] | 0.335 | 0.593 | 0.077 | 0.504 | 0.368 | 0.593 | 0.143 |
| L-mfcc30-nolab-K12 / 42 | 0.600 [0.262, 0.899] | 0.578 | 0.525 | 0.630 | 0.515 | 0.450 | 0.519 | 0.381 |
| M-heads4 / 42 | 0.600 [0.352, 0.865] | 0.626 | 0.957 | 0.294 | 0.366 | 0.469 | 0.779 | 0.158 |
| M-heads4-K12 / 42 | 0.650 [0.417, 0.897] | 0.631 | 0.943 | 0.319 | 0.356 | 0.489 | 0.836 | 0.143 |
| R-AST-lin / 42 | 0.792 [0.580, 0.972] | 0.666 | 0.960 | 0.371 | 0.559 | 0.445 | 0.706 | 0.184 |
| R-AST-lin / 43 | 0.856 [0.704, 0.974] | 0.673 | 0.900 | 0.446 | 0.694 | 0.610 | 0.721 | 0.500 |
| R-AST-lin / 44 | 0.893 [0.775, 0.988] | 0.727 | 0.907 | 0.547 | 0.723 | 0.664 | 0.750 | 0.579 |
| R-AST-lin / 45 | 0.880 [0.766, 0.978] | 0.686 | 0.863 | 0.509 | 0.759 | 0.632 | 0.765 | 0.500 |
| R-AST-lin / 46 | 0.881 [0.757, 0.976] | 0.624 | 0.918 | 0.330 | 0.810 | 0.729 | 0.853 | 0.605 |
| R-Awin-K12 / 42 | 0.541 [0.238, 0.851] | 0.657 | 0.901 | 0.412 | 0.171 | 0.426 | 0.787 | 0.065 |
| R-Awin-K12 / 43 | 0.663 [0.424, 0.893] | 0.608 | 0.904 | 0.312 | 0.319 | 0.436 | 0.820 | 0.052 |
| R-Awin-K12 / 44 | 0.632 [0.370, 0.889] | 0.554 | 0.900 | 0.208 | 0.239 | 0.421 | 0.803 | 0.039 |
| R-Awin-K12 / 45 | 0.593 [0.288, 0.895] | 0.631 | 0.924 | 0.337 | 0.220 | 0.456 | 0.861 | 0.052 |
| R-Awin-K12 / 46 | 0.669 [0.412, 0.902] | 0.623 | 0.926 | 0.320 | 0.327 | 0.477 | 0.902 | 0.052 |
| R-B0 / 42 | 0.668 [0.427, 0.896] | 0.647 | 0.853 | 0.441 | 0.286 | 0.382 | 0.632 | 0.132 |
| S-skip / 42 | 0.649 [0.443, 0.864] | 0.610 | 0.941 | 0.279 | 0.447 | 0.505 | 0.853 | 0.158 |
| S-skip-K12 / 42 | 0.745 [0.570, 0.905] | 0.635 | 0.909 | 0.361 | 0.536 | 0.546 | 0.820 | 0.273 |
| T-transformer / 42 | 0.629 [0.343, 0.899] | 0.622 | 0.964 | 0.280 | 0.411 | 0.455 | 0.779 | 0.132 |
| T-transformer-K12 / 42 | 0.681 [0.411, 0.915] | 0.635 | 0.968 | 0.301 | 0.409 | 0.457 | 0.770 | 0.143 |
| XS-mfcc15-K12 / 42 | 0.779 [0.580, 0.938] | 0.638 | 0.804 | 0.473 | 0.330 | 0.284 | 0.361 | 0.208 |
| XS-mfcc30-K12 / 42 | 0.766 [0.567, 0.938] | 0.658 | 0.882 | 0.434 | 0.346 | 0.315 | 0.566 | 0.065 |

## Additional non-overlapping-window results

All quantities below are unweighted, stride K, threshold >0.5. Fractional values would indicate weights; these counts should be integers. Recall/specificity/AUC are in the preceding table.

| Run / seed | TP | FN | FP | TN | Precision | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A-all / 42 | 60.000 | 8.000 | 35.000 | 3.000 | 0.632 | 0.736 | 0.594 |
| A-all-K12 / 42 | 90.000 | 32.000 | 68.000 | 9.000 | 0.570 | 0.643 | 0.497 |
| A-win-K12 / 42 | 110.000 | 12.000 | 58.000 | 19.000 | 0.655 | 0.759 | 0.648 |
| A1-lab / 42 | 55.000 | 13.000 | 37.000 | 1.000 | 0.598 | 0.688 | 0.528 |
| A1-lab-K12 / 42 | 97.000 | 25.000 | 74.000 | 3.000 | 0.567 | 0.662 | 0.503 |
| A2-noise / 42 | 53.000 | 15.000 | 35.000 | 3.000 | 0.602 | 0.679 | 0.528 |
| A2-noise-K12 / 42 | 111.000 | 11.000 | 61.000 | 16.000 | 0.645 | 0.755 | 0.638 |
| A3-hardneg / 42 | 48.000 | 20.000 | 35.000 | 3.000 | 0.578 | 0.636 | 0.481 |
| A3-hardneg-K12 / 42 | 99.000 | 23.000 | 68.000 | 9.000 | 0.593 | 0.685 | 0.543 |
| A4-tilt / 42 | 47.000 | 21.000 | 30.000 | 8.000 | 0.610 | 0.648 | 0.519 |
| A4-tilt-K12 / 42 | 105.000 | 17.000 | 61.000 | 16.000 | 0.633 | 0.729 | 0.608 |
| A5-specaug / 42 | 46.000 | 22.000 | 35.000 | 3.000 | 0.568 | 0.617 | 0.462 |
| A5-specaug-K12 / 42 | 94.000 | 28.000 | 65.000 | 12.000 | 0.591 | 0.669 | 0.533 |
| B0 / 42 | 55.000 | 13.000 | 33.000 | 5.000 | 0.625 | 0.705 | 0.566 |
| B0-K12 / 42 | 98.000 | 24.000 | 66.000 | 11.000 | 0.598 | 0.685 | 0.548 |
| B0-s2 / 43 | 56.000 | 12.000 | 35.000 | 3.000 | 0.615 | 0.704 | 0.557 |
| B0-s2-K12 / 43 | 108.000 | 14.000 | 71.000 | 6.000 | 0.603 | 0.718 | 0.573 |
| B0-site25 / 42 | 44.000 | 24.000 | 35.000 | 3.000 | 0.557 | 0.599 | 0.443 |
| E-AST-lin / 42 | 51.000 | 17.000 | 21.000 | 17.000 | 0.708 | 0.729 | 0.642 |
| E-AST-lin-K12 / 42 | 86.000 | 36.000 | 51.000 | 26.000 | 0.628 | 0.664 | 0.563 |
| E-AST-mlp / 42 | 58.000 | 10.000 | 34.000 | 4.000 | 0.630 | 0.725 | 0.585 |
| E-AST-mlp-K12 / 42 | 97.000 | 25.000 | 63.000 | 14.000 | 0.606 | 0.688 | 0.558 |
| E-BEATs-lin / 42 | 54.000 | 14.000 | 31.000 | 7.000 | 0.635 | 0.706 | 0.575 |
| E-BEATs-lin-K12 / 42 | 91.000 | 31.000 | 61.000 | 16.000 | 0.599 | 0.664 | 0.538 |
| E-BEATs-mlp / 42 | 59.000 | 9.000 | 33.000 | 5.000 | 0.641 | 0.738 | 0.604 |
| E-BEATs-mlp-K12 / 42 | 92.000 | 30.000 | 64.000 | 13.000 | 0.590 | 0.662 | 0.528 |
| E-PANNs-lin / 42 | 54.000 | 14.000 | 35.000 | 3.000 | 0.607 | 0.688 | 0.538 |
| E-PANNs-lin-K12 / 42 | 82.000 | 40.000 | 59.000 | 18.000 | 0.582 | 0.624 | 0.503 |
| E-PANNs-mlp / 42 | 53.000 | 15.000 | 31.000 | 7.000 | 0.631 | 0.697 | 0.566 |
| E-PANNs-mlp-K12 / 42 | 87.000 | 35.000 | 61.000 | 16.000 | 0.588 | 0.644 | 0.518 |
| FINAL / 42 | 51.000 | 17.000 | 21.000 | 17.000 | 0.708 | 0.729 | 0.642 |
| FINAL-K12 / 42 | 110.000 | 12.000 | 58.000 | 19.000 | 0.655 | 0.759 | 0.648 |
| FINAL-s2 / 43 | 51.000 | 17.000 | 21.000 | 17.000 | 0.708 | 0.729 | 0.642 |
| FINAL-s2-K12 / 43 | 108.000 | 14.000 | 72.000 | 5.000 | 0.600 | 0.715 | 0.568 |
| H-mlp / 42 | 53.000 | 15.000 | 33.000 | 5.000 | 0.616 | 0.688 | 0.547 |
| H-mlp-K12 / 42 | 102.000 | 20.000 | 62.000 | 15.000 | 0.622 | 0.713 | 0.588 |
| L-AST-lab-K12 / 42 | 27.000 | 0.000 | 21.000 | 0.000 | 0.562 | 0.720 | 0.562 |
| L-AST-nolab-K12 / 42 | 27.000 | 0.000 | 21.000 | 0.000 | 0.562 | 0.720 | 0.562 |
| L-cnn-lab-K12 / 42 | 22.000 | 5.000 | 21.000 | 0.000 | 0.512 | 0.629 | 0.458 |
| L-cnn-nolab-K12 / 42 | 26.000 | 1.000 | 20.000 | 1.000 | 0.565 | 0.712 | 0.562 |
| L-mfcc30-lab-K12 / 42 | 16.000 | 11.000 | 18.000 | 3.000 | 0.471 | 0.525 | 0.396 |
| L-mfcc30-nolab-K12 / 42 | 14.000 | 13.000 | 13.000 | 8.000 | 0.519 | 0.519 | 0.458 |
| M-heads4 / 42 | 53.000 | 15.000 | 32.000 | 6.000 | 0.624 | 0.693 | 0.557 |
| M-heads4-K12 / 42 | 102.000 | 20.000 | 66.000 | 11.000 | 0.607 | 0.703 | 0.568 |
| R-AST-lin / 42 | 48.000 | 20.000 | 31.000 | 7.000 | 0.608 | 0.653 | 0.519 |
| R-AST-lin / 43 | 49.000 | 19.000 | 19.000 | 19.000 | 0.721 | 0.721 | 0.642 |
| R-AST-lin / 44 | 51.000 | 17.000 | 16.000 | 22.000 | 0.761 | 0.756 | 0.689 |
| R-AST-lin / 45 | 52.000 | 16.000 | 19.000 | 19.000 | 0.732 | 0.748 | 0.670 |
| R-AST-lin / 46 | 58.000 | 10.000 | 15.000 | 23.000 | 0.795 | 0.823 | 0.764 |
| R-Awin-K12 / 42 | 96.000 | 26.000 | 72.000 | 5.000 | 0.571 | 0.662 | 0.508 |
| R-Awin-K12 / 43 | 100.000 | 22.000 | 73.000 | 4.000 | 0.578 | 0.678 | 0.523 |
| R-Awin-K12 / 44 | 98.000 | 24.000 | 74.000 | 3.000 | 0.570 | 0.667 | 0.508 |
| R-Awin-K12 / 45 | 105.000 | 17.000 | 73.000 | 4.000 | 0.590 | 0.700 | 0.548 |
| R-Awin-K12 / 46 | 110.000 | 12.000 | 73.000 | 4.000 | 0.601 | 0.721 | 0.573 |
| R-B0 / 42 | 43.000 | 25.000 | 33.000 | 5.000 | 0.566 | 0.597 | 0.453 |
| S-skip / 42 | 58.000 | 10.000 | 32.000 | 6.000 | 0.644 | 0.734 | 0.604 |
| S-skip-K12 / 42 | 100.000 | 22.000 | 56.000 | 21.000 | 0.641 | 0.719 | 0.608 |
| T-transformer / 42 | 53.000 | 15.000 | 33.000 | 5.000 | 0.616 | 0.688 | 0.547 |
| T-transformer-K12 / 42 | 94.000 | 28.000 | 66.000 | 11.000 | 0.588 | 0.667 | 0.528 |
| XS-mfcc15-K12 / 42 | 44.000 | 78.000 | 61.000 | 16.000 | 0.419 | 0.388 | 0.302 |
| XS-mfcc30-K12 / 42 | 69.000 | 53.000 | 72.000 | 5.000 | 0.489 | 0.525 | 0.372 |

## Exact architecture and training differences by run

### A-all

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v2: MIL v1 + all five augmentations (A1-A5).

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=noise, lab, tilt, specaug, hardneg; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/A-all/seed42`.

### A-all-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v2: MIL v1 + all five augmentations (A1-A5) [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=noise, lab, tilt, specaug, hardneg; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/A-all-K12/seed42`.

### A-win-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 + the augmentations that passed: noise mixing, EQ tilt [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=noise, tilt; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/A-win-K12/seed42`.

### A1-lab

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 + synthetic infested windows (lab pulses mixed into healthy clips).

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=lab; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/A1-lab/seed42`.

### A1-lab-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 + synthetic infested windows (lab pulses mixed into healthy clips) [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=lab; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/A1-lab-K12/seed42`.

### A2-noise

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 + noise mixing from other healthy training trees.

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=noise; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/A2-noise/seed42`.

### A2-noise-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 + noise mixing from other healthy training trees [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=noise; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/A2-noise-K12/seed42`.

### A3-hardneg

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 + hard-negative mining (healthy windows scoring > 0.2).

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=hardneg; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/A3-hardneg/seed42`.

### A3-hardneg-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 + hard-negative mining (healthy windows scoring > 0.2) [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=hardneg; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/A3-hardneg-K12/seed42`.

### A4-tilt

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 + EQ tilt augmentation (+/-3 dB).

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=tilt; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/A4-tilt/seed42`.

### A4-tilt-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 + EQ tilt augmentation (+/-3 dB) [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=tilt; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/A4-tilt-K12/seed42`.

### A5-specaug

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 + SpecAugment (1 band span, 1 time span).

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=specaug; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/A5-specaug/seed42`.

### A5-specaug-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 + SpecAugment (1 band span, 1 time span) [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=specaug; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/A5-specaug-K12/seed42`.

### B0

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 baseline.

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/B0/seed42`.

### B0-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 baseline: trainable 4-block CNN clip encoder (log-mel 64x626) + gated attention + linear classifier [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/B0-K12/seed42`.

### B0-s2

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 baseline; second seed.

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 43: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/B0-s2/seed43`.

### B0-s2-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 baseline, second seed (seed-noise check) [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 43: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/B0-s2-K12/seed43`.

### B0-site25

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 baseline with folds by 25 m site groups (sensitivity check).

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=site25; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/B0-site25/seed42`.

### E-AST-lin

**Architecture:** frozen AudioSet AST, 768-d; gated pooling; linear classifier.

**Approach:** frozen AST AudioSet embeddings (768-d) + gated attention + linear classifier.

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/E-AST-lin/seed42`.

### E-AST-lin-K12

**Architecture:** frozen AudioSet AST, 768-d; gated pooling; linear classifier.

**Approach:** frozen AST AudioSet embeddings (768-d) + gated attention + linear classifier [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/E-AST-lin-K12/seed42`.

### E-AST-mlp

**Architecture:** frozen AudioSet AST, 768-d; gated pooling; mlp classifier.

**Approach:** frozen AST AudioSet embeddings (768-d) + gated attention + MLP classifier.

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/E-AST-mlp/seed42`.

### E-AST-mlp-K12

**Architecture:** frozen AudioSet AST, 768-d; gated pooling; mlp classifier.

**Approach:** frozen AST AudioSet embeddings (768-d) + gated attention + MLP classifier [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/E-AST-mlp-K12/seed42`.

### E-BEATs-lin

**Architecture:** frozen BEATs iter3+ AS2M, 768-d; gated pooling; linear classifier.

**Approach:** frozen BEATs iter3+ AS2M embeddings (768-d) + gated attention + linear classifier.

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/E-BEATs-lin/seed42`.

### E-BEATs-lin-K12

**Architecture:** frozen BEATs iter3+ AS2M, 768-d; gated pooling; linear classifier.

**Approach:** frozen BEATs iter3+ AS2M embeddings (768-d) + gated attention + linear classifier [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/E-BEATs-lin-K12/seed42`.

### E-BEATs-mlp

**Architecture:** frozen BEATs iter3+ AS2M, 768-d; gated pooling; mlp classifier.

**Approach:** frozen BEATs iter3+ AS2M embeddings (768-d) + gated attention + MLP classifier.

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/E-BEATs-mlp/seed42`.

### E-BEATs-mlp-K12

**Architecture:** frozen BEATs iter3+ AS2M, 768-d; gated pooling; mlp classifier.

**Approach:** frozen BEATs iter3+ AS2M embeddings (768-d) + gated attention + MLP classifier [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/E-BEATs-mlp-K12/seed42`.

### E-PANNs-lin

**Architecture:** frozen PANNs CNN14, 2048-d; gated pooling; linear classifier.

**Approach:** frozen PANNs CNN14 embeddings (AudioSet, 2048-d) + gated attention + linear classifier.

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/E-PANNs-lin/seed42`.

### E-PANNs-lin-K12

**Architecture:** frozen PANNs CNN14, 2048-d; gated pooling; linear classifier.

**Approach:** frozen PANNs CNN14 embeddings (AudioSet, 2048-d) + gated attention + linear classifier [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/E-PANNs-lin-K12/seed42`.

### E-PANNs-mlp

**Architecture:** frozen PANNs CNN14, 2048-d; gated pooling; mlp classifier.

**Approach:** frozen PANNs CNN14 embeddings (AudioSet, 2048-d) + gated attention + MLP classifier.

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/E-PANNs-mlp/seed42`.

### E-PANNs-mlp-K12

**Architecture:** frozen PANNs CNN14, 2048-d; gated pooling; mlp classifier.

**Approach:** frozen PANNs CNN14 embeddings (AudioSet, 2048-d) + gated attention + MLP classifier [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/E-PANNs-mlp-K12/seed42`.

### FINAL

**Architecture:** frozen AudioSet AST, 768-d; gated pooling; linear classifier.

**Approach:** FINAL: encoder: frozen AST (best passing encoder run E-AST-lin); classifier: linear (-mlp did not beat -lin); augmentations: none (pretrained encoder; O6 not run); training windows: contiguous (S-skip did not pass); pooling: gated (T-transformer did not pass, M-heads4 did not pass).

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/FINAL/seed42`.

This is a repeated configuration/seed, not an independent algorithmic gain: identical to E-AST-lin.

### FINAL-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** FINAL: encoder: trainable CNN (no pretrained-encoder run passed the improvement rule against B0); classifier: linear (H-mlp did not pass); augmentations: noise mixing, EQ tilt; training windows: contiguous (S-skip did not pass); pooling: gated (T-transformer did not pass, M-heads4 did not pass) [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=noise, tilt; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/FINAL-K12/seed42`.

This is a repeated configuration/seed, not an independent algorithmic gain: identical to A-win-K12.

### FINAL-all

**Architecture:** frozen AudioSet AST, 768-d; gated pooling; linear classifier.

**Approach:** FINAL: encoder: frozen AST (best passing encoder run E-AST-lin); classifier: linear (-mlp did not beat -lin); augmentations: none (pretrained encoder; O6 not run); training windows: contiguous (S-skip did not pass); pooling: gated (T-transformer did not pass, M-heads4 did not pass) (trained on all cross-validation trees; not evaluated).

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: trained on all development groups; 1 checkpoint(s); artifact directory `outputs/mil/experiments/FINAL-all/seed42`.

### FINAL-all-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** FINAL: encoder: trainable CNN (no pretrained-encoder run passed the improvement rule against B0); classifier: linear (H-mlp did not pass); augmentations: noise mixing, EQ tilt; training windows: contiguous (S-skip did not pass); pooling: gated (T-transformer did not pass, M-heads4 did not pass) (trained on all cross-validation trees; not evaluated) [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=noise, tilt; excluded folders=[].

- Seed 42: trained on all development groups; 1 checkpoint(s); artifact directory `outputs/mil/experiments/FINAL-all-K12/seed42`.

### FINAL-s2

**Architecture:** frozen AudioSet AST, 768-d; gated pooling; linear classifier.

**Approach:** FINAL: encoder: frozen AST (best passing encoder run E-AST-lin); classifier: linear (-mlp did not beat -lin); augmentations: none (pretrained encoder; O6 not run); training windows: contiguous (S-skip did not pass); pooling: gated (T-transformer did not pass, M-heads4 did not pass) (seed 43).

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 43: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/FINAL-s2/seed43`.

### FINAL-s2-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** FINAL: encoder: trainable CNN (no pretrained-encoder run passed the improvement rule against B0); classifier: linear (H-mlp did not pass); augmentations: noise mixing, EQ tilt; training windows: contiguous (S-skip did not pass); pooling: gated (T-transformer did not pass, M-heads4 did not pass) (seed 43) [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=noise, tilt; excluded folders=[].

- Seed 43: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/FINAL-s2-K12/seed43`.

### H-AST-K12

**Architecture:** frozen AudioSet AST, 768-d; gated pooling; linear classifier.

**Approach:** frozen AST AudioSet embeddings (768-d) + gated attention + linear classifier; trained on all development trees [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: trained on all development groups; 1 checkpoint(s); artifact directory `outputs/mil/experiments/H-AST-K12/seed42`.

### H-AST-noF8

**Architecture:** frozen AudioSet AST, 768-d; gated pooling; linear classifier.

**Approach:** frozen AST AudioSet embeddings (768-d) + gated attention + linear classifier; trained on all development trees except folder 8.

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=['folder_8'].

- Seed 42: trained on all development groups; 1 checkpoint(s); artifact directory `outputs/mil/experiments/H-AST-noF8/seed42`.

### H-AST-noF8-K12

**Architecture:** frozen AudioSet AST, 768-d; gated pooling; linear classifier.

**Approach:** frozen AST AudioSet embeddings (768-d) + gated attention + linear classifier; trained on all development trees except folder 8 [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=['folder_8'].

- Seed 42: trained on all development groups; 1 checkpoint(s); artifact directory `outputs/mil/experiments/H-AST-noF8-K12/seed42`.

### H-mlp

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; mlp classifier.

**Approach:** trainable 4-block CNN clip encoder (log-mel 64x626) + gated attention + MLP classifier (128-128-1).

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/H-mlp/seed42`.

### H-mlp-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; mlp classifier.

**Approach:** trainable 4-block CNN clip encoder (log-mel 64x626) + gated attention + MLP classifier (128-128-1) [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/H-mlp-K12/seed42`.

### L-AST-lab-K12

**Architecture:** frozen AudioSet AST, 768-d; gated pooling; linear classifier.

**Approach:** frozen AST AudioSet embeddings (768-d) + gated attention + linear; skip-window training, 10x training windows, single split + lab recordings (483 clean / 967 infested 20 s pseudo-clips) in training [K=12].

**Recipe:** K=12; 30 epochs; 640 steps/epoch; 6 bags/step; grouping=single; skip=True; active options=lab_data; excluded folders=[].

- Seed 42: complete fixed split; 1 checkpoint(s); artifact directory `outputs/mil/experiments/L-AST-lab-K12/seed42`.

### L-AST-nolab-K12

**Architecture:** frozen AudioSet AST, 768-d; gated pooling; linear classifier.

**Approach:** frozen AST AudioSet embeddings (768-d) + gated attention + linear; skip-window training, 10x training windows, single split, no lab data [K=12].

**Recipe:** K=12; 30 epochs; 640 steps/epoch; 6 bags/step; grouping=single; skip=True; active options=none; excluded folders=[].

- Seed 42: complete fixed split; 1 checkpoint(s); artifact directory `outputs/mil/experiments/L-AST-nolab-K12/seed42`.

### L-cnn-lab-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** trainable 4-block CNN clip encoder (log-mel 64x626) + gated attention + linear; skip-window training, 10x training windows, single split + lab recordings (483 clean / 967 infested 20 s pseudo-clips) in training [K=12].

**Recipe:** K=12; 30 epochs; 640 steps/epoch; 6 bags/step; grouping=single; skip=True; active options=lab_data; excluded folders=[].

- Seed 42: complete fixed split; 1 checkpoint(s); artifact directory `outputs/mil/experiments/L-cnn-lab-K12/seed42`.

### L-cnn-nolab-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** trainable 4-block CNN clip encoder (log-mel 64x626) + gated attention + linear; skip-window training, 10x training windows, single split, no lab data [K=12].

**Recipe:** K=12; 30 epochs; 640 steps/epoch; 6 bags/step; grouping=single; skip=True; active options=none; excluded folders=[].

- Seed 42: complete fixed split; 1 checkpoint(s); artifact directory `outputs/mil/experiments/L-cnn-nolab-K12/seed42`.

### L-mfcc30-lab-K12

**Architecture:** 4-conv CNN, mfcc13x30; gated pooling; linear classifier.

**Approach:** trainable 4-conv CNN clip encoder (whole-clip MFCC 13x30, per-coefficient standardised) + gated attention + linear; skip-window training, 10x training windows, single split + lab recordings (483 clean / 967 infested 20 s pseudo-clips) in training [K=12].

**Recipe:** K=12; 30 epochs; 640 steps/epoch; 6 bags/step; grouping=single; skip=True; active options=lab_data; excluded folders=[].

- Seed 42: complete fixed split; 1 checkpoint(s); artifact directory `outputs/mil/experiments/L-mfcc30-lab-K12/seed42`.

### L-mfcc30-nolab-K12

**Architecture:** 4-conv CNN, mfcc13x30; gated pooling; linear classifier.

**Approach:** trainable 4-conv CNN clip encoder (whole-clip MFCC 13x30, per-coefficient standardised) + gated attention + linear; skip-window training, 10x training windows, single split, no lab data [K=12].

**Recipe:** K=12; 30 epochs; 640 steps/epoch; 6 bags/step; grouping=single; skip=True; active options=none; excluded folders=[].

- Seed 42: complete fixed split; 1 checkpoint(s); artifact directory `outputs/mil/experiments/L-mfcc30-nolab-K12/seed42`.

### M-heads4

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); heads4 pooling; linear classifier.

**Approach:** trainable 4-block CNN clip encoder (log-mel 64x626) + 4-head gated attention + linear classifier.

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/M-heads4/seed42`.

### M-heads4-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); heads4 pooling; linear classifier.

**Approach:** trainable 4-block CNN clip encoder (log-mel 64x626) + 4-head gated attention + linear classifier [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/M-heads4-K12/seed42`.

### NF8-AST-lin-K12

**Architecture:** frozen AudioSet AST, 768-d; gated pooling; linear classifier.

**Approach:** frozen AST AudioSet embeddings (768-d) + gated attention + linear classifier; folder 8 removed [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=['folder_8'].

- Seed 42: partial; no complete CV result; 6 checkpoint(s); artifact directory `outputs/mil/experiments/NF8-AST-lin-K12/seed42`.

### R-AST-lin

**Architecture:** frozen AudioSet AST, 768-d; gated pooling; linear classifier.

**Approach:** frozen AST AudioSet embeddings (768-d) + gated attention + linear classifier, tree-safe leave-one-recorder-out.

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=recorder; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 6 checkpoint(s); artifact directory `outputs/mil/experiments/R-AST-lin/seed42`.
- Seed 43: complete CV; 6 checkpoint(s); artifact directory `outputs/mil/experiments/R-AST-lin/seed43`.
- Seed 44: complete CV; 6 checkpoint(s); artifact directory `outputs/mil/experiments/R-AST-lin/seed44`.
- Seed 45: complete CV; 6 checkpoint(s); artifact directory `outputs/mil/experiments/R-AST-lin/seed45`.
- Seed 46: complete CV; 6 checkpoint(s); artifact directory `outputs/mil/experiments/R-AST-lin/seed46`.

### R-Awin-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 + noise mixing + EQ tilt (A-win), tree-safe leave-one-recorder-out [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=recorder; skip=False; active options=noise, tilt; excluded folders=[].

- Seed 42: complete CV; 6 checkpoint(s); artifact directory `outputs/mil/experiments/R-Awin-K12/seed42`.
- Seed 43: complete CV; 6 checkpoint(s); artifact directory `outputs/mil/experiments/R-Awin-K12/seed43`.
- Seed 44: complete CV; 6 checkpoint(s); artifact directory `outputs/mil/experiments/R-Awin-K12/seed44`.
- Seed 45: complete CV; 6 checkpoint(s); artifact directory `outputs/mil/experiments/R-Awin-K12/seed45`.
- Seed 46: complete CV; 6 checkpoint(s); artifact directory `outputs/mil/experiments/R-Awin-K12/seed46`.

### R-B0

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 baseline: trainable 4-block CNN clip encoder (log-mel 64x626) + gated attention + linear classifier, tree-safe leave-one-recorder-out.

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=recorder; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 6 checkpoint(s); artifact directory `outputs/mil/experiments/R-B0/seed42`.
- Seed 43: started/stopped; no saved completed fold; 0 checkpoint(s); artifact directory `outputs/mil/experiments/R-B0/seed43`.

### S-skip

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 + skip windows in training (24 of the recordings within 48 h).

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=True; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/S-skip/seed42`.

### S-skip-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); gated pooling; linear classifier.

**Approach:** MIL v1 + skip windows in training (24 of the recordings within 48 h) [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=True; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/S-skip-K12/seed42`.

### T-transformer

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); transformer pooling; linear classifier.

**Approach:** trainable 4-block CNN clip encoder (log-mel 64x626) + 1 transformer layer with positional encoding + gated attention + linear classifier.

**Recipe:** K=24; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/T-transformer/seed42`.

### T-transformer-K12

**Architecture:** 4-block log-mel CNN (1→32→64→128→128); transformer pooling; linear classifier.

**Approach:** trainable 4-block CNN clip encoder (log-mel 64x626) + 1 transformer layer with positional encoding + gated attention + linear classifier [K=12].

**Recipe:** K=12; 30 epochs; 64 steps/epoch; 6 bags/step; grouping=tree; skip=False; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/T-transformer-K12/seed42`.

### XS-mfcc15

**Architecture:** 4-conv CNN, mfcc13x15; gated pooling; linear classifier.

**Approach:** trainable 4-conv CNN clip encoder (whole-clip MFCC 13x15, per-coefficient standardised) + gated attention + linear classifier; skip-window training, 10x training windows.

**Recipe:** K=24; 30 epochs; 640 steps/epoch; 6 bags/step; grouping=tree; skip=True; active options=none; excluded folders=[].

- Seed 42: partial; no complete CV result; 3 checkpoint(s); artifact directory `outputs/mil/experiments/XS-mfcc15/seed42`.

### XS-mfcc15-K12

**Architecture:** 4-conv CNN, mfcc13x15; gated pooling; linear classifier.

**Approach:** trainable 4-conv CNN clip encoder (whole-clip MFCC 13x15, per-coefficient standardised) + gated attention + linear classifier; skip-window training, 10x training windows [K=12].

**Recipe:** K=12; 30 epochs; 640 steps/epoch; 6 bags/step; grouping=tree; skip=True; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/XS-mfcc15-K12/seed42`.

### XS-mfcc30-K12

**Architecture:** 4-conv CNN, mfcc13x30; gated pooling; linear classifier.

**Approach:** trainable 4-conv CNN clip encoder (whole-clip MFCC 13x30, per-coefficient standardised) + gated attention + linear classifier; skip-window training, 10x training windows [K=12].

**Recipe:** K=12; 30 epochs; 640 steps/epoch; 6 bags/step; grouping=tree; skip=True; active options=none; excluded folders=[].

- Seed 42: complete CV; 10 checkpoint(s); artifact directory `outputs/mil/experiments/XS-mfcc30-K12/seed42`.

## All-data models and consumed holdout

H1 = FINAL-all (K24, folder 8 included); H2 = H-AST-K12 (K12, included); H3 = H-AST-noF8 (K24, excluded); H4 = H-AST-noF8-K12 (K12, excluded). All use frozen AST + gated attention + linear classifier, seed 42 and the default training budget. FINAL-all-K12 is a CNN + noise/tilt all-data fit; it was **not** one of the four holdout models.

| ID | Saved model | UW SK AUC | TW SK AUC | TW SK balanced accuracy | Group verdicts at 0.5 |
| --- | --- | ---: | ---: | ---: | --- |
| H1 | FINAL-all | 0.768 | 0.915 | 0.500 | all four called infested |
| H2 | H-AST-K12 | 0.719 | 0.894 | 0.554 | all four called infested |
| H3 | H-AST-noF8 | 0.498 | 0.782 | 0.500 | all four called healthy |
| H4 | H-AST-noF8-K12 | 0.613 | 0.821 | 0.500 | all four called healthy |

Source: [holdout results](../../mil/HOLDOUT_RESULTS.md). Window verdicts vary within H2; 'all four' describes the mean-score group decision only. The holdout has one healthy and three infested groups, 1,284 recordings, known site/time links to development. Consumed November 11 at 21:51:48 (marker as recorded). It is no longer an independent test for any pipeline.

## Partial attempts, planned methods and non-training analyses

- XS-mfcc15 K24 has 3/10 completed folds; NF8-AST-lin-K12 has 6/10. No full-CV metric is assigned to either. R-B0 seed43 has no saved completed fold. A finished log epoch or a queued command is not a completed CV run.
- Remaining XS queue entries (XS-mfcc30 K24; AST/BEATs/PANNs at K12/K24; log-mel CNN at K12/K24) have no production run directories at this snapshot. They are planned, not completed models.
- P-base, P-fn, P-pcen, P-mix, P-randeq, P-fn-mix, P-mix-randeq, P-pcen-mix are implemented/planned; no production checkpoints were found. PHASE2_RESULTS.md contains R-* reference rows only. Do not claim frequency centring, PCEN, Freq-MixStyle or random EQ improved MIL here.
- Other NF8 variants are defined but have no saved run directories. NO_F8_RESULTS.md removes folder 8 from scoring **without retraining**; it is not evidence for NF8 retraining.
- A-win K24 was deliberately skipped because zero individual augmentations passed its rule; A-win K12 ran.
- MEM-alarm, tree-day aggregation, day/night slices and folder-8 scoring exclusions reuse saved predictions; they are not separately trained neural models. See the findings report for results and limitations.
- Smoke outputs under `outputs/mil/experiments/smoke/` are software checks, excluded from scientific model totals. The original crashed/restarted XS attempt is historical and not counted twice.

## Instructions for a receiving AI agent

1. Read AGENTS.md and the report before proposing experiments. Preserve user edits and existing artifact paths.
2. Treat the consumed holdout as consumed for **all** pipelines. Never rerun holdout_eval.py or manufacture a fresh test from these groups.
3. Keep architecture, input pipeline, K, grouping, budget, seed, threshold, weighting and decision unit attached to every number.
4. Do not merge the five seed metrics into an ensemble claim; mean/SD over seed-level metrics is not averaging predictions.
5. Do not count identical FINAL/precursor predictions as replication. Do not treat selected CV as an unbiased test.
6. Missing evidence is n/a. Restore ignored artifacts explicitly; a Git clone or this appendix alone does not contain the trained models.
7. No model here establishes species identification, field sensitivity/specificity, or MIL deployment feasibility on ESP32.
