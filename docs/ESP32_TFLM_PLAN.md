# ESP32-WROOM32 TFLite Micro plan — 1 November 2026

Goal: lightweight but more capable int8 models that run under TensorFlow Lite Micro (esp-tflite-micro with ESP-NN) on an ESP32-WROOM32, with recall raised as far as possible while limiting the precision loss. Labels are weak borer annotations, not confirmed red palm weevil.

## Limiting factors found (measured with `scripts/esp32_limits.py`)

1. **Evaluation optimism (largest measured effect).** Earlier operating points were chosen on the same pooled out-of-fold scores they were reported on. When each held-out fold instead uses a threshold chosen on the other four folds, the 10 s models' FPR rises from 8–11% to 23–30% and precision falls from ~93% to ~81–85%, with recall roughly unchanged. Score scales shift between recording folders, so thresholds do not transfer well. All new results use cross-fitted thresholds.
2. **Tensor arena, not weights.** The current 10 s CNNs keep 320×64 resolution through the first convolution: int8 arena lower bounds are 100 KiB (w4), 200 KiB (w8) and 800 KiB (w32), while weights are only 6–395 KiB. The ESP32 has ≤320 KiB DRAM shared with the application, and the reference harness reserves 80 KiB. The 1 s w8 CNN needs about 37 KiB. Target: arena ≤ 96 KiB, weights ≤ 100 KiB.
3. **Audio buffering.** Retaining 10 s of int16 audio costs 160 KiB. The firmware must stream: compute log-mel frames as audio arrives and keep only the 320×64 int8 input (20 KiB), or score 1 s windows as they complete.
4. **Quantisation and conversion are unmeasured.** No model has yet passed float → int8 TFLite conversion. BatchNorm folding, int8 post-training quantisation and padding semantics (PyTorch symmetric vs TensorFlow `same`) can change outputs and must pass parity checks.
5. **Seed variance.** Recall at a fixed FPR varies by up to 30 points between seeds. Single-seed results are unreliable, and on-device ensembles multiply arena time and latency.
6. **Data confounding (limits real-world validity).** Only five negative folders exist in development data; the same-device negative folder 24 is scored as infested by every model. Training-method changes cannot fix this. Same-host, same-site, same-device negatives are required for a trustworthy field claim.
7. **Toolchain and hardware.** TensorFlow was not installed (now isolated in the ignored `outputs/tf-venv`), and there is no ESP-IDF toolchain or connected board. Host TFLite int8 results are the closest available proxy; on-board arena, latency, energy and feature parity remain unmeasured.

## Plan

1. **Honest evaluation protocol.** Same development-only, folder-grouped 5-fold split, test never read. Primary metric: cross-fitted operating points. Recall-oriented rule: threshold maximising F2 on the other folds (recall weighted 2× precision), plus a precision-floor rule (highest recall whose precision on the other folds is ≥ 90%). Report recall, precision, FPR, F1/F2 and ROC-AUC, with 3 seeds.
2. **TFLM-friendly architectures** (Conv2D, DepthwiseConv2D, MaxPool, Mean, FullyConnected, Add, Logistic; all supported by TFLM, with ESP-NN kernels for convolutions):
   - 10 s strided-stem CNN: stride-2 stem so the first activation is 160×32.
   - 10 s DS-CNN: stride-2 stem, depthwise-separable blocks with residual adds (more depth per byte).
   - 1 s streaming CNN: the existing best 1 s w8 model, run on ten consecutive 1 s windows of a 10 s capture and pooled on the MCU (arena ~37 KiB; the decision uses the full 10 s).
3. **Recall-oriented training.** Positive-weighted BCE (pos_weight ×2 relative to class balance), SpecAugment and mixup, and knowledge distillation from a wider teacher trained only on the same training folds.
4. **Export path.** Train in PyTorch on the RTX 2070; port weights into an identical Keras graph (explicit zero-padding to match PyTorch); assert float parity; convert to full-int8 TFLite with a training-fold representative dataset; score every held-out fold with the **int8 TFLite** model, so reported metrics are what the device arithmetic produces.
5. **Device-readiness checks per model.** Measured TFLite tensor sizes / arena estimate, int8 weight size, MACs, operator list against the TFLM resolver, float-vs-int8 agreement, and a C header (`model_data.cc`) plus an input fixture for the existing ESP-IDF benchmark harness (`outputs/reference-ml-for-senior/deployment/esp32`).
6. **Blocked until hardware/tooling:** an on-board build, arena allocation, latency, energy and C log-mel parity need an ESP-IDF install and a connected ESP32-WROOM32.

## Results — int8 TFLite, cross-fitted thresholds (1 November 2026)

Pipeline: `scripts/esp32_train.py` (PyTorch, RTX 2070; 3 seeds × 5 folder-grouped folds, positive weight 2× class balance, SpecAugment, mixup) → `scripts/esp32_export.py` (TensorFlow 2.18.1 CPU in `outputs/tf-venv`; Keras twin from `scripts/esp32_arch.py`, float parity ≤1.4e-6 on every fold model, full-int8 PTQ calibrated on training-fold data, held-out folds scored with the int8 reference-kernel interpreter). Development data only (2,542 recordings: 1,567 positive / 975 negative); the test partition was never read. Thresholds are chosen on the other four folds and applied to the held-out fold.

| Candidate (3-seed ensemble, int8) | int8 .tflite | Arena lower bound | ROC-AUC (int8 / float) | F2 rule: recall / precision / FPR | Precision-floor rule: recall / precision |
| --- | ---: | ---: | --- | --- | --- |
| **stream1s_w8**: 1 s windows ×10, averaged on MCU | 13.8 KB per model | 37.6 KB | 0.922 / 0.925 | **98.9% / 87.0% / 23.7%** | 27.4% / 58.7% |
| strided10s_w8: 10.24 s input, stride-2 stem | 72.1 KB | 62.2 KB | 0.905 / 0.906 | 98.7% / 80.6% / 38.3% | 60.9% / 75.1% |
| dscnn10s_res: 10.24 s depthwise-separable residual | 34.2 KB | 62.2 KB | 0.885 / 0.886 | 99.0% / 79.9% / 39.9% | 53.6% / 74.8% |

- **Int8 conversion is not a limiting factor:** AUC drops by at most 0.003 and the operating points barely move.
- **All operators are TFLM-supported** (ADD, CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, LOGISTIC, MAX_POOL_2D, MEAN, MUL, PAD), and every arena estimate is under the 96 KiB target. These are lower bounds without planner overhead.
- **Best ESP32 candidate:** stream1s_w8. Running three 13.8 KB models in sequence on ten 1 s windows reuses one ~38 KB arena and needs no 10 s audio buffer.
- **The 90% precision-floor rule fails out of fold:** thresholds that reach 90% precision on four folds give 59–75% precision on the fifth. Folder-to-folder score shift (limiting factor 1) is the binding constraint, not model capacity. Recall-weighted (F2) thresholds transfer much better.
- The F2 operating point flags 23.7% of negative recordings. Precision (87%) depends on this dataset's 61.6% positive prevalence and would be lower at a realistic field prevalence.
- Every candidate still scores the same-device negative folder 24 as infested (see `docs/ACOUSTIC_DATA_AUDIT.md`).
- **Still unmeasured:** on-board build, real arena allocation, latency, energy and C log-mel parity. Artifacts are in `outputs/esp32-tflm/full/<candidate>/`: `model_int8.tflite`, `model_data.cc`, `fixture_input_int8.npy`, plus `device_report.json`. They are ready for the ESP-IDF harness in `outputs/reference-ml-for-senior/deployment/esp32` once ESP-IDF and a board are available.
