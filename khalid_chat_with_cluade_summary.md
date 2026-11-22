# Summary of Khalid's work with Claude Code

Written 5 November 2026 for a teammate joining the project. It covers the sessions up to this date. For full detail, read the documents listed at the end.

## Where things stand

- We trained small neural networks on the TreeVibes dataset to detect palm borer vibrations, and picked one model for the ESP32.
- That model **builds, fits and runs on a real ESP32 board**, and its output matches the PC exactly.
- **It is not proven to detect red palm weevil in the field.** On this dataset it mostly picks up vibration activity at 3–4 kHz and recording conditions.
- The next real step is recording with our own sensor, not more training on TreeVibes.
- All code is pushed to GitHub on `master` (latest commit `0875371`). The exception is `examples/piezo_stage_test/`, which is uncommitted and not on GitHub. It holds one Arduino sketch (`piezo_stage_test.ino`, dated 5 November) that wasn't reviewed for this summary.

## The chosen model

`stream1s_w8_d4`, trained with pos_boost 2 and folder balancing (p = 0.5), deployed as one int8 TFLite model.

- **Input:** 1 s of 8 kHz audio as a 59×64 log-mel spectrogram. Ten 1 s scores are averaged into one 10 s decision.
- **Why this one:** it fits the 64 KiB working memory and gave the best results on recorders it never trained on, after per-device calibration. That result held on a second set of training seeds.
- **Expected on an unseen device:** about 86–90% recall with 14–15% false alarms on healthy trees (3-model average). These are rough figures from development data, based on only three test recorders.
- **Its lead is small.** The runner-up is folder-balanced `stream1s_w8` (about 80% recall at 13–15% false alarms).

**Calibration is required.** The firmware has no alarm threshold. A threshold copied from another device can flag every healthy tree. For each device, record about 50 known-healthy-tree recordings and set the threshold at the 90th percentile of their scores.

The embedded model is a single model from the leave-one-recorder-out run (seed 0, fold 0). It never heard recorder 2538, so only that recorder's folders are a fair test of it: 1, 3, 5, 11, 13, 16, 18, 20, 22, 26, 34 (infested) and 24 (clean). It was not trained on all the data. This single model is weaker than the 3-model average: about 76% recall at 16% false alarms on recorder 2538.

## Measured on the ESP32

ESP32-D0WD-V3 at 160 MHz, ESP-IDF v5.3.6, one run of a real 1 s clip repeated ten times:

| Measurement | Result |
| --- | ---: |
| Firmware size | 301,664 bytes (29% of the 1 MB app partition) |
| Model working memory (arena) used | 40,652 of 65,536 bytes |
| Free RAM after setup | 227,528 bytes |
| Audio processing per 1 s | 81 ms |
| Model inference per 1 s | 348 ms |
| Output vs PC | Identical |

**Not measured yet:** power draw, a real microphone or sensor, LoRa, sleep modes, and the complete sensing-node application.

## What we learned about the data

- **The 3–4 kHz band carries the decision.** Trained on those 8 mel bands alone, a model reaches AUC 0.94; without them, about 0.60.
- **One untrained number does as well.** The level drop near 4 kHz gives F1 0.965 across the dataset, but it fails within individual recorders, so it wasn't used.
- **Recorder 8409 recorded only clean trees** (683 of the 975 clean recordings), and folder 9 is that recorder's own noise floor.
- **Every infested tree is an urban palm;** the clean trees come from other sites. Some infested folders are marked CONFIRMATION = NO.
- **Holding out folders leaks recorder identity,** because other folders from the same recorder stay in training. Use leave-one-recorder-out (`esp32_train.py --folds recorder`).
- **Thresholds don't transfer between devices,** which is why calibration is required.
- **Noise:** recall holds up with rain or wind at 10 dB (91–94%) but drops to 45% with white noise.

## What didn't help

- 10 s input models: worse than averaging 1 s windows (AUC 0.84–0.91).
- Bigger models: `w16` scored highest overall, but needs 74 KiB and was not better on new devices.
- `ds` (depthwise-separable): 87 KiB of memory and no accuracy gain.
- `w4`: too small, and varies a lot between seeds.
- 30×13 MFCC inputs: near chance.
- Distillation, 40 epochs, top-3-of-10 scoring, and 3-model ensembles (+0.005 F1 for three times the compute).

## Rules to follow

- **Do not read or score the sealed test split.** It is meant to be used once, at the end, through the notebook's gate. `score_wavs.py` refuses those recordings.
- **Do not change the locked split** in `outputs/real-training-cuda/`.
- **Do not present results as proof of field detection.** Keep simulated, validation and hardware results clearly separate.
- **Do not keep tuning on this development data.** Every selected figure is already somewhat optimistic.
- **Never invent measurements.** Power draw stays "unmeasured" until someone measures it.
- Tell Claude to read `AGENTS.md` first. It has the full rules and session log, and may not load automatically.
- Run these checks after changes:

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check app.py engineering_app.py meeting_app.py simulator tests scripts
```

## How to run things on this PC

**Score audio on the PC** (uses the TensorFlow environment in `outputs/tf-venv`):

```powershell
$env:PYTHONUTF8 = '1'
outputs/tf-venv/Scripts/python.exe scripts/score_wavs.py folder_1 folder_3 folder_5 folder_11 folder_13 folder_16 folder_18 folder_20 folder_22 folder_26 folder_34 --calibrate-folder folder_24 --calibration-share 0.5
```

Scores go to `outputs/score-wavs/scores.csv`. The output marks whether the model trained on each recording.

**Build and flash the ESP32.** The board was on COM9 and the CP210x driver is installed. Python isn't on PATH here, so add it first:

```powershell
$env:PATH = "$env:APPDATA\uv\python\cpython-3.12-windows-x86_64-none;" + $env:PATH
. $HOME\esp\esp-idf\export.ps1
cd firmware\esp32_stream1s
idf.py build
idf.py -p COM9 flash monitor
```

Press `Ctrl+]` to leave the monitor.

## What's on this PC but not on GitHub

- `outputs/`: all other trained models, results, the locked split and the Python environments. Be careful here: retraining with the same settings overwrites results, so use a new `--tag`.
- TreeVibes dataset: `%USERPROFILE%\.cache\kagglehub\datasets\potamitis\treevibes\versions\1`
- ESP-IDF: `%USERPROFILE%\esp\esp-idf`, with tools in `%USERPROFILE%\.espressif`

The recommended model *is* on GitHub, as a C array in `firmware/esp32_stream1s/main/generated/`.

## Suggested next steps

1. **Add deep-sleep duty cycling** to the firmware, for example one 10 s check per hour. This is by far the biggest energy saving, and it matters more than shrinking the model.
2. **Measure current draw** with a USB power meter or INA219. Software can't measure it, and a dev board's USB chip and regulator will inflate sleep current.
3. **Test deciding from 5 windows instead of 10** on development data, with recorder hold-out.
4. **Connect the real sensor** and capture audio (I2S or ADC) into `logmel_push()`.
5. **Characterise the sensor's noise floor** with no tree attached. The models depend on near-Nyquist behaviour that will differ on our hardware, so a TreeVibes-trained model may not transfer.
6. **Collect verified recordings:** healthy palms at the same urban sites, on the same days and with the same devices as inspection-confirmed infested palms.
7. **Only then** freeze a model and threshold and run the one-time final test.

## Read more

- `docs/ESP32_DEV_DAY_REPORT.md`: the main report, with evidence for everything above
- `firmware/README.md`: on-board results, build steps and calibration
- `AGENTS.md`: project rules and full session log
- `docs/ACOUSTIC_DATA_AUDIT.md`: data limitations and the collection plan
- `docs/ESP32_TFLM_PLAN.md`: the ESP32 model pipeline
- Slide deck (27 slides): https://claude.ai/artifact/5c6w8KdkJbhWmzWFUfuixw. It's private, so ask Khalid to share it. Slides on the tested model say "unmeasured" for on-board results, because they were written before the board test.
