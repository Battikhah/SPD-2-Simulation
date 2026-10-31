# Continue on another device

Repository: https://github.com/Battikhah/SPD-2-Simulation (private), branch `master`.

## Clone and run the simulator

Authenticate GitHub with access to the private repository, then run:

```sh
git clone https://github.com/Battikhah/SPD-2-Simulation.git
cd SPD-2-Simulation
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m streamlit run app.py
```

On Windows, use `.venv\Scripts\python` instead of `.venv/bin/python`.
Open the cloned folder in Codex. Git LFS is not required for this code-only clone.

## What is included

The simulator source, tests, documentation, configuration, small examples, the
training notebook, audio preparation and burst-extraction scripts, and the
source-selection manifest are included. The audio scripts and guides retain their
original paths under `outputs/treevibes-exciter-2026-10-30/`; only these explicitly
tracked reproduction files are committed from that otherwise ignored directory.

**Audio files, trained models, feature caches and generated training results are
intentionally excluded. Regenerate them on the destination PC.** Virtual
environments and disposable caches are excluded too. The notebook retains its
previous executed text/results as context; rerun it for new results.

## Regenerate models and training outputs

Open `notebooks/TreeVibes_Three_Track_Training.ipynb` and run its dependency/setup
cells. They list the additional training packages, including scikit-learn,
soundfile, matplotlib, openpyxl, joblib, PyTorch and optional TensorFlow/kagglehub.

Supply the original TreeVibes dataset separately and set `TREEVIBES_DATA_ROOT` to
its location, set the notebook's `ARCHIVE` to a local dataset ZIP, or enable
`DOWNLOAD_IF_MISSING` (with Kaggle access configured if required). The original
source dataset is external to this repository and is not transferred by Git.
Replace the notebook's source-device default path before using a local dataset.

Start with `PRESET='smoke'`, then use `standard` or `full` for the intended
campaign. Outputs and models are regenerated in the configured output directory.
Run one training campaign at a time. A synthetic smoke run checks the pipeline;
it does not establish real-data detection performance. Preserve the notebook's
lineage review and final-evaluation gates.

## Regenerate playback audio and bursts

Install the additional audio dependencies:

```sh
.venv/bin/python -m pip install soundfile kagglehub
.venv/bin/python outputs/treevibes-exciter-2026-10-30/prepare_audio.py --selfcheck
.venv/bin/python outputs/treevibes-exciter-2026-10-30/prepare_audio.py
.venv/bin/python -m unittest discover -s outputs/treevibes-exciter-2026-10-30/rpw_bursts -p 'test_extract_bursts.py'
.venv/bin/python outputs/treevibes-exciter-2026-10-30/rpw_bursts/extract_bursts.py
```

`prepare_audio.py` downloads the selected TreeVibes v1 source clips through
kagglehub and verifies hashes, then recreates the playback set. Keep the committed
`selection.json` beside it: this lets it reuse the selected source metadata without
requiring the source computer's external audit CSV. Kaggle download access and
sufficient disk space are required. The script uses kagglehub internal download
APIs, so compatibility with the installed version must be checked if download fails.
The burst extractor depends on the parent audio and manifest generated first.

Read the retained `EXCITER_GUIDE.md` and `rpw_bursts/README.md` before physical
playback. Generic lab infestation labels and processed bursts do not establish
confirmed RPW identity or measured physical playback performance. Paths in older
guides and executed notebook messages may still refer to the source computer.

## Verification and continuation

The simulator test suite passed all 90 tests before transfer. Application lint
passed; repository-wide lint reported 73 existing training-notebook findings.
No application behavior or notebook training logic was changed for the transfer.

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check app.py engineering_app.py simulator tests
```

Git transfers project files. Codex account sign-in does not install dependencies,
copy external datasets, or resume source-device processes. Use this handoff and
the repository documentation as the destination Codex session's starting context.
