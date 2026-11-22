# TreeVibes data audit for tree-level MIL

Written 9 November 2026, for Section 3 of the MIL handoff. Produced by `mil/data_audit.py`; the raw tables are in `mil/audit/` (`files.csv`, `workbook_vs_disk.csv`, `duplicate_md5.csv`, `shared_filenames.csv`, `timing.csv`, `near_pairs.csv`, `components_25m.json`, `audit_log.txt`).

The audit reads only directory listings, filenames, raw file bytes (for MD5) and `TreeVibes.xlsx`. No audio was decoded, no features were made and no model was run.

**Status: decisions made and applied (Section 8).** Sections 1–7 record the audit as it was first run. Where Section 6's first grouping proposal differs from Section 8, Section 8 is what the experiments use.

## 0. Existing clip-level work in this repository

The clip-level CNNs described in the handoff (CNN-A/CNN-B, 15×13 MFCC, random 70/15/15 file split) are **not in this repository**.

| Work | How it was split | Leaky? |
| --- | --- | --- |
| Original CUDA campaign (`notebooks/TreeVibes_Three_Track_Training.ipynb`) | Locked split by connected component; every folder sits wholly in train, val or test | Not file-level. But nearby folders were not merged, so folders 29 (test), 30 (val) and 31 (train) cross the split (Section 4). |
| Grouped-CV campaigns and the ESP32 models (`scripts/grouped_cv_*.py`, `scripts/esp32_*.py`) | Folder-grouped 5-fold, then leave-one-recorder-out, on train+val only | Not file-level. Near-pair leakage between neighbouring folders is possible in the folder-grouped runs. |
| ML-for-Senior reference (`outputs/reference-ml-for-senior/`) | One file per group (`file_and_duplicate_only`), so effectively a random file-level split | **Yes, file-level split, leaky.** Already documented in `docs/ACOUSTIC_DATA_AUDIT.md`; `scripts/split_leakage_demo.py` shows 100% recall under that split vs 66% when folders are held out. |

## 1. Directories

| Area | Contents |
| --- | --- |
| `field/field/train/infested/` | `folder_1`–`folder_6`, `folder_11`–`folder_23`: 19 folders, 731 WAVs |
| `field/field/train/clean/` | `folder_7`–`folder_10`, `folder_24`, `folder_25`, `folder_35`: 7 folders, 1,754 WAVs (includes the nested `folder_9/folder_9`, 20 WAVs) |
| `field/field/test/` | `folder_26`–`folder_34`: 9 folders, 1,654 WAVs; plus `difficult_cases/`, 15 WAVs |
| `lab/lab/` | `clean/` 967 and `infested/` 53,676 WAVs; flat, no sub-folders |

- **Field total:** 4,154 WAVs (2,485 train + 1,654 test folders + 15 difficult cases). The paper's 4,165 is 11 more; this wasn't resolved.
- **Test folders are labelled.** The `test/` directory carries no class sub-folder, but the workbook gives every one of `folder_26`–`folder_34` `AUDIO = 1`. **They are all infested; there is no healthy tree among them.**
- **`difficult_cases/`** (15 WAVs) has no workbook row and no label.
- **Field filenames:** `F_<YYYYMMDDhhmmss>_<index>[_T<temp>].wav`. The temperature suffix is missing in folders 12, 14, 17, 19, 21, 22, 25–33 and part of 35.

## 2. Workbook vs disk

- **Matching:** all 35 workbook rows have a folder on disk, and all 35 folders have a row.
- **Labels:** every folder's directory label agrees with `AUDIO`.
- **Folder 35** (`AUDIO = 0`, `VISUAL = B`, comment "Negative from infested trees") is in `train/clean/`, but it is not one healthy tree; see Section 3 and Section 5.
- **`VISUAL` vs `AUDIO`:**
  - **No visible signs:** folders 19, 21, 22 and 23 are `VISUAL = X` but `AUDIO = 1`.
  - **Not confirmed:** folders 3, 4, 18, 22 and 23 have `CONFIRMATION = NO`.
- **Folders without coordinates or a device:** 10, 15 and 35. Folders 10 and 15 are cut trunks on a balcony ("tree trank placed in a balcony … waveguide"), not standing trees.
- **Hidden character:** the `VISUAL` value "Α" in folders 3–6 is a Greek capital alpha, not a Latin A. Any code comparing with `"A"` must normalise it.
- **Species:** folders 7, 8 and 9 are orange, fig and apricot trees. Tree species is out of scope for now, at the user's request.

## 3. Duplicates

**Identical bytes (MD5, field files):** 62 files in 31 groups.

| Where | What | Files |
| --- | --- | ---: |
| `folder_9/folder_9` | Exact byte copies of files in `folder_9`, with the same filenames. They inflate folder 9 from 508 to 528. **Drop them.** | 20 |
| `folder_27` + `folder_34` | Identical files across two folders, with the same filenames | 10 pairs |
| `folder_1` + `folder_35` | `F_20200218002959_23_T9.0.wav` appears in both; that's one infested-tree recording also labelled clean | 1 pair |

**Same filename, different bytes:**
- **`folder_3` + `folder_4`:** 15 filenames appear in both folders with different contents. They're two devices whose files share timestamp and index.

**PCM-level duplicates:** the existing manifest (which excludes folder 35) has 90 PCM-identical recordings, 25 of them across folders.

## 4. Nearby folders (less than 25 m apart)

There are 57 pairs; see `near_pairs.csv`.

- **Folders 20 and 22:** identical coordinates and the same device, recorded a day apart.
- **Pairs recorded at the same time on two devices:**
  - 1+2 (4.6 m)
  - 5+6 (20.9 m)
  - 29+30 (1.7 m)
  - 29+31 and 30+31 (about 10 m)
  - 26+28 (12.6 m)
  - 27+28 (23.9 m)

  Many other folder pairs also start within minutes of each other on two devices (Section 5). The data can't tell two neighbouring trees from one tree with two probes.
- **Folders 6 and 8:** the same device, 7 m apart, on consecutive days. Folder 6 is infested; folder 8 is a clean fig tree.
- **Merging every pair under 25 m** ("merge when in doubt") gives:
  - **A mixed-label group:** 5+6+8.
  - **One 12-folder group:** 16–23, 26–28 and 34, with 174 recordings.
- **Locked-split crossings:** folders 29 (sealed test), 30 (val) and 31 (train) were recorded at the same time within 11 m. 26/27 (val) and 28 (train) are within 25 m. **The existing sealed test is therefore not independent of development data at the near-tree level.**

## 5. Timing

See `timing.csv`.

- **Recording interval:** about every 5 minutes (median gap 0.08 h in most folders), **not hourly**. Folders 1–6 have median gaps of 10–43 minutes, which suggests a subset of 5-minute recordings was kept. That fits the hand-picking suspicion, at a 5-minute scale.
- **Time span:** most folders cover **less than a day**; many cover 25–90 minutes.
- **What K = 24 means:** 24 consecutive recordings is about 2 hours. **12 of the 19 training-infested folders have fewer than 24 recordings**, so they give a single window.
- **Paired starts:** folders were recorded in device pairs that start within minutes of each other: 1/2, 3/4, 5/6, 7/8, 12/14, 13/14, 16/17, 18/19, 20/21, 22/23, 24/25, 26/27, 28–31 and 32/33.
- **Gaps:** only folders 34 and 35 have gaps over 7 days.
- **Folder 34:** spans 40 days, and 10 of its files are identical to folder 27's.
- **Folder 35:** 333 recordings over 96 days (18 February – 23 May) on 8 distinct days. Those days match the infested folders' sessions, and one file is an exact copy of a folder 1 recording.
  - **Conclusion:** folder 35 is a pool of quiet clips taken from infested trees, not a healthy tree.
  - **This affects handoff finding 6:** folder 35's high-scoring windows are clips from infested trees, labelled clean.
- **Folder 9:** contains 20 duplicate timestamps, caused by the nested copies.

## 6. Proposed tree groups (not final)

`mil/groups.json` holds the proposal; it is marked `"status": "proposed"` until the user confirms.

Proposal A follows the handoff's 25 m rule, with two exceptions:
- **Folder 8 stays separate.** It is a clean fig tree, merging it would create a mixed-label group, and it was recorded on a different day.
- **Folder 35 is excluded** (Section 5).
- **Folder 9:** the 20 nested copies are dropped.

**Result:** 9 infested groups and 6 healthy groups (7, 8, 9, 10, 24, 25), counting all labelled folders including 26–34.

| Group | Folders | Label |
| --- | --- | --- |
| g01 | 1, 2, 3 | infested |
| g02 | 4 | infested |
| g03 | 5, 6 | infested |
| g04 | 7 | healthy (orange) |
| g05 | 8 | healthy (fig) |
| g06 | 9 | healthy (apricot) |
| g07 | 10 | healthy (cut trunk) |
| g08 | 11, 12, 13, 14 | infested |
| g09 | 15 | infested (cut trunk) |
| g10 | 16–23, 26, 27, 28, 34 | infested |
| g11 | 24 | healthy ("healthy trees", plural) |
| g12 | 25 | healthy ("healthy trees", plural) |
| g13 | 29, 30, 31 | infested |
| g14 | 32 | infested |
| g15 | 33 | infested |

**Where proposal A leaves the handoff's numbers:**
- **Cross-validation:** about 15 groups, not 26 trees.
- **Healthy groups:** 6, and only two of them (24 and 25, "forest") might be the same kind of host. Healthy and infested trees mostly differ in species, site and weather.
- **Lockbox:** none is possible, because folders 26–34 are all infested and were already used in development (Section 7).

## 7. Contradictions and decisions needed before any training

1. **The sealed holdout in `AGENTS.md` vs 10-fold CV over all trees.**
   - **The locked split's sealed test** is folders 10, 15, 29 and 32 (1,264 recordings). No model has been scored on it.
   - **The handoff's plan** would train and evaluate on all trees, which uses up that holdout. The repository rule is to keep it sealed; changing it needs a deliberate decision to retire the old split.
   - **Option 1:** exclude folders 10, 15, 29 and 32 from MIL work. That leaves **5 healthy groups**.
   - **Option 2:** formally retire the old split and use everything in CV. Then there is no independent test at all.
2. **There is no usable lockbox.**
   - **Folders 26–34 are all infested,** so they can't measure false alarms.
   - **They were already used in development:** 26, 27, 30 and 34 in val; 28, 31 and 33 in train.
3. **Folder 35 is not a healthy tree.** It should be excluded (proposed), not used as one of 7 healthy trees.
4. **Grouping.** The handoff's "merge when in doubt" at 25 m creates a mixed-label group (5+6+8) and one 12-folder group. Proposal A is above; the alternative is merging only pairs under 5 m plus the folders that share files.
5. **Recording interval is about 5 minutes, not hourly.** This changes the meaning of K = 24 (about 2 hours, not a day) and of the 36 h / 48 h window rules. The `MEM-alarm` decay `0.8 ** (1/24)` "per recording" would then decay over about 2 hours, not a day.
6. **Very small effective sample.** About 5–6 healthy groups means the bootstrap intervals will be very wide and almost no comparison can pass the improvement rule. The plan's roughly 150 trainings (12–18 GPU hours) may give little information. A reduced plan may be better: B0, B0-s2, then the most informative 3–4 runs.

## 8. Decisions (user, 9 November 2026) and how they were applied

1. **Holdout: kept sealed, as whole tree groups.**
   - **Sealed groups:** folders 10, 15, 29 and 32, plus **folder 30**, which is the same tree as 29 (1.7 m apart, recorded at the same time).
   - **Enforcement:** `mil/common.py` refuses to index any sealed folder, and none of their audio was read.
   - **Earlier repo models trained on folder 30** (it was in the locked split's validation part). Any of their holdout results on folder 29 would not be independent; this is noted in `mil/RESULTS.md`.
   - **Folder 31 is not sealed.** It is 9.5 m from folder 29 and was recorded at the same time, so it is a near neighbour, not same-tree evidence under the 5 m rule.
2. **Folder 35: excluded completely** (neither healthy nor infested).
   - **Correction to handoff finding 6:** folder 35's high-scoring windows came from a pool of clips from infested trees, so they may contain real borer activity. They are not clean evidence of false alarms on healthy trees.
3. **Grouping: same-tree evidence only, never across labels.**
   - **Distance threshold:** positions under **5.0 m** apart, or shared recordings.
   - **Merges:**
     - 1+2 (4.6 m)
     - **3+4: 15 recordings with identical PCM audio** (file headers differ, so MD5 missed them)
     - 12+14 (3.9 m)
     - 16+17+18 (3.1 m and 3.5 m)
     - 19+23 (4.5 m)
     - 20+22 (0 m)
     - 27+34 (10 byte-identical files)
     - 29+30 (sealed)
   - **Duplicates removed:**
     - 20 nested `folder_9/folder_9` copies
     - 25 duplicate recordings: 15 from folders 3/4 and 10 from folders 27/34
   - **No recording counts twice:** 2,477 development recordings remain.
   - **The 25 m site grouping** is kept in `groups.json` (`site_groups_25m`), to be used later in one B0 sensitivity run.
4. **Scope:** build the infrastructure and monitoring, then run B0 and B0-s2. K, the memory decay, the skip-window span and the rest of the run list are decided after that report.

**Cross-validation data** (from `mil/groups.json` and `mil/folds.json`):
- **Tree groups:** 21, of which 5 are healthy (folders 7, 8, 9, 24, 25) and 16 are infested.
- **Folds:** 10, from `StratifiedGroupKFold(10, shuffle=True, random_state=42)` over groups. Folds 1, 4, 6, 7 and 9 have no healthy group.
