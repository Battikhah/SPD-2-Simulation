"""Experiment definitions for the tree-level MIL run list (handoff Section 7). Every run is B0 plus the
listed changes; `approach` is the human-readable label used in every results table.

K=12 variants are the same runs with the suffix "-K12" (window of 12 recordings, about 1 hour); they are
compared with B0-K12, and their seed noise comes from B0-s2-K12.
"""

BASE = dict(k=24, per_class=3, epochs=30, steps=64, lr=3e-4, weight_decay=1e-4, max_span_h=36.0, tries=20,
            amp=True, accum=1, workers=6, embed_batch=256, input="cnn", classifier="linear", pooling="gated",
            noise=False, lab=False, tilt=False, specaug=False, hardneg=False, skip=False, skip_span_h=48.0,
            grouping="tree", windows="per folder, time order",
            group_sampling="tree group uniform within class, folder proportional to clip count",
            schedule="Adam + cosine over epochs, last epoch kept")

CNN = "trainable 4-block CNN clip encoder (log-mel 64x626)"
EMB = {"panns": "frozen PANNs CNN14 embeddings (AudioSet, 2048-d)",
       "beats": "frozen BEATs iter3+ AS2M embeddings (768-d)",
       "ast": "frozen AST AudioSet embeddings (768-d)"}
MFCC_CNN = "trainable 4-conv CNN clip encoder (whole-clip MFCC 13x{t}, per-coefficient standardised)"
XS = dict(skip=True, steps=640, skip_eval_n=100)
XS_TXT = "skip-window training, 10x training windows"
# L-*: one fixed tree-grouped split (no CV): test = tree folds 5, 6, 8 of mil/folds.json, i.e. healthy folders 8
# and 24 (both share a recorder with infested trees) and infested folders 5, 11, 19+23, 28.
SINGLE_TEST = ("g03", "g06", "g08", "g12", "g15", "g19")
LB = dict(grouping="single", skip=True, steps=640, skip_eval_n=100)
LB_TXT = "skip-window training, 10x training windows, single split"
LAB_TXT = "+ lab recordings (483 clean / 967 infested 20 s pseudo-clips) in training"
NF8 = ["folder_8"]
PB = dict(grouping="recorder")  # phase-2 base; training-window settings are fixed in mil/PHASE2_PLAN.md
PB_TXT = "tree-safe leave-one-recorder-out"

RUNS = {
    "B0": (dict(seed=42), f"MIL v1 baseline: {CNN} + gated attention + linear classifier"),
    "B0-s2": (dict(seed=43), "MIL v1 baseline, second seed (seed-noise check)"),
    "E-PANNs-lin": (dict(input="emb_panns"), f"{EMB['panns']} + gated attention + linear classifier"),
    "E-PANNs-mlp": (dict(input="emb_panns", classifier="mlp"), f"{EMB['panns']} + gated attention + MLP classifier"),
    "E-BEATs-lin": (dict(input="emb_beats"), f"{EMB['beats']} + gated attention + linear classifier"),
    "E-BEATs-mlp": (dict(input="emb_beats", classifier="mlp"), f"{EMB['beats']} + gated attention + MLP classifier"),
    "E-AST-lin": (dict(input="emb_ast"), f"{EMB['ast']} + gated attention + linear classifier"),
    "E-AST-mlp": (dict(input="emb_ast", classifier="mlp"), f"{EMB['ast']} + gated attention + MLP classifier"),
    "H-mlp": (dict(classifier="mlp"), f"{CNN} + gated attention + MLP classifier (128-128-1)"),
    "A3-hardneg": (dict(hardneg=True), "MIL v1 + hard-negative mining (healthy windows scoring > 0.2)"),
    "A2-noise": (dict(noise=True), "MIL v1 + noise mixing from other healthy training trees"),
    "A1-lab": (dict(lab=True), "MIL v1 + synthetic infested windows (lab pulses mixed into healthy clips)"),
    "A4-tilt": (dict(tilt=True), "MIL v1 + EQ tilt augmentation (+/-3 dB)"),
    "A5-specaug": (dict(specaug=True), "MIL v1 + SpecAugment (1 band span, 1 time span)"),
    "A-all": (dict(hardneg=True, noise=True, lab=True, tilt=True, specaug=True),
              "MIL v2: MIL v1 + all five augmentations (A1-A5)"),
    "S-skip": (dict(skip=True), "MIL v1 + skip windows in training (24 of the recordings within 48 h)"),
    "M-heads4": (dict(pooling="heads4"), f"{CNN} + 4-head gated attention + linear classifier"),
    "T-transformer": (dict(pooling="transformer"),
                      f"{CNN} + 1 transformer layer with positional encoding + gated attention + linear classifier"),
    "R-B0": (dict(grouping="recorder"), f"MIL v1 baseline: {CNN} + gated attention + linear classifier, "
             "tree-safe leave-one-recorder-out"),
    "R-AST-lin": (dict(input="emb_ast", grouping="recorder"),
                  f"{EMB['ast']} + gated attention + linear classifier, tree-safe leave-one-recorder-out"),
    "R-Awin": (dict(noise=True, tilt=True, grouping="recorder"),
               "MIL v1 + noise mixing + EQ tilt (A-win), tree-safe leave-one-recorder-out"),
    # XS-*: skip-window training with 10x the training windows (30 epochs x 640 steps x 6 bags = 115,200 bags per
    # fold, vs 11,520), tree folds; test also scored on 100 random skip windows per held-out folder.
    "XS-cnn": (dict(XS, input="cnn"), f"{CNN} + gated attention + linear classifier; {XS_TXT}"),
    "XS-mfcc15": (dict(XS, input="mfcc13x15"), f"{MFCC_CNN.format(t=15)} + gated attention + linear classifier; {XS_TXT}"),
    "XS-mfcc30": (dict(XS, input="mfcc13x30"), f"{MFCC_CNN.format(t=30)} + gated attention + linear classifier; {XS_TXT}"),
    "XS-AST-lin": (dict(XS, input="emb_ast"), f"{EMB['ast']} + gated attention + linear classifier; {XS_TXT}"),
    "XS-BEATs-lin": (dict(XS, input="emb_beats"), f"{EMB['beats']} + gated attention + linear classifier; {XS_TXT}"),
    "XS-PANNs-lin": (dict(XS, input="emb_panns"), f"{EMB['panns']} + gated attention + linear classifier; {XS_TXT}"),
    # P-*: device-robustness phase (research-backed, see mil/PHASE2_PLAN.md). Tree-safe leave-one-recorder-out
    # folds (the test recorder is never heard in training), log-mel CNN, compared with P-base at the same K.
    "P-base": (dict(PB), f"MIL v1 baseline: {CNN} + gated attention + linear classifier; {PB_TXT}"),
    "P-fn": (dict(PB, front="fn"), f"{CNN} on frequency-centred log-mel + gated attention + linear; {PB_TXT}"),
    "P-pcen": (dict(PB, front="pcen"), f"{CNN} on PCEN mel + gated attention + linear; {PB_TXT}"),
    "P-mix": (dict(PB, mixstyle=True), f"{CNN} + Freq-MixStyle + gated attention + linear; {PB_TXT}"),
    "P-randeq": (dict(PB, randeq=True), f"{CNN} + random-EQ augmentation + gated attention + linear; {PB_TXT}"),
    "P-fn-mix": (dict(PB, front="fn", mixstyle=True),
                 f"{CNN} on frequency-centred log-mel + Freq-MixStyle + gated attention + linear; {PB_TXT}"),
    "P-mix-randeq": (dict(PB, mixstyle=True, randeq=True),
                     f"{CNN} + Freq-MixStyle + random EQ + gated attention + linear; {PB_TXT}"),
    "P-pcen-mix": (dict(PB, front="pcen", mixstyle=True),
                   f"{CNN} on PCEN mel + Freq-MixStyle + gated attention + linear; {PB_TXT}"),
    "L-cnn-lab": (dict(LB, input="cnn", lab_data=True), f"{CNN} + gated attention + linear; {LB_TXT} {LAB_TXT}"),
    "L-cnn-nolab": (dict(LB, input="cnn"), f"{CNN} + gated attention + linear; {LB_TXT}, no lab data"),
    "L-mfcc30-lab": (dict(LB, input="mfcc13x30", lab_data=True),
                     f"{MFCC_CNN.format(t=30)} + gated attention + linear; {LB_TXT} {LAB_TXT}"),
    "L-mfcc30-nolab": (dict(LB, input="mfcc13x30"),
                       f"{MFCC_CNN.format(t=30)} + gated attention + linear; {LB_TXT}, no lab data"),
    "L-AST-lab": (dict(LB, input="emb_ast", lab_data=True),
                  f"{EMB['ast']} + gated attention + linear; {LB_TXT} {LAB_TXT}"),
    "L-AST-nolab": (dict(LB, input="emb_ast"), f"{EMB['ast']} + gated attention + linear; {LB_TXT}, no lab data"),
    # H-*: frozen-for-holdout models (mil/HOLDOUT_PREREG.md), trained on all 21 development trees; same config as
    # E-AST-lin(-K12). H-AST at K = 24 is FINAL-all (already trained), so only the others are defined here.
    "H-AST": (dict(input="emb_ast", train_all=True),
              f"{EMB['ast']} + gated attention + linear classifier; trained on all development trees"),
    "H-AST-noF8": (dict(input="emb_ast", train_all=True, exclude_folders=["folder_8"]),
                   f"{EMB['ast']} + gated attention + linear classifier; trained on all development trees except folder 8"),
    # NF8-*: the best runs retrained with healthy folder 8 removed from training and testing (old runs kept).
    "NF8-AST-lin": (dict(input="emb_ast", exclude_folders=NF8),
                    f"{EMB['ast']} + gated attention + linear classifier; folder 8 removed"),
    "NF8-R-AST-lin": (dict(input="emb_ast", grouping="recorder", exclude_folders=NF8),
                      f"{EMB['ast']} + gated attention + linear classifier, tree-safe leave-one-recorder-out; folder 8 removed"),
    "NF8-skip": (dict(skip=True, exclude_folders=NF8),
                 "MIL v1 + skip windows in training (K recordings within 48 h); folder 8 removed"),
    "NF8-Awin": (dict(noise=True, tilt=True, exclude_folders=NF8),
                 "MIL v1 + noise mixing + EQ tilt (A-win); folder 8 removed"),
    "B0-site25": (dict(seed=42, grouping="site25"),
                  "MIL v1 baseline with folds by 25 m site groups (sensitivity check)"),
}
AUG_RUNS = {"A3-hardneg": "hardneg", "A2-noise": "noise", "A1-lab": "lab", "A4-tilt": "tilt", "A5-specaug": "specaug"}
AUG_NAMES = {"hardneg": "hard negatives", "noise": "noise mixing", "lab": "lab pulses", "tilt": "EQ tilt",
             "specaug": "SpecAugment"}
LIST_K24 = ["E-PANNs-lin", "E-PANNs-mlp", "E-BEATs-lin", "E-BEATs-mlp", "E-AST-lin", "E-AST-mlp", "H-mlp",
            "A3-hardneg", "A2-noise", "A1-lab", "A4-tilt", "A5-specaug", "A-all", "A-win", "S-skip", "M-heads4",
            "T-transformer", "FINAL", "B0-site25"]
LIST_K12 = ["B0", "B0-s2"] + [r for r in LIST_K24 if r != "B0-site25"]


def base_name(name):
    return name[:-4] if name.endswith("-K12") else name


def k_of(name):
    return 12 if name.endswith("-K12") else 24


def config(name, extra=None):
    """Full config for a run name (dynamic runs A-win / FINAL-* pass their changes in `extra`)."""
    b = base_name(name)
    cfg = dict(BASE)
    if b in RUNS:
        changes, approach = RUNS[b]
    else:
        changes, approach = extra["changes"], extra["approach"]
    cfg.update(seed=42)
    cfg.update(changes)
    cfg["k"] = k_of(name)
    cfg["approach"] = approach + (" [K=12]" if cfg["k"] == 12 else "")
    return cfg


def approach(name):
    b = base_name(name)
    if b in RUNS:
        return RUNS[b][1] + (" [K=12]" if k_of(name) == 12 else "")
    return None
