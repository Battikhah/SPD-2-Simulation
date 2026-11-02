"""Checks for the ESP32 evaluation helpers and the generated C feature tables (no GPU/TF needed)."""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

esp32_export = pytest.importorskip("esp32_export")
campaign = pytest.importorskip("grouped_cv_campaign")


def test_choose_threshold_rules_on_separable_scores():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    for rule in ("f2", "f1", "f0.5", "fpr10", "precision_floor"):
        thr = esp32_export.choose_threshold(y, s, rule)
        assert 0.4 < thr <= 0.6, rule


def test_fpr10_rule_respects_cap_on_selection_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = rng.normal(y * 1.0, 1.0)
    thr = esp32_export.choose_threshold(y, s, "fpr10")
    assert ((s >= thr) & (y == 0)).sum() / (y == 0).sum() <= 0.10


def test_crossfit_never_uses_held_out_labels():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = rng.random(300)
    folds = [(np.r_[0:200], np.r_[200:300]), (np.r_[100:300], np.r_[0:100]),
             (np.r_[np.r_[0:100], np.r_[200:300]], np.r_[100:200])]
    flipped = y.copy()
    flipped[200:300] = 1 - flipped[200:300]
    base = esp32_export.crossfit(y, s, folds[:1], rules=["f1"])
    other = esp32_export.crossfit(flipped, s, folds[:1], rules=["f1"])
    # Threshold comes from rows 0-199 only, so the held-out predictions are identical.
    assert base["f1_tp"] + base["f1_fp"] == other["f1_tp"] + other["f1_fp"]


def test_top_k_decisions_matches_device_rule():
    rid = np.repeat([0, 1], [20, 5])
    scores = np.r_[np.arange(10) / 10, np.zeros(10), np.array([0.9, 0.1, 0.2, 0.3, 0.4])]
    out = esp32_export.top_k_decisions(scores, rid, 2)
    assert out[0] == pytest.approx(((0.9 + 0.8 + 0.7) / 3 + 0.0) / 2)
    assert out[1] == pytest.approx((0.9 + 0.4) / 2)  # 5 windows -> top round(1.5) = 2


def test_input_transforms_hide_high_bands_and_level():
    esp32_train = pytest.importorskip("esp32_train")
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 59, 64)).astype("float32")
    cut = esp32_train.transform(x, max_band=56)
    assert np.array_equal(cut[..., :56], x[..., :56])
    assert np.array_equal(cut[..., 56:], np.repeat(x[..., 55:56], 8, axis=-1))
    assert np.array_equal(esp32_train.transform(x), x)
    gn = esp32_train.transform(x + 5.0, gain_norm=True)
    assert np.allclose(gn, esp32_train.transform(x, gain_norm=True), atol=1e-5)  # absolute level removed
    assert np.allclose(gn.mean(axis=(1, 2)), 0, atol=1e-5)


def test_mac_count_matches_hand_calculation():
    compute = pytest.importorskip("esp32_compute")
    arch = pytest.importorskip("esp32_arch")
    # conv 59x64x9x1x8 + conv 29x32x9x8x16 + conv 14x16x9x16x32 + dense 32x16 + logit 16
    assert compute.macs(arch.CANDIDATES["stream1s_w8"]["spec"], (59, 64)) == 2_373_648


def test_recall_rule_reaches_target_recall_on_selection_data():
    study = pytest.importorskip("threshold_rules_study")
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    s = rng.normal(y * 1.5, 1.0)
    thr = study.rule_threshold(y, s, np.zeros(500), "recall97", rng)
    assert ((s >= thr) & (y == 1)).sum() / (y == 1).sum() >= 0.97


def _c_array(text, name):
    body = re.search(rf"{name}\[[^\]]*\] = \{{(.*?)\}};", text, re.S).group(1)
    return np.array([float(v.rstrip("f")) for v in re.findall(r"[-0-9.e+]+f?", body)])


def test_generated_logmel_tables_match_training_features():
    header = Path("firmware/logmel/logmel_tables.h")
    if not header.exists():
        pytest.skip("run scripts/gen_logmel_tables.py")
    text = header.read_text()
    window = _c_array(text, "LOGMEL_WINDOW")
    assert np.array_equal(window.astype(np.float32), campaign.WINDOW.astype(np.float32))
    starts, lens = _c_array(text, "LOGMEL_BAND_START").astype(int), _c_array(text, "LOGMEL_BAND_LEN").astype(int)
    weights = _c_array(text, "LOGMEL_MEL_W").astype(np.float32)
    dense = np.zeros((64, 257), np.float32)
    offset = 0
    for b in range(64):
        dense[b, starts[b]:starts[b] + lens[b]] = weights[offset:offset + lens[b]]
        offset += lens[b]
    assert np.array_equal(dense, campaign.MEL64.astype(np.float32))
