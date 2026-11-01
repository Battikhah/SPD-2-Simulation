import numpy as np

from scripts.review_audio_pairs import best_waveform_match


def test_waveform_match_finds_gain_and_offset_with_partial_overlap():
    rng = np.random.default_rng(4)
    original = rng.normal(size=400)
    excerpt = 2.4 * original[80:320] + 0.3
    score, overlap, lag = best_waveform_match(original, excerpt, min_overlap=160)
    assert score > 0.999999
    assert overlap == 240
    assert lag == 80


def test_waveform_match_does_not_treat_silence_as_evidence():
    score, overlap, lag = best_waveform_match(np.zeros(200), np.zeros(200), min_overlap=100)
    assert score is None and overlap is None and lag is None
