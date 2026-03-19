"""Tests for HVEncoder."""
import numpy as np
import pytest
import sys
sys.path.insert(0, "/home/dschmidt/repos/hypervector-lab")

from hdc.space import HypervectorSpace
from hdc.arithmetic import HVArithmetic
from hdc.encoder import HVEncoder

D = 5000


def make(hv_type="bipolar", seed=42):
    space = HypervectorSpace(dim=D, hv_type=hv_type, seed=seed)
    enc = HVEncoder(space)
    arith = HVArithmetic(space)
    return space, enc, arith


# -----------------------------------------------------------------------
# Symbol encoding
# -----------------------------------------------------------------------

def test_symbol_deterministic():
    _, enc, _ = make()
    a = enc.symbol("cat")
    b = enc.symbol("cat")
    np.testing.assert_array_equal(a, b)


def test_different_symbols_dissimilar():
    _, enc, arith = make()
    a = enc.symbol("cat")
    b = enc.symbol("dog")
    sim = abs(arith.cosine(a, b))
    assert sim < 0.15


def test_symbol_shape():
    _, enc, _ = make()
    assert enc.symbol("hello").shape == (D,)


# -----------------------------------------------------------------------
# Integer / scalar encoding
# -----------------------------------------------------------------------

def test_integer_adjacent_similar():
    """Adjacent integers should have higher similarity than distant ones."""
    _, enc, arith = make()
    hv0 = enc.integer(0, 0, 100)
    hv1 = enc.integer(1, 0, 100)
    hv50 = enc.integer(50, 0, 100)
    hv100 = enc.integer(100, 0, 100)

    sim_adjacent = arith.cosine(hv0, hv1)
    sim_far = arith.cosine(hv0, hv100)
    assert sim_adjacent > sim_far, (
        f"adjacent sim {sim_adjacent:.3f} not > distant sim {sim_far:.3f}"
    )


def test_integer_same_returns_same():
    _, enc, _ = make()
    a = enc.integer(5, 0, 10)
    b = enc.integer(5, 0, 10)
    np.testing.assert_array_equal(a, b)


def test_scalar_encoding():
    _, enc, arith = make()
    hv0 = enc.scalar(0.0)
    hv1 = enc.scalar(1.0)
    hv_mid = enc.scalar(0.5)
    # mid should be more similar to both extremes than extremes to each other
    sim_0_1 = arith.cosine(hv0, hv1)
    sim_0_mid = arith.cosine(hv0, hv_mid)
    assert sim_0_mid > sim_0_1


def test_circular_encoding_wraps():
    """Circular: 0 and max should be similar (unlike linear)."""
    _, enc, arith = make()
    hv_start = enc.integer(0, 0, 10, circular=True)
    hv_end = enc.integer(10, 0, 10, circular=True)
    hv_mid = enc.integer(5, 0, 10, circular=True)

    sim_start_end = arith.cosine(hv_start, hv_end)
    sim_start_mid = arith.cosine(hv_start, hv_mid)
    # In circular encoding end wraps toward start — should be more similar
    # than the midpoint is to the start.
    assert sim_start_end > sim_start_mid, (
        f"circular start-end sim {sim_start_end:.3f} should be > "
        f"start-mid sim {sim_start_mid:.3f}"
    )


# -----------------------------------------------------------------------
# Sequence encoding
# -----------------------------------------------------------------------

def test_sequence_returns_correct_shape():
    _, enc, _ = make()
    hv = enc.sequence(["a", "b", "c"])
    assert hv.shape == (D,)


def test_sequence_order_matters():
    """Different orderings should produce different HVs."""
    _, enc, arith = make()
    s1 = enc.sequence(["cat", "sat", "mat"])
    s2 = enc.sequence(["mat", "sat", "cat"])
    sim = arith.cosine(s1, s2)
    # Same symbols, different order → should be dissimilar
    assert sim < 0.5, f"sequences too similar despite different order: {sim:.3f}"


def test_sequence_same_order_same_result():
    _, enc, _ = make()
    s1 = enc.sequence(["a", "b", "c"])
    s2 = enc.sequence(["a", "b", "c"])
    np.testing.assert_array_equal(s1, s2)


# -----------------------------------------------------------------------
# N-gram encoding
# -----------------------------------------------------------------------

def test_ngram_shape():
    _, enc, _ = make()
    hv = enc.ngram(["the", "quick", "brown", "fox"], n=2)
    assert hv.shape == (D,)


def test_ngram_similar_sentences():
    """Sentences sharing most n-grams should be more similar."""
    _, enc, arith = make()
    s1 = enc.ngram(["the", "cat", "sat", "on", "mat"], n=2)
    s2 = enc.ngram(["the", "cat", "sat", "on", "floor"], n=2)
    s3 = enc.ngram(["dogs", "run", "fast", "in", "park"], n=2)

    sim_12 = arith.cosine(s1, s2)
    sim_13 = arith.cosine(s1, s3)
    assert sim_12 > sim_13, f"related: {sim_12:.3f}, unrelated: {sim_13:.3f}"


# -----------------------------------------------------------------------
# Image encoding
# -----------------------------------------------------------------------

def test_image_encoding_shape():
    _, enc, _ = make()
    img = np.random.rand(32, 32)
    hv = enc.image(img)
    assert hv.shape == (D,)


def test_similar_images_similar_hvs():
    """Near-identical images should yield similar HVs."""
    _, enc, arith = make()
    img1 = np.ones((16, 16)) * 0.5
    img2 = np.ones((16, 16)) * 0.5 + np.random.rand(16, 16) * 0.05  # tiny noise
    img3 = np.zeros((16, 16))

    hv1 = enc.image(img1)
    hv2 = enc.image(img2)
    hv3 = enc.image(img3)

    sim_12 = arith.cosine(hv1, hv2)
    sim_13 = arith.cosine(hv1, hv3)
    assert sim_12 > sim_13, f"similar images: {sim_12:.3f}, dissimilar: {sim_13:.3f}"
