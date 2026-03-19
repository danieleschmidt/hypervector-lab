"""Tests for HVBenchmark."""
import numpy as np
import pytest
import sys
sys.path.insert(0, "/home/dschmidt/repos/hypervector-lab")

from hdc.space import HypervectorSpace
from hdc.benchmark import HVBenchmark

D = 5000  # fast enough for test runs


def make(hv_type="bipolar"):
    space = HypervectorSpace(dim=D, hv_type=hv_type, seed=42)
    return HVBenchmark(space)


def test_recognition_accuracy_perfect():
    """10 items in a 5000-D space should be perfectly recoverable."""
    bench = make()
    result = bench.recognition_accuracy(n_items=10, n_trials=5)
    assert result["accuracy"] == 1.0, f"accuracy={result['accuracy']}"


def test_noise_tolerance_degrades():
    """Accuracy should decrease as noise increases."""
    bench = make()
    result = bench.noise_tolerance(
        n_items=5, noise_rates=[0.0, 0.1, 0.3, 0.5], n_trials=5
    )
    rates_and_accs = result["accuracy_by_rate"]
    # 0% noise should be 1.0
    assert rates_and_accs[0][1] == 1.0, f"0% noise accuracy: {rates_and_accs[0][1]}"
    # 50% noise should be worse than 0% noise
    assert rates_and_accs[-1][1] < rates_and_accs[0][1]


def test_orthogonality():
    """Random HVs should be approximately orthogonal (mean cosine ≈ 0)."""
    bench = make()
    result = bench.orthogonality(n_pairs=500)
    assert abs(result["mean_cosine"]) < 0.05, (
        f"mean cosine too high: {result['mean_cosine']:.4f}"
    )
    # std should be close to 1/sqrt(D)
    expected_std = 1.0 / np.sqrt(D)
    assert abs(result["std_cosine"] - expected_std) < 0.01, (
        f"std {result['std_cosine']:.4f} vs expected {expected_std:.4f}"
    )


def test_capacity_returns_structure():
    bench = make()
    result = bench.capacity(max_items=10, n_trials=3, accuracy_threshold=0.9)
    assert "capacity" in result
    assert "accuracy_by_k" in result
    assert result["capacity"] >= 1


def test_recognition_accuracy_returns_keys():
    bench = make()
    result = bench.recognition_accuracy(n_items=5, n_trials=2)
    assert "accuracy" in result
    assert "n_items" in result
    assert "dimension" in result
    assert "hv_type" in result
