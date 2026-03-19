"""Tests for HypervectorSpace."""
import numpy as np
import pytest
import sys
sys.path.insert(0, "/home/dschmidt/repos/hypervector-lab")

from hdc.space import HypervectorSpace


@pytest.mark.parametrize("hv_type", ["binary", "bipolar", "complex", "thermometer"])
def test_random_shape(hv_type):
    space = HypervectorSpace(dim=1000, hv_type=hv_type, seed=42)
    hv = space.random()
    assert hv.shape == (1000,)


@pytest.mark.parametrize("hv_type", ["binary", "bipolar"])
def test_binary_bipolar_values(hv_type):
    space = HypervectorSpace(dim=5000, hv_type=hv_type, seed=0)
    hv = space.random()
    if hv_type == "binary":
        assert set(np.unique(hv)).issubset({0, 1})
    else:
        assert set(np.unique(hv)).issubset({-1, 1})


def test_complex_unit_modulus():
    space = HypervectorSpace(dim=2000, hv_type="complex", seed=1)
    hv = space.random()
    np.testing.assert_allclose(np.abs(hv), 1.0, atol=1e-5)


def test_basis_deterministic():
    space = HypervectorSpace(dim=1000, hv_type="bipolar", seed=99)
    a = space.basis("word")
    b = space.basis("word")
    np.testing.assert_array_equal(a, b)


def test_basis_different_keys():
    space = HypervectorSpace(dim=1000, hv_type="bipolar", seed=7)
    a = space.basis("cat")
    b = space.basis("dog")
    # Should not be identical (astronomically unlikely)
    assert not np.array_equal(a, b)


def test_seeded_reproducibility():
    space1 = HypervectorSpace(dim=500, hv_type="bipolar", seed=42)
    space2 = HypervectorSpace(dim=500, hv_type="bipolar", seed=42)
    np.testing.assert_array_equal(space1.random(), space2.random())


def test_identity_bind_shape():
    for hv_type in ["binary", "bipolar", "complex", "thermometer"]:
        space = HypervectorSpace(dim=1000, hv_type=hv_type, seed=0)
        assert space.identity_bind().shape == (1000,)


def test_invalid_type():
    with pytest.raises(ValueError):
        HypervectorSpace(hv_type="quaternion")


def test_invalid_dim():
    with pytest.raises(ValueError):
        HypervectorSpace(dim=0)
