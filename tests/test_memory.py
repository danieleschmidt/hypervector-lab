"""Tests for AssociativeMemory."""
import numpy as np
import pytest
import sys
sys.path.insert(0, "/home/dschmidt/repos/hypervector-lab")

from hdc.space import HypervectorSpace
from hdc.memory import AssociativeMemory

D = 5000


def make(hv_type="bipolar"):
    space = HypervectorSpace(dim=D, hv_type=hv_type, seed=77)
    mem = AssociativeMemory(space)
    return space, mem


def test_store_and_retrieve_exact():
    space, mem = make()
    items = {f"item_{i}": space.random() for i in range(10)}
    for name, hv in items.items():
        mem.store(name, hv)

    for name, hv in items.items():
        retrieved, _, score = mem.cleanup(hv)
        assert retrieved == name, f"Expected {name}, got {retrieved}"
        assert score > 0.9


def test_cleanup_with_noise():
    """Cleanup should recover the original item despite 20% bit flips."""
    space, mem = make()
    hv = space.random()
    mem.store("target", hv)
    # Add some distractor items
    for i in range(9):
        mem.store(f"noise_{i}", space.random())

    # Flip 20% of bits
    noisy = hv.copy()
    n_flip = D // 5
    idx = space.rng.choice(D, size=n_flip, replace=False)
    noisy[idx] = -noisy[idx]

    retrieved, _, score = mem.cleanup(noisy)
    assert retrieved == "target", f"Cleanup failed: got {retrieved} (score={score:.3f})"


def test_store_overwrite():
    space, mem = make()
    hv1 = space.random()
    hv2 = space.random()
    mem.store("key", hv1)
    mem.store("key", hv2)
    assert len(mem) == 1
    np.testing.assert_array_equal(mem.get("key"), hv2)


def test_contains():
    space, mem = make()
    mem.store("cat", space.random())
    assert "cat" in mem
    assert "dog" not in mem


def test_query_top_k():
    space, mem = make()
    for i in range(20):
        mem.store(f"item_{i}", space.random())
    query = mem.get("item_5")
    results = mem.query(query, top_k=3)
    assert len(results) == 3
    assert results[0][0] == "item_5"


def test_empty_memory_cleanup_raises():
    space = HypervectorSpace(dim=D, hv_type="bipolar", seed=0)
    mem = AssociativeMemory(space)
    with pytest.raises(RuntimeError):
        mem.cleanup(space.random())


def test_clear():
    space, mem = make()
    for i in range(5):
        mem.store(f"x{i}", space.random())
    mem.clear()
    assert len(mem) == 0


# -----------------------------------------------------------------------
# Fractional binding
# -----------------------------------------------------------------------

def test_fractional_bind_alpha_0_is_identity():
    space, mem = make("bipolar")
    hv = space.random()
    result = mem.fractional_bind(hv, 0.0)
    identity = space.identity_bind()
    np.testing.assert_array_equal(result, identity)


def test_fractional_bind_alpha_1_is_original():
    space, mem = make("bipolar")
    hv = space.random()
    result = mem.fractional_bind(hv, 1.0)
    np.testing.assert_array_equal(result, hv)


def test_fractional_bind_complex():
    """For complex HVs, fractional bind should scale phase."""
    space = HypervectorSpace(dim=D, hv_type="complex", seed=11)
    mem = AssociativeMemory(space)
    hv = space.random()

    half = mem.fractional_bind(hv, 0.5)
    # half bound twice should recover original: (e^{iθ/2})^2 = e^{iθ}
    from hdc.arithmetic import HVArithmetic
    arith = HVArithmetic(space)
    recovered = arith.bind(half, half)
    sim = arith.cosine(recovered, hv)
    assert sim > 0.99, f"fractional binding property: {sim:.4f}"
