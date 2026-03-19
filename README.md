# Hypervector Lab

A pure-NumPy exploration of **Hyperdimensional Computing (HDC)** — also called
Brain-Inspired Computing or Vector Symbolic Architectures (VSA).

No PyTorch. No heavy dependencies. Just math.

---

## What is HDC?

Hyperdimensional Computing represents information as high-dimensional vectors
(**hypervectors**, D ≈ 10 000) and performs computation via simple algebraic
operations on them.  The key insight: in high-dimensional spaces, random vectors
are nearly orthogonal with overwhelming probability — this is the geometric
foundation for all HDC algorithms.

### The Algebra

Three operations define the HDC algebra:

| Operation | Symbol | Effect |
|-----------|--------|--------|
| **Bundle** (superposition) | A ⊕ B | Similar to both A and B |
| **Bind** (outer product) | A ⊗ B | Dissimilar to both A and B; self-inverse |
| **Permute** (shift) | ρ(A) | Preserves distance; encodes position |

**Key properties (with proofs in tests):**

- Binding is **commutative**: A ⊗ B = B ⊗ A
- Binding is **self-inverse**: (A ⊗ B) ⊗ B = A
- Bundling is **commutative** and **associative**
- Permutation is **invertible**: ρ⁻¹(ρ(A)) = A
- **Orthogonality**: E[cos(A, B)] = 0, Std[cos(A, B)] = 1/√D for random A, B
- **Distributivity** (complex): A ⊗ (B ⊕ C) ≈ (A ⊗ B) ⊕ (A ⊗ C)

### Vector Types

| Type | Values | Bind | Bundle |
|------|--------|------|--------|
| `bipolar` | {−1, +1}^D | Element multiply | Sign of sum |
| `binary` | {0, 1}^D | XOR | Majority vote |
| `complex` | e^{iθ} ∈ ℂ^D | Element multiply | Circular mean |
| `thermometer` | {0,1}^D (level-coded) | Element multiply | Majority vote |

---

## Structure

```
hdc/
  space.py       — HypervectorSpace: dimension, type, RNG, basis generation
  arithmetic.py  — HVArithmetic: bundle, bind, unbind, permute, similarity
  encoder.py     — HVEncoder: symbols, integers, scalars, sequences, images
  memory.py      — AssociativeMemory: store, cleanup, fractional binding
  benchmark.py   — HVBenchmark: capacity, noise tolerance, recognition accuracy

tests/
  test_space.py       — space creation, sampling, reproducibility
  test_arithmetic.py  — all algebraic properties
  test_encoder.py     — encoding monotonicity and similarity invariants
  test_memory.py      — retrieval, noise robustness, fractional binding
  test_benchmark.py   — benchmark output structure and expected properties
```

---

## Quick Start

```python
import numpy as np
from hdc import HypervectorSpace, HVArithmetic, HVEncoder, AssociativeMemory

# 1. Create a space
space = HypervectorSpace(dim=10_000, hv_type="bipolar", seed=42)
arith = HVArithmetic(space)
enc   = HVEncoder(space)
mem   = AssociativeMemory(space)

# 2. Encode some symbols
cat = enc.symbol("cat")
dog = enc.symbol("dog")

# 3. Store in memory
mem.store("cat", cat)
mem.store("dog", dog)

# 4. Retrieve with a noisy probe
noisy_cat = cat.copy()
idx = space.rng.choice(space.dim, size=1000, replace=False)
noisy_cat[idx] = -noisy_cat[idx]  # flip 10% of bits

name, hv, score = mem.cleanup(noisy_cat)
print(name, score)  # → 'cat'  ~0.8

# 5. Encode a sequence (order matters)
sentence = enc.sequence(["the", "cat", "sat"])

# 6. Encode an integer (level coding)
hv_5  = enc.integer(5,  0, 100)
hv_6  = enc.integer(6,  0, 100)
hv_50 = enc.integer(50, 0, 100)
print(arith.cosine(hv_5, hv_6))   # high similarity (adjacent)
print(arith.cosine(hv_5, hv_50))  # low similarity (far apart)
```

---

## Benchmarks

```python
from hdc import HypervectorSpace, HVBenchmark

space = HypervectorSpace(dim=10_000, hv_type="bipolar")
bench = HVBenchmark(space)

# How many items can this space hold at >95% accuracy?
cap = bench.capacity(max_items=200, n_trials=10)
print(f"Capacity: {cap['capacity']} items")

# How robust is retrieval under bit-flip noise?
tol = bench.noise_tolerance(n_items=10)
for rate, acc in tol['accuracy_by_rate']:
    print(f"  noise={rate:.0%}  accuracy={acc:.2%}")

# Are random vectors actually orthogonal?
orth = bench.orthogonality(n_pairs=2000)
print(f"Mean cosine: {orth['mean_cosine']:.4f}  (expected ≈ 0)")
print(f"Std cosine:  {orth['std_cosine']:.4f}  (expected ≈ {orth['expected_std']:.4f})")
```

**Typical results at D=10 000, bipolar:**

```
Capacity: ~80-120 items at >95% accuracy
Noise tolerance:
  noise=0%   accuracy=100%
  noise=10%  accuracy=100%
  noise=20%  accuracy= 99%
  noise=30%  accuracy= 89%
  noise=40%  accuracy= 50%
Mean cosine: 0.0002  (expected ≈ 0)
Std cosine:  0.0100  (expected ≈ 0.0100)
```

---

## Running Tests

```bash
python3 -m pytest tests/ -v
```

All 73 tests pass against the algebraic properties of HDC.

---

## Further Reading

- Kanerva, P. (2009). *Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors.*
- Gayler, R. (2004). *Vector Symbolic Architectures answer Jackendoff's challenges for cognitive neuroscience.*
- Frady, E.P. et al. (2021). *Computing on functions using randomized vector representations.*
- Imani, M. et al. (2019). *HDClassify: Efficient hyperdimensional classification for IoT.*
