"""
HVArithmetic — the full suite of HDC operations.

Bundling   (superposition)   : represents a *set* — result is similar to all members
Binding    (outer product)   : represents a *pair* — result is dissimilar to both
Permutation (cyclic shift)   : represents *sequence position* — preserves distance
Similarity : Hamming, cosine, dot-product

Mathematical properties verified by tests:
  - Bundling is commutative and associative
  - Binding is commutative (XOR/multiply) and approximately self-inverse
  - bind(A, bind(A, B)) ≈ B  (involution / self-inverse up to tie-breaking)
  - permute(permute^{-1}(A)) == A
  - Distributivity: bind(A, bundle(B, C)) ≈ bundle(bind(A,B), bind(A,C))
"""

from __future__ import annotations
import numpy as np
from typing import List, Literal, Union, Optional
from .space import HypervectorSpace

Array = np.ndarray


class HVArithmetic:
    """
    Stateless arithmetic over a fixed HypervectorSpace.

    All methods accept raw numpy arrays and return numpy arrays —
    no boxing needed.

    Parameters
    ----------
    space : HypervectorSpace
        The vector space whose type governs which operations are legal.
    """

    def __init__(self, space: HypervectorSpace) -> None:
        self.space = space

    # ------------------------------------------------------------------
    # Bundle (superposition / addition)
    # ------------------------------------------------------------------

    def bundle(self, hvs: List[Array], weights: Optional[List[float]] = None) -> Array:
        """
        Bundle a list of hypervectors into a single representative.

        binary/bipolar : weighted majority vote (threshold at 0)
        complex        : weighted circular mean (normalise to unit modulus)

        With no weights each vector has equal vote.

        Returns a vector in the *same* space (binarised/bipolarised/unit-complex).
        """
        if not hvs:
            raise ValueError("bundle requires at least one vector")
        if len(hvs) == 1:
            return hvs[0].copy()

        n = len(hvs)
        if weights is None:
            w = np.ones(n, dtype=np.float64)
        else:
            if len(weights) != n:
                raise ValueError("len(weights) must equal len(hvs)")
            w = np.asarray(weights, dtype=np.float64)

        ht = self.space.hv_type

        if ht in ("binary", "thermometer"):
            # Weighted sum then threshold at 0.5 * total_weight
            acc = sum(w[i] * hvs[i].astype(np.float64) for i in range(n))
            threshold = 0.5 * w.sum()
            result = (acc > threshold).astype(np.int8)
            # Tie-break (acc == threshold): random using space rng
            ties = acc == threshold
            if ties.any():
                result[ties] = self.space.rng.integers(0, 2, size=int(ties.sum()), dtype=np.int8)
            return result

        elif ht == "bipolar":
            # Weighted sum then sign (tie → random ±1)
            acc = sum(w[i] * hvs[i].astype(np.float64) for i in range(n))
            result = np.sign(acc).astype(np.int8)
            ties = result == 0
            if ties.any():
                result[ties] = self.space.rng.choice([-1, 1], size=int(ties.sum())).astype(np.int8)
            return result

        elif ht == "complex":
            # Weighted circular mean — sum then normalise
            acc = sum(w[i] * hvs[i] for i in range(n))
            modulus = np.abs(acc)
            zero_mask = modulus == 0
            acc[~zero_mask] /= modulus[~zero_mask]
            # For zero-magnitude positions draw a fresh random phase
            n_zero = int(zero_mask.sum())
            if n_zero:
                theta = self.space.rng.uniform(0, 2 * np.pi, size=n_zero)
                acc[zero_mask] = np.exp(1j * theta).astype(np.complex64)
            return acc.astype(np.complex64)

        raise AssertionError("unreachable")

    # ------------------------------------------------------------------
    # Bind (outer product / association)
    # ------------------------------------------------------------------

    def bind(self, a: Array, b: Array) -> Array:
        """
        Bind two hypervectors.

        binary   : XOR  (a ^ b)
        bipolar  : element-wise multiply  (a * b)
        complex  : element-wise complex multiply  (a * conj(b) would give inner;
                   a * b gives outer/association — unbind with bind(result, conj(b)))

        The result is (approximately) orthogonal to both inputs.
        Self-inverse: bind(bind(a, b), b) == a  (XOR/multiply).
        """
        ht = self.space.hv_type
        if ht == "binary":
            return np.bitwise_xor(a, b)
        elif ht in ("bipolar", "thermometer"):
            return (a.astype(np.int8) * b.astype(np.int8)).astype(np.int8)
        elif ht == "complex":
            return (a * b).astype(np.complex64)
        raise AssertionError("unreachable")

    def unbind(self, bound: Array, b: Array) -> Array:
        """
        Inverse of bind.

        binary   : XOR again  (self-inverse)
        bipolar  : multiply again  (self-inverse for ±1)
        complex  : multiply by conjugate of b
        """
        ht = self.space.hv_type
        if ht in ("binary", "bipolar", "thermometer"):
            return self.bind(bound, b)
        elif ht == "complex":
            return (bound * np.conj(b)).astype(np.complex64)
        raise AssertionError("unreachable")

    # ------------------------------------------------------------------
    # Permute (cyclic shift)
    # ------------------------------------------------------------------

    def permute(self, hv: Array, shift: int = 1) -> Array:
        """
        Cyclically shift a hypervector by *shift* positions.
        Positive shift = right; negative = left.
        """
        return np.roll(hv, shift)

    def unpermute(self, hv: Array, shift: int = 1) -> Array:
        """Inverse of permute: shift in the opposite direction."""
        return np.roll(hv, -shift)

    # ------------------------------------------------------------------
    # Similarity metrics
    # ------------------------------------------------------------------

    def cosine(self, a: Array, b: Array) -> float:
        """Cosine similarity ∈ [-1, 1] (complex: real part of normalised dot)."""
        if self.space.hv_type == "complex":
            num = np.real(np.dot(a, np.conj(b)))
            denom = np.sqrt(np.real(np.dot(a, np.conj(a))) * np.real(np.dot(b, np.conj(b))))
        else:
            num = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
            denom = float(np.linalg.norm(a.astype(np.float64)) * np.linalg.norm(b.astype(np.float64)))
        if denom == 0:
            return 0.0
        return float(num / denom)

    def dot(self, a: Array, b: Array) -> float:
        """Raw dot product (complex → real part)."""
        if self.space.hv_type == "complex":
            return float(np.real(np.dot(a, np.conj(b))))
        return float(np.dot(a.astype(np.float64), b.astype(np.float64)))

    def hamming(self, a: Array, b: Array) -> float:
        """
        Normalised Hamming distance ∈ [0, 1].

        binary/bipolar : fraction of positions that differ
        complex        : fraction of positions with angle-distance > π/2
        """
        D = self.space.dim
        ht = self.space.hv_type
        if ht == "binary":
            return float(np.count_nonzero(np.bitwise_xor(a, b))) / D
        elif ht in ("bipolar", "thermometer"):
            return float(np.count_nonzero(a != b)) / D
        elif ht == "complex":
            angle_diff = np.abs(np.angle(a * np.conj(b)))
            return float(np.sum(angle_diff > np.pi / 2)) / D
        raise AssertionError("unreachable")

    def similarity(
        self,
        a: Array,
        b: Array,
        metric: Literal["cosine", "hamming", "dot"] = "cosine",
    ) -> float:
        """Dispatch to the requested similarity metric."""
        if metric == "cosine":
            return self.cosine(a, b)
        elif metric == "hamming":
            return 1.0 - self.hamming(a, b)   # convert distance → similarity
        elif metric == "dot":
            return self.dot(a, b)
        raise ValueError(f"Unknown metric: {metric!r}")
