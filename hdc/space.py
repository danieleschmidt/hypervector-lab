"""
HypervectorSpace — the algebraic context for hyperdimensional computing.

Four vector types supported:
  binary       — {0, 1}^D, operations via XOR / majority vote
  bipolar      — {-1, +1}^D, operations via multiplication / sign(sum)
  complex      — unit-complex e^{i*theta}, bind via element-wise product
  thermometer  — level-coded real scalars in [0,1] range
"""

from __future__ import annotations
import numpy as np
from typing import Literal, Optional

HVType = Literal["binary", "bipolar", "complex", "thermometer"]


class HypervectorSpace:
    """
    Defines a D-dimensional hypervector space with a fixed RNG seed for
    reproducible basis generation.

    Parameters
    ----------
    dim : int
        Dimensionality D (typically 1 000 – 10 000).
    hv_type : HVType
        One of "binary", "bipolar", "complex", "thermometer".
    seed : int | None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        dim: int = 10_000,
        hv_type: HVType = "bipolar",
        seed: Optional[int] = None,
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be ≥ 1, got {dim}")
        if hv_type not in ("binary", "bipolar", "complex", "thermometer"):
            raise ValueError(f"Unknown hv_type: {hv_type!r}")
        self.dim = dim
        self.hv_type = hv_type
        self.rng = np.random.default_rng(seed)
        self._basis: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Random basis generation
    # ------------------------------------------------------------------

    def random(self) -> np.ndarray:
        """Sample a single random hypervector from this space."""
        return self._sample()

    def basis(self, name: str) -> np.ndarray:
        """
        Return (or lazily generate) a named basis vector.
        Useful for building symbol codebooks.
        """
        if name not in self._basis:
            self._basis[name] = self._sample()
        return self._basis[name]

    def codebook(self, symbols) -> dict[str, np.ndarray]:
        """Generate a codebook mapping each symbol → random HV."""
        return {s: self.basis(str(s)) for s in symbols}

    # ------------------------------------------------------------------
    # Internal sampling
    # ------------------------------------------------------------------

    def _sample(self) -> np.ndarray:
        D = self.dim
        if self.hv_type == "binary":
            return self.rng.integers(0, 2, size=D, dtype=np.int8)
        elif self.hv_type == "bipolar":
            raw = self.rng.integers(0, 2, size=D, dtype=np.int8)
            return np.where(raw == 0, np.int8(-1), np.int8(1))
        elif self.hv_type == "complex":
            theta = self.rng.uniform(0, 2 * np.pi, size=D)
            return np.exp(1j * theta).astype(np.complex64)
        elif self.hv_type == "thermometer":
            # Thermometer level vectors are computed via HVEncoder;
            # raw random for the thermometer space is still binary-coded.
            return self.rng.integers(0, 2, size=D, dtype=np.int8)
        raise AssertionError("unreachable")

    # ------------------------------------------------------------------
    # Zero / identity elements
    # ------------------------------------------------------------------

    def zeros(self) -> np.ndarray:
        """Additive identity (bundle accumulator start)."""
        if self.hv_type == "complex":
            return np.zeros(self.dim, dtype=np.complex64)
        return np.zeros(self.dim, dtype=np.int32)

    def identity_bind(self) -> np.ndarray:
        """Multiplicative identity for binding (all-ones / all-zeros XOR)."""
        if self.hv_type == "binary":
            return np.zeros(self.dim, dtype=np.int8)   # XOR identity
        elif self.hv_type == "bipolar":
            return np.ones(self.dim, dtype=np.int8)    # multiply identity
        elif self.hv_type == "complex":
            return np.ones(self.dim, dtype=np.complex64)  # complex multiply identity
        elif self.hv_type == "thermometer":
            return np.zeros(self.dim, dtype=np.int8)
        raise AssertionError("unreachable")

    def __repr__(self) -> str:  # pragma: no cover
        return f"HypervectorSpace(dim={self.dim}, hv_type={self.hv_type!r})"
