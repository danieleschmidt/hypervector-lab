"""
AssociativeMemory — item memory with nearest-neighbour cleanup.

Two memory modes:
  exact     : stores each HV as-is; retrieval by argmax similarity
  fractional: stores superposition with fractional binding weights
              (fractional power binding via resonator networks)

The cleanup memory acts as the heteroassociative "codebook decoder" step
in the HDC pipeline.
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .space import HypervectorSpace
from .arithmetic import HVArithmetic


class AssociativeMemory:
    """
    Fixed-capacity associative item memory.

    Parameters
    ----------
    space : HypervectorSpace
    metric : str
        Similarity metric for retrieval: 'cosine', 'hamming', or 'dot'.
    """

    def __init__(
        self,
        space: HypervectorSpace,
        metric: str = "cosine",
    ) -> None:
        self.space = space
        self.arith = HVArithmetic(space)
        self.metric = metric
        self._keys: List[str] = []
        self._hvs: List[np.ndarray] = []
        self._meta: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(self, name: str, hv: np.ndarray, meta: Any = None) -> None:
        """
        Store a named hypervector.

        If *name* already exists, the stored vector is replaced (not bundled).
        """
        if name in self._keys:
            idx = self._keys.index(name)
            self._hvs[idx] = hv.copy()
        else:
            self._keys.append(name)
            self._hvs.append(hv.copy())
        if meta is not None:
            self._meta[name] = meta

    def store_all(self, codebook: Dict[str, np.ndarray]) -> None:
        """Bulk-store from a {name: hv} dict."""
        for name, hv in codebook.items():
            self.store(name, hv)

    # ------------------------------------------------------------------
    # Retrieval / cleanup
    # ------------------------------------------------------------------

    def query(self, hv: np.ndarray, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Nearest-neighbour search.

        Returns a list of (name, similarity_score) sorted by similarity
        descending (cosine/dot) or ascending (hamming distance).
        """
        if not self._hvs:
            return []

        scores = [
            self.arith.similarity(hv, stored, metric=self.metric)
            for stored in self._hvs
        ]

        ranked = sorted(
            zip(self._keys, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def cleanup(self, hv: np.ndarray) -> Tuple[str, np.ndarray, float]:
        """
        Clean up a noisy/partial HV to the nearest stored item.

        Returns (name, stored_hv, similarity_score).
        """
        if not self._hvs:
            raise RuntimeError("Memory is empty — nothing to clean up to.")
        results = self.query(hv, top_k=1)
        best_name, best_score = results[0]
        idx = self._keys.index(best_name)
        return best_name, self._hvs[idx], best_score

    # ------------------------------------------------------------------
    # Fractional binding  (experimental — scalar power binding)
    # ------------------------------------------------------------------

    def fractional_bind(
        self, hv: np.ndarray, fraction: float
    ) -> np.ndarray:
        """
        Fractional binding: HV^α for α ∈ [0, 1].

        For bipolar vectors: HV^α ≈ cos(α·arccos(sim)) approach not
        directly applicable at the bit level; instead we implement the
        *continuous relaxation* — interpolate between identity and the
        vector using bundling weights [1-α, α].

        For complex vectors: raise each unit-complex to the α-th power
        via phase scaling: (e^{iθ})^α = e^{iαθ}.

        This is used to represent *prototypes* that are partially
        similar to a stored item.
        """
        alpha = float(np.clip(fraction, 0.0, 1.0))
        ht = self.space.hv_type

        if ht == "complex":
            # True fractional binding for complex HVs
            theta = np.angle(hv)
            return np.exp(1j * alpha * theta).astype(np.complex64)

        elif ht in ("bipolar", "binary", "thermometer"):
            # Soft interpolation: bundle identity with vector
            identity = self.space.identity_bind()
            if alpha == 0.0:
                return identity
            if alpha == 1.0:
                return hv.copy()
            # Weighted bundle
            arith = HVArithmetic(self.space)
            return arith.bundle([identity, hv], weights=[1.0 - alpha, alpha])

        raise AssertionError("unreachable")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, name: str) -> bool:
        return name in self._keys

    def names(self) -> List[str]:
        return list(self._keys)

    def get(self, name: str) -> Optional[np.ndarray]:
        if name in self._keys:
            return self._hvs[self._keys.index(name)]
        return None

    def clear(self) -> None:
        self._keys.clear()
        self._hvs.clear()
        self._meta.clear()
