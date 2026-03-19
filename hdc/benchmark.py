"""
HVBenchmark — empirical benchmarks for hyperdimensional computing.

Measures:
  capacity        : maximum items storable with >threshold recognition accuracy
  noise_tolerance : accuracy as a function of bit-flip rate
  recognition     : per-item recognition accuracy over many random trials
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple

from .space import HypervectorSpace
from .arithmetic import HVArithmetic
from .memory import AssociativeMemory


class HVBenchmark:
    """
    Empirical benchmark suite for a given HypervectorSpace.

    Parameters
    ----------
    space : HypervectorSpace
    metric : str
        Similarity metric used for retrieval ('cosine', 'hamming', 'dot').
    """

    def __init__(self, space: HypervectorSpace, metric: str = "cosine") -> None:
        self.space = space
        self.arith = HVArithmetic(space)
        self.metric = metric

    # ------------------------------------------------------------------
    # Capacity test
    # ------------------------------------------------------------------

    def capacity(
        self,
        max_items: int = 200,
        n_trials: int = 10,
        accuracy_threshold: float = 0.95,
    ) -> Dict:
        """
        Measure how many random items can be stored in memory and still be
        correctly retrieved.

        Procedure:
          For k = 1, 2, …, max_items:
            1. Store k random HVs in AssociativeMemory.
            2. Query each stored HV (identity probe) — count correct retrievals.
            3. Average over n_trials.
          Report the largest k where accuracy ≥ accuracy_threshold.

        Returns
        -------
        dict with keys:
          'capacity'      : largest k meeting threshold
          'accuracy_by_k' : list of (k, accuracy) pairs
        """
        accuracy_by_k: List[Tuple[int, float]] = []
        capacity_k = 0

        for k in range(1, max_items + 1):
            correct_total = 0
            for _ in range(n_trials):
                mem = AssociativeMemory(self.space, metric=self.metric)
                hvs = {f"item_{j}": self.space.random() for j in range(k)}
                mem.store_all(hvs)

                correct = 0
                for name, hv in hvs.items():
                    retrieved_name, _, _ = mem.cleanup(hv)
                    if retrieved_name == name:
                        correct += 1
                correct_total += correct / k

            acc = correct_total / n_trials
            accuracy_by_k.append((k, acc))

            if acc >= accuracy_threshold:
                capacity_k = k
            else:
                # early exit once accuracy drops below threshold
                if k > 5:  # give some warm-up
                    break

        return {
            "capacity": capacity_k,
            "accuracy_by_k": accuracy_by_k,
            "dimension": self.space.dim,
            "hv_type": self.space.hv_type,
        }

    # ------------------------------------------------------------------
    # Noise tolerance test
    # ------------------------------------------------------------------

    def noise_tolerance(
        self,
        n_items: int = 10,
        noise_rates: Optional[List[float]] = None,
        n_trials: int = 20,
    ) -> Dict:
        """
        Measure recognition accuracy as bit-flip (or phase) noise is added.

        For each noise rate p:
          1. Store n_items random HVs in memory.
          2. For each item, flip p% of bits (or rotate phases by uniform noise).
          3. Query with the noisy vector; check if cleanup returns the correct item.
          4. Report accuracy.

        Returns
        -------
        dict with keys:
          'noise_rates'      : list of tested rates
          'accuracy_by_rate' : list of (rate, accuracy) pairs
        """
        if noise_rates is None:
            noise_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

        results = []
        for p in noise_rates:
            correct_total = 0
            for _ in range(n_trials):
                mem = AssociativeMemory(self.space, metric=self.metric)
                hvs = {f"item_{j}": self.space.random() for j in range(n_items)}
                mem.store_all(hvs)

                correct = 0
                for name, hv in hvs.items():
                    noisy = self._add_noise(hv, p)
                    retrieved_name, _, _ = mem.cleanup(noisy)
                    if retrieved_name == name:
                        correct += 1
                correct_total += correct / n_items

            accuracy = correct_total / n_trials
            results.append((p, accuracy))

        return {
            "n_items": n_items,
            "noise_rates": noise_rates,
            "accuracy_by_rate": results,
            "dimension": self.space.dim,
            "hv_type": self.space.hv_type,
        }

    def _add_noise(self, hv: np.ndarray, rate: float) -> np.ndarray:
        """Add noise to a hypervector at the given flip rate."""
        D = self.space.dim
        n_flip = int(rate * D)
        noisy = hv.copy()
        indices = self.space.rng.choice(D, size=n_flip, replace=False)

        ht = self.space.hv_type
        if ht == "binary":
            noisy[indices] = 1 - noisy[indices]
        elif ht in ("bipolar", "thermometer"):
            noisy[indices] = -noisy[indices]
        elif ht == "complex":
            noise_theta = self.space.rng.uniform(-np.pi * rate, np.pi * rate, size=n_flip)
            noisy[indices] = (noisy[indices] * np.exp(1j * noise_theta)).astype(np.complex64)
        return noisy

    # ------------------------------------------------------------------
    # Recognition accuracy test
    # ------------------------------------------------------------------

    def recognition_accuracy(
        self,
        n_items: int = 50,
        n_trials: int = 10,
    ) -> Dict:
        """
        Measure per-item recognition accuracy (exact retrieval from clean probe).

        Returns
        -------
        dict with keys:
          'accuracy'   : overall mean accuracy
          'n_items'    : items stored
          'n_trials'   : number of random trials
          'dimension'  : D
        """
        correct_total = 0
        for _ in range(n_trials):
            mem = AssociativeMemory(self.space, metric=self.metric)
            hvs = {f"item_{j}": self.space.random() for j in range(n_items)}
            mem.store_all(hvs)

            for name, hv in hvs.items():
                retrieved_name, _, _ = mem.cleanup(hv)
                if retrieved_name == name:
                    correct_total += 1

        accuracy = correct_total / (n_items * n_trials)
        return {
            "accuracy": accuracy,
            "n_items": n_items,
            "n_trials": n_trials,
            "dimension": self.space.dim,
            "hv_type": self.space.hv_type,
        }

    # ------------------------------------------------------------------
    # Orthogonality test
    # ------------------------------------------------------------------

    def orthogonality(self, n_pairs: int = 1000) -> Dict:
        """
        Verify that random HVs are (approximately) orthogonal.
        Expected cosine similarity ≈ 0, std ≈ 1/sqrt(D).

        Returns mean and std of pairwise cosine similarities.
        """
        sims = []
        for _ in range(n_pairs):
            a = self.space.random()
            b = self.space.random()
            sims.append(self.arith.cosine(a, b))

        sims = np.array(sims)
        return {
            "mean_cosine": float(sims.mean()),
            "std_cosine": float(sims.std()),
            "expected_std": float(1.0 / np.sqrt(self.space.dim)),
            "n_pairs": n_pairs,
        }
