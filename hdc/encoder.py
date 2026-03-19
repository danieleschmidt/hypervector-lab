"""
HVEncoder — maps real-world data types to hypervectors.

Encoding strategies:
  symbols     : random lookup (identity map over a codebook)
  integers    : level coding — D/2 bit-flip sequence from 0→max
  circular    : circular level coding — smooth wrap-around
  sequence    : positional binding — each element bound to permuted position key
  ngram       : n-gram position binding (for text)
  image       : regional spatial encoding — image split into tiles
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .space import HypervectorSpace
from .arithmetic import HVArithmetic


class HVEncoder:
    """
    Encodes various data types into hypervectors within the given space.

    Parameters
    ----------
    space : HypervectorSpace
    """

    def __init__(self, space: HypervectorSpace) -> None:
        self.space = space
        self.arith = HVArithmetic(space)
        self._level_cache: Dict[str, List[np.ndarray]] = {}
        self._symbol_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Symbol encoding
    # ------------------------------------------------------------------

    def symbol(self, sym: Any) -> np.ndarray:
        """
        Encode a discrete symbol as a random hypervector.
        Repeated calls with the same symbol return the *same* vector.
        """
        key = str(sym)
        if key not in self._symbol_cache:
            self._symbol_cache[key] = self.space.random()
        return self._symbol_cache[key]

    def codebook(self, symbols) -> Dict[str, np.ndarray]:
        """Build and return a symbol → HV codebook."""
        return {str(s): self.symbol(s) for s in symbols}

    # ------------------------------------------------------------------
    # Integer / scalar encoding
    # ------------------------------------------------------------------

    def integer(
        self,
        value: int,
        min_val: int = 0,
        max_val: int = 100,
        circular: bool = False,
    ) -> np.ndarray:
        """
        Encode an integer in [min_val, max_val] as a level hypervector.

        Level coding: generate max_val - min_val + 1 base vectors where
        consecutive integers share D*(1 - 1/(max-min)) bits, interpolating
        between two poles L0 and L1.

        circular=True: wraps around (useful for angles, days-of-week, etc.)
        """
        n_levels = max_val - min_val + 1
        cache_key = f"level_{min_val}_{max_val}_{circular}"

        if cache_key not in self._level_cache:
            self._level_cache[cache_key] = self._build_level_vectors(
                n_levels, circular
            )

        idx = int(np.clip(value - min_val, 0, n_levels - 1))
        return self._level_cache[cache_key][idx]

    def scalar(
        self,
        value: float,
        min_val: float = 0.0,
        max_val: float = 1.0,
        n_levels: int = 100,
        circular: bool = False,
    ) -> np.ndarray:
        """Encode a real scalar by quantising to n_levels then level-coding."""
        clipped = float(np.clip(value, min_val, max_val))
        idx = int(round((clipped - min_val) / (max_val - min_val) * (n_levels - 1)))
        return self.integer(idx, 0, n_levels - 1, circular=circular)

    def _build_level_vectors(
        self, n_levels: int, circular: bool
    ) -> List[np.ndarray]:
        """
        Build n_levels level vectors.

        Linear: start from L0, flip D/(n_levels-1) bits at each step → L1.
        Circular: same but the last level wraps back to being close to L0.
        """
        D = self.space.dim
        L0 = self.space.random()
        levels = [L0.copy()]

        if n_levels == 1:
            return levels

        if not circular:
            # Linear gradient: flip roughly D/(n_levels-1) new bits each step
            n_flips = D // (n_levels - 1)
            remaining = list(range(D))
            self.space.rng.shuffle(remaining)
            flip_order = remaining
            current = L0.copy()
            flipped = 0
            for _ in range(1, n_levels):
                flip_count = min(n_flips, D - flipped)
                indices = flip_order[flipped: flipped + flip_count]
                flipped += flip_count
                current = current.copy()
                if self.space.hv_type == "binary":
                    current[indices] = 1 - current[indices]
                elif self.space.hv_type in ("bipolar", "thermometer"):
                    current[indices] = -current[indices]
                elif self.space.hv_type == "complex":
                    current[indices] = np.exp(
                        1j * self.space.rng.uniform(0, 2 * np.pi, len(indices))
                    ).astype(np.complex64)
                levels.append(current.copy())
        else:
            # Circular: levels arranged on a ring — opposite pole is most
            # dissimilar, then it wraps back toward L0.
            n_half = n_levels // 2
            linear_half = self._build_level_vectors(n_half + 1, circular=False)
            # Second half mirrors back
            for i in range(1, n_levels - n_half):
                levels.append(linear_half[n_half - i])
            levels = linear_half[:n_half + 1] + [linear_half[n_half - i] for i in range(1, n_levels - n_half)]
            levels = levels[:n_levels]

        return levels

    # ------------------------------------------------------------------
    # Sequence encoding
    # ------------------------------------------------------------------

    def sequence(self, items: Sequence[Any], item_hvs: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Encode an ordered sequence using positional binding.

        Each item is bound to its permuted position key:
            S = bundle( bind(item_0, perm^0(P)), bind(item_1, perm^1(P)), ... )

        where P is a position prototype vector.

        Parameters
        ----------
        items : sequence of symbols (or use item_hvs for pre-encoded HVs)
        item_hvs : pre-encoded item hypervectors (overrides items if provided)
        """
        if item_hvs is None:
            item_hvs = [self.symbol(x) for x in items]

        bound = []
        for i, hv in enumerate(item_hvs):
            pos_hv = self.arith.permute(hv, shift=i)
            bound.append(pos_hv)

        return self.arith.bundle(bound)

    def ngram(self, tokens: Sequence[Any], n: int = 3) -> np.ndarray:
        """
        Encode text/token sequence using n-gram binding.
        Each n-gram is a bound product of n consecutive token HVs.
        The sequence is the bundle of all n-gram HVs.
        """
        if len(tokens) < n:
            return self.sequence(tokens)

        ngram_hvs = []
        for i in range(len(tokens) - n + 1):
            gram = tokens[i: i + n]
            # bind all elements in this gram
            result = self.symbol(gram[0])
            for j, tok in enumerate(gram[1:], start=1):
                result = self.arith.bind(result, self.arith.permute(self.symbol(tok), shift=j))
            ngram_hvs.append(result)

        return self.arith.bundle(ngram_hvs)

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    def image(
        self,
        pixels: np.ndarray,
        n_tiles_x: int = 4,
        n_tiles_y: int = 4,
    ) -> np.ndarray:
        """
        Encode a 2-D image (H×W or H×W×C) using regional spatial encoding.

        The image is divided into n_tiles_x × n_tiles_y tiles.
        Each tile mean-intensity is scalar-encoded and bound to a
        spatial-position vector.  All tile HVs are bundled.

        Parameters
        ----------
        pixels : np.ndarray, shape (H, W) or (H, W, C), values in [0, 1]
        """
        if pixels.ndim == 3:
            gray = pixels.mean(axis=-1)
        elif pixels.ndim == 2:
            gray = pixels.astype(np.float64)
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {pixels.shape}")

        H, W = gray.shape
        tile_hvs = []
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                y0, y1 = ty * H // n_tiles_y, (ty + 1) * H // n_tiles_y
                x0, x1 = tx * W // n_tiles_x, (tx + 1) * W // n_tiles_x
                tile = gray[y0:y1, x0:x1]
                intensity = float(tile.mean())

                # spatial position key
                pos_hv = self.symbol(f"tile_{ty}_{tx}")
                # intensity-coded value
                val_hv = self.scalar(intensity, 0.0, 1.0)
                # bind position to value
                tile_hvs.append(self.arith.bind(pos_hv, val_hv))

        return self.arith.bundle(tile_hvs)
