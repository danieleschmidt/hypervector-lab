"""
Hypervector Lab — Hyperdimensional Computing in pure NumPy.

Core abstractions:
  HypervectorSpace   — defines the vector algebra (dimension, type, basis generation)
  HVArithmetic       — bundle, bind, permute, similarity metrics
  HVEncoder          — maps data types to hypervectors
  AssociativeMemory  — item memory with nearest-neighbour cleanup
  HVBenchmark        — capacity, noise tolerance, recognition accuracy
"""

from .space import HypervectorSpace
from .arithmetic import HVArithmetic
from .encoder import HVEncoder
from .memory import AssociativeMemory
from .benchmark import HVBenchmark

__all__ = [
    "HypervectorSpace",
    "HVArithmetic",
    "HVEncoder",
    "AssociativeMemory",
    "HVBenchmark",
]
