"""Research-oriented HDC algorithms and experimental features."""

from .novel_algorithms import *
from .experimental_encoders import *
from .comparative_studies import *

__all__ = [
    # Novel algorithms
    "HierarchicalHDC",
    "AdaptiveBindingOperator",
    "QuantumInspiredHDC",
    "TemporalHDC",
    
    # Experimental encoders
    "GraphEncoder",
    "SequenceEncoder", 
    "AttentionBasedEncoder",
    
    # Comparative studies
    "ComparisonFramework",
    "BenchmarkComparator",
    "StatisticalAnalyzer"
]