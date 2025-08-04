"""Benchmarking and performance monitoring for HDC operations."""

from .benchmarks import run_benchmarks, BenchmarkSuite
from .profiler import HDCProfiler, profile_operation

__all__ = ["run_benchmarks", "BenchmarkSuite", "HDCProfiler", "profile_operation"]