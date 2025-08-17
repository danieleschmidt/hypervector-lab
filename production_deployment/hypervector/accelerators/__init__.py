"""Hardware acceleration modules for HDC operations."""

from .cpu_optimized import CPUAccelerator
from .batch_processor import BatchProcessor
from .memory_manager import MemoryManager, MemoryPool, SmartCache
from .performance_optimizer import (
    PerformanceProfiler, 
    BatchProcessor as OptimizedBatchProcessor,
    ParallelProcessor,
    AdaptiveOptimizer,
    create_optimized_hdc_system
)

__all__ = [
    "CPUAccelerator", 
    "BatchProcessor", 
    "MemoryManager",
    "MemoryPool",
    "SmartCache",
    "PerformanceProfiler",
    "OptimizedBatchProcessor", 
    "ParallelProcessor",
    "AdaptiveOptimizer",
    "create_optimized_hdc_system"
]