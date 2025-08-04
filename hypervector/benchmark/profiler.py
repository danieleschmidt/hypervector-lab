"""Profiling utilities for HDC operations."""

import time
import functools
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
import torch

from ..utils.logging import get_logger, MetricsLogger

logger = get_logger(__name__)


class HDCProfiler:
    """Profiler for HDC operations with detailed metrics."""
    
    def __init__(self, name: str = "HDCProfiler"):
        self.name = name
        self.metrics = MetricsLogger(name)
        self.active_timers = {}
        self.memory_tracking = torch.cuda.is_available()
    
    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling operations.
        
        Args:
            operation_name: Name of the operation being profiled
        """
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage() if self.memory_tracking else None
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Record timing
            self.metrics.record_time(operation_name, duration)
            
            # Record memory if available
            if self.memory_tracking and start_memory is not None:
                end_memory = self._get_memory_usage()
                memory_diff = end_memory - start_memory
                self.metrics.record_time(f"{operation_name}_memory_mb", memory_diff)
            
            logger.debug(f"Profiled {operation_name}: {duration*1000:.2f}ms")
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)
        return 0.0
    
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self.active_timers[name] = time.perf_counter()
    
    def stop_timer(self, name: str) -> Optional[float]:
        """Stop a named timer and return duration."""
        if name in self.active_timers:
            duration = time.perf_counter() - self.active_timers[name]
            del self.active_timers[name]
            self.metrics.record_time(name, duration)
            return duration
        return None
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        self.metrics.increment_counter(name, value)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        summary = {
            'counters': self.metrics.counters.copy(),
            'timers': {}
        }
        
        for name, times in self.metrics.timers.items():
            summary['timers'][name] = {
                'count': len(times),
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        
        return summary
    
    def print_summary(self) -> None:
        """Print profiling summary."""
        self.metrics.log_summary()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.reset()
        self.active_timers.clear()


# Global profiler instance
_global_profiler = HDCProfiler("Global")


def get_profiler() -> HDCProfiler:
    """Get global profiler instance."""
    return _global_profiler


def profile_operation(operation_name: str):
    """Decorator for profiling function calls.
    
    Args:
        operation_name: Name of the operation for profiling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with _global_profiler.profile(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def profile_context(operation_name: str, profiler: Optional[HDCProfiler] = None):
    """Context manager for profiling with optional custom profiler.
    
    Args:
        operation_name: Name of the operation
        profiler: Optional custom profiler (uses global if None)
    """
    if profiler is None:
        profiler = _global_profiler
    
    with profiler.profile(operation_name):
        yield profiler


class MemoryProfiler:
    """Memory usage profiler for HDC operations."""
    
    def __init__(self):
        self.snapshots = []
        self.peak_memory = 0.0
    
    def take_snapshot(self, name: str) -> float:
        """Take a memory snapshot.
        
        Args:
            name: Name/description of the snapshot
            
        Returns:
            Current memory usage in MB
        """
        current_memory = self._get_memory_usage()
        self.snapshots.append({
            'name': name,
            'memory_mb': current_memory,
            'timestamp': time.time()
        })
        
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
        
        return current_memory
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024**2)
            else:
                import psutil
                return psutil.Process().memory_info().rss / (1024**2)
        except ImportError:
            return 0.0
    
    def get_memory_diff(self, start_snapshot: str, end_snapshot: str) -> Optional[float]:
        """Get memory difference between two snapshots.
        
        Args:
            start_snapshot: Name of start snapshot
            end_snapshot: Name of end snapshot
            
        Returns:
            Memory difference in MB (None if snapshots not found)
        """
        start_mem = None
        end_mem = None
        
        for snapshot in self.snapshots:
            if snapshot['name'] == start_snapshot:
                start_mem = snapshot['memory_mb']
            elif snapshot['name'] == end_snapshot:
                end_mem = snapshot['memory_mb']
        
        if start_mem is not None and end_mem is not None:
            return end_mem - start_mem
        
        return None
    
    def print_summary(self) -> None:
        """Print memory profiling summary."""
        if not self.snapshots:
            print("No memory snapshots available.")
            return
        
        print("\nMemory Usage Summary:")
        print("-" * 50)
        print(f"{'Snapshot':<25} {'Memory (MB)':<15}")
        print("-" * 50)
        
        for snapshot in self.snapshots:
            print(f"{snapshot['name']:<25} {snapshot['memory_mb']:<15.2f}")
        
        print("-" * 50)
        print(f"Peak memory usage: {self.peak_memory:.2f}MB")
        
        if len(self.snapshots) >= 2:
            total_diff = self.snapshots[-1]['memory_mb'] - self.snapshots[0]['memory_mb']
            print(f"Total memory change: {total_diff:+.2f}MB")
    
    def reset(self) -> None:
        """Reset memory profiler."""
        self.snapshots.clear()
        self.peak_memory = 0.0


@contextmanager
def memory_profile(operation_name: str):
    """Context manager for memory profiling.
    
    Args:
        operation_name: Name of the operation
    """
    profiler = MemoryProfiler()
    
    start_memory = profiler.take_snapshot(f"{operation_name}_start")
    
    try:
        yield profiler
    finally:
        end_memory = profiler.take_snapshot(f"{operation_name}_end")
        memory_diff = end_memory - start_memory
        
        logger.info(f"Memory usage for {operation_name}: {memory_diff:+.2f}MB")


class PerformanceMonitor:
    """Real-time performance monitoring for HDC systems."""
    
    def __init__(self, window_size: int = 100):
        """Initialize performance monitor.
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        self.operation_times = {}
        self.operation_counts = {}
        self.error_counts = {}
    
    def record_operation(self, operation: str, duration: float, success: bool = True) -> None:
        """Record an operation.
        
        Args:
            operation: Name of the operation
            duration: Execution time in seconds
            success: Whether the operation succeeded
        """
        # Initialize lists if needed
        if operation not in self.operation_times:
            self.operation_times[operation] = []
            self.operation_counts[operation] = 0
            self.error_counts[operation] = 0
        
        # Record timing
        self.operation_times[operation].append(duration)
        
        # Maintain window size
        if len(self.operation_times[operation]) > self.window_size:
            self.operation_times[operation].pop(0)
        
        # Update counts
        self.operation_counts[operation] += 1
        if not success:
            self.error_counts[operation] += 1
    
    def get_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """Get statistics for an operation.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Dictionary with statistics or None if operation not found
        """
        if operation not in self.operation_times:
            return None
        
        times = self.operation_times[operation]
        if not times:
            return None
        
        return {
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'total_count': self.operation_counts[operation],
            'error_count': self.error_counts[operation],
            'error_rate': self.error_counts[operation] / self.operation_counts[operation],
            'throughput': len(times) / sum(times) if sum(times) > 0 else 0
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {op: self.get_stats(op) for op in self.operation_times.keys()}
    
    def print_dashboard(self) -> None:
        """Print real-time performance dashboard."""
        stats = self.get_all_stats()
        
        if not stats:
            print("No performance data available.")
            return
        
        print("\nPerformance Dashboard:")
        print("=" * 80)
        print(f"{'Operation':<20} {'Avg Time':<12} {'Throughput':<12} {'Errors':<10} {'Total':<10}")
        print("-" * 80)
        
        for operation, data in stats.items():
            if data:
                avg_time = f"{data['avg_time']*1000:.2f}ms"
                throughput = f"{data['throughput']:.1f} ops/s"
                errors = f"{data['error_count']}/{data['total_count']}"
                total = str(data['total_count'])
                
                print(f"{operation:<20} {avg_time:<12} {throughput:<12} {errors:<10} {total:<10}")
        
        print("=" * 80)