"""Advanced performance optimization for HDC operations."""

import torch
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass
import numpy as np

from ..utils.logging import get_logger
from ..core.hypervector import HyperVector
from .memory_manager import MemoryManager

logger = get_logger(__name__)


@dataclass
class OptimizationProfile:
    """Performance optimization profile."""
    operation_name: str
    input_size: int
    execution_time_ms: float
    memory_usage_mb: float
    optimal_batch_size: Optional[int] = None
    optimal_device: Optional[str] = None
    optimization_factor: float = 1.0


class PerformanceProfiler:
    """Intelligent performance profiling and optimization."""
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        """Initialize performance profiler.
        
        Args:
            memory_manager: Memory manager instance
        """
        self.memory_manager = memory_manager or MemoryManager()
        self.profiles: Dict[str, List[OptimizationProfile]] = {}
        self.optimization_cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
        # Performance thresholds
        self.slow_operation_threshold_ms = 100.0
        self.memory_threshold_mb = 1000.0
        
        logger.info("Initialized PerformanceProfiler")
    
    def profile_operation(self, operation_name: str):
        """Decorator to profile operation performance.
        
        Args:
            operation_name: Name of the operation to profile
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self._profile_execution(operation_name, func, *args, **kwargs)
            return wrapper
        return decorator
    
    def _profile_execution(self, operation_name: str, func: Callable, *args, **kwargs) -> Any:
        """Profile function execution."""
        # Measure memory before
        memory_before = self.memory_manager.get_current_memory_usage()
        
        # Estimate input size
        input_size = self._estimate_input_size(*args, **kwargs)
        
        # Execute function
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Measure memory after
        memory_after = self.memory_manager.get_current_memory_usage()
        
        # Calculate metrics
        execution_time_ms = (end_time - start_time) * 1000
        memory_usage_mb = (memory_after - memory_before) * 1024
        
        # Create profile
        profile = OptimizationProfile(
            operation_name=operation_name,
            input_size=input_size,
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_usage_mb
        )
        
        # Store profile
        with self.lock:
            if operation_name not in self.profiles:
                self.profiles[operation_name] = []
            self.profiles[operation_name].append(profile)
            
            # Keep only recent profiles (last 100)
            if len(self.profiles[operation_name]) > 100:
                self.profiles[operation_name] = self.profiles[operation_name][-100:]
        
        # Check for optimization opportunities
        self._check_optimization_opportunity(profile)
        
        return result
    
    def _estimate_input_size(self, *args, **kwargs) -> int:
        """Estimate input size for profiling."""
        total_size = 0
        
        for arg in args:
            if isinstance(arg, torch.Tensor):
                total_size += arg.numel()
            elif isinstance(arg, HyperVector):
                total_size += arg.dim
            elif isinstance(arg, (list, tuple)):
                total_size += len(arg)
            elif isinstance(arg, str):
                total_size += len(arg)
        
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                total_size += value.numel()
            elif isinstance(value, HyperVector):
                total_size += value.dim
            elif isinstance(value, (list, tuple)):
                total_size += len(value)
        
        return total_size
    
    def _check_optimization_opportunity(self, profile: OptimizationProfile):
        """Check for optimization opportunities."""
        # Check for slow operations
        if profile.execution_time_ms > self.slow_operation_threshold_ms:
            logger.warning(
                f"Slow operation detected: {profile.operation_name} "
                f"took {profile.execution_time_ms:.1f}ms"
            )
            self._suggest_optimizations(profile)
        
        # Check for high memory usage
        if profile.memory_usage_mb > self.memory_threshold_mb:
            logger.warning(
                f"High memory usage: {profile.operation_name} "
                f"used {profile.memory_usage_mb:.1f}MB"
            )
    
    def _suggest_optimizations(self, profile: OptimizationProfile):
        """Suggest optimizations for slow operations."""
        suggestions = []
        
        # Suggest batching for large inputs
        if profile.input_size > 10000:
            suggestions.append("Consider batch processing for large inputs")
        
        # Suggest GPU acceleration
        if not torch.cuda.is_available():
            suggestions.append("Consider using GPU acceleration if available")
        
        # Suggest caching for repeated operations
        suggestions.append("Consider caching results for repeated operations")
        
        for suggestion in suggestions:
            logger.info(f"OPTIMIZATION: {profile.operation_name} - {suggestion}")
    
    def get_optimization_recommendations(self, operation_name: str) -> List[str]:
        """Get optimization recommendations for an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            List of optimization recommendations
        """
        if operation_name not in self.profiles:
            return ["No performance data available"]
        
        profiles = self.profiles[operation_name]
        recent_profiles = profiles[-10:]  # Last 10 executions
        
        recommendations = []
        
        # Analyze execution time trends
        times = [p.execution_time_ms for p in recent_profiles]
        avg_time = sum(times) / len(times)
        
        if avg_time > self.slow_operation_threshold_ms:
            recommendations.append(f"Operation is slow (avg: {avg_time:.1f}ms)")
            
            # Check for performance degradation
            if len(times) >= 5:
                early_avg = sum(times[:len(times)//2]) / (len(times)//2)
                late_avg = sum(times[len(times)//2:]) / (len(times) - len(times)//2)
                
                if late_avg > early_avg * 1.2:
                    recommendations.append("Performance degradation detected")
        
        # Analyze memory usage
        memory_usage = [p.memory_usage_mb for p in recent_profiles]
        avg_memory = sum(memory_usage) / len(memory_usage)
        
        if avg_memory > self.memory_threshold_mb:
            recommendations.append(f"High memory usage (avg: {avg_memory:.1f}MB)")
        
        # Input size analysis
        input_sizes = [p.input_size for p in recent_profiles]
        avg_input_size = sum(input_sizes) / len(input_sizes)
        
        if avg_input_size > 100000:
            recommendations.append("Consider batch processing for large inputs")
        
        return recommendations if recommendations else ["No optimizations needed"]


class BatchProcessor:
    """Intelligent batch processing for HDC operations."""
    
    def __init__(self, max_batch_size: int = 1000, memory_limit_gb: float = 2.0):
        """Initialize batch processor.
        
        Args:
            max_batch_size: Maximum batch size
            memory_limit_gb: Memory limit for batch processing
        """
        self.max_batch_size = max_batch_size
        self.memory_limit_gb = memory_limit_gb
        self.optimal_batch_sizes: Dict[str, int] = {}
        
        logger.info(f"Initialized BatchProcessor (max_batch={max_batch_size})")
    
    def process_batch(
        self,
        items: List[Any],
        process_fn: Callable,
        operation_name: str = "batch_op"
    ) -> List[Any]:
        """Process items in optimized batches.
        
        Args:
            items: Items to process
            process_fn: Processing function
            operation_name: Operation name for optimization
            
        Returns:
            Processed results
        """
        if not items:
            return []
        
        # Determine optimal batch size
        batch_size = self._get_optimal_batch_size(operation_name, len(items))
        
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Process batch
            start_time = time.perf_counter()
            batch_results = process_fn(batch)
            end_time = time.perf_counter()
            
            # Update performance metrics
            self._update_batch_performance(
                operation_name, 
                len(batch), 
                (end_time - start_time) * 1000
            )
            
            # Collect results
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
        
        return results
    
    def _get_optimal_batch_size(self, operation_name: str, total_items: int) -> int:
        """Get optimal batch size for operation.
        
        Args:
            operation_name: Operation name
            total_items: Total number of items
            
        Returns:
            Optimal batch size
        """
        # Use cached optimal size if available
        if operation_name in self.optimal_batch_sizes:
            return min(self.optimal_batch_sizes[operation_name], self.max_batch_size)
        
        # Start with conservative batch size
        initial_batch_size = min(100, total_items, self.max_batch_size)
        return initial_batch_size
    
    def _update_batch_performance(
        self, 
        operation_name: str, 
        batch_size: int, 
        execution_time_ms: float
    ):
        """Update batch performance metrics.
        
        Args:
            operation_name: Operation name
            batch_size: Batch size used
            execution_time_ms: Execution time in milliseconds
        """
        # Simple heuristic: prefer batch sizes with better time per item
        time_per_item = execution_time_ms / batch_size
        
        # Update optimal batch size if this performance is better
        if operation_name not in self.optimal_batch_sizes:
            self.optimal_batch_sizes[operation_name] = batch_size
        else:
            current_optimal = self.optimal_batch_sizes[operation_name]
            
            # Prefer larger batches if time per item is similar or better
            if time_per_item < 10.0 and batch_size > current_optimal:
                self.optimal_batch_sizes[operation_name] = batch_size


class ParallelProcessor:
    """Parallel processing for HDC operations."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of worker threads/processes
        """
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
        
        logger.info(f"Initialized ParallelProcessor (workers={self.max_workers})")
    
    def parallel_map(
        self,
        func: Callable,
        items: List[Any],
        use_processes: bool = False,
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """Apply function to items in parallel.
        
        Args:
            func: Function to apply
            items: Items to process
            use_processes: Use process pool instead of thread pool
            chunk_size: Chunk size for processing
            
        Returns:
            Processed results
        """
        if not items:
            return []
        
        if len(items) == 1:
            return [func(items[0])]
        
        # Choose executor
        executor = self.process_executor if use_processes else self.thread_executor
        
        # Determine chunk size
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.max_workers * 2))
        
        # Submit tasks
        futures = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            if len(chunk) == 1:
                future = executor.submit(func, chunk[0])
            else:
                future = executor.submit(self._map_chunk, func, chunk)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
            except Exception as e:
                logger.error(f"Parallel processing error: {e}")
                raise
        
        return results
    
    def _map_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Apply function to a chunk of items."""
        return [func(item) for item in chunk]
    
    def parallel_reduce(
        self,
        func: Callable[[Any, Any], Any],
        items: List[Any],
        initializer: Any = None
    ) -> Any:
        """Reduce items in parallel.
        
        Args:
            func: Reduction function
            items: Items to reduce
            initializer: Initial value
            
        Returns:
            Reduced result
        """
        if not items:
            return initializer
        
        if len(items) == 1:
            return items[0] if initializer is None else func(initializer, items[0])
        
        # Recursive parallel reduction
        if len(items) <= self.max_workers:
            result = items[0] if initializer is None else func(initializer, items[0])
            for item in items[1:]:
                result = func(result, item)
            return result
        
        # Split into chunks and reduce in parallel
        chunk_size = max(1, len(items) // self.max_workers)
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        # Reduce each chunk
        futures = []
        for chunk in chunks:
            future = self.thread_executor.submit(self._reduce_chunk, func, chunk)
            futures.append(future)
        
        # Collect chunk results
        chunk_results = []
        for future in futures:
            chunk_results.append(future.result())
        
        # Final reduction
        result = chunk_results[0] if initializer is None else func(initializer, chunk_results[0])
        for chunk_result in chunk_results[1:]:
            result = func(result, chunk_result)
        
        return result
    
    def _reduce_chunk(self, func: Callable, chunk: List[Any]) -> Any:
        """Reduce a chunk of items."""
        if not chunk:
            return None
        
        result = chunk[0]
        for item in chunk[1:]:
            result = func(result, item)
        return result
    
    def shutdown(self):
        """Shutdown executors."""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)


class AdaptiveOptimizer:
    """Adaptive optimization system that learns from usage patterns."""
    
    def __init__(self):
        """Initialize adaptive optimizer."""
        self.operation_stats: Dict[str, Dict[str, List[float]]] = {}
        self.optimization_history: Dict[str, List[Dict[str, Any]]] = {}
        self.learning_rate = 0.1
        self.lock = threading.Lock()
        
        logger.info("Initialized AdaptiveOptimizer")
    
    def record_performance(
        self,
        operation_name: str,
        input_size: int,
        execution_time_ms: float,
        memory_usage_mb: float,
        optimization_params: Dict[str, Any]
    ):
        """Record performance metrics for adaptive learning.
        
        Args:
            operation_name: Name of the operation
            input_size: Size of input data
            execution_time_ms: Execution time in milliseconds
            memory_usage_mb: Memory usage in MB
            optimization_params: Optimization parameters used
        """
        with self.lock:
            if operation_name not in self.operation_stats:
                self.operation_stats[operation_name] = {
                    'input_sizes': [],
                    'execution_times': [],
                    'memory_usage': [],
                    'params': []
                }
            
            stats = self.operation_stats[operation_name]
            stats['input_sizes'].append(input_size)
            stats['execution_times'].append(execution_time_ms)
            stats['memory_usage'].append(memory_usage_mb)
            stats['params'].append(optimization_params.copy())
            
            # Keep only recent history (last 1000 recordings)
            for key in stats:
                if len(stats[key]) > 1000:
                    stats[key] = stats[key][-1000:]
    
    def get_optimal_parameters(
        self, 
        operation_name: str, 
        input_size: int
    ) -> Dict[str, Any]:
        """Get optimal parameters for an operation based on learning.
        
        Args:
            operation_name: Name of the operation
            input_size: Expected input size
            
        Returns:
            Optimal parameters
        """
        with self.lock:
            if operation_name not in self.operation_stats:
                return self._get_default_parameters()
            
            stats = self.operation_stats[operation_name]
            
            if not stats['input_sizes']:
                return self._get_default_parameters()
            
            # Find similar input sizes
            similar_indices = []
            tolerance = 0.2  # 20% tolerance
            
            for i, size in enumerate(stats['input_sizes']):
                if abs(size - input_size) / max(size, input_size, 1) <= tolerance:
                    similar_indices.append(i)
            
            if not similar_indices:
                # No similar cases, return default
                return self._get_default_parameters()
            
            # Find best performing parameters among similar cases
            best_index = min(
                similar_indices,
                key=lambda i: stats['execution_times'][i] + stats['memory_usage'][i] * 0.01
            )
            
            return stats['params'][best_index].copy()
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default optimization parameters."""
        return {
            'batch_size': 100,
            'use_parallel': True,
            'max_workers': 4,
            'use_cache': True,
            'memory_optimize': True
        }
    
    def update_learning(self, operation_name: str, feedback_score: float):
        """Update learning based on user feedback.
        
        Args:
            operation_name: Name of the operation
            feedback_score: Feedback score (0.0 to 1.0)
        """
        with self.lock:
            if operation_name not in self.optimization_history:
                self.optimization_history[operation_name] = []
            
            self.optimization_history[operation_name].append({
                'timestamp': time.time(),
                'feedback_score': feedback_score
            })
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary and statistics.
        
        Returns:
            Optimization summary
        """
        with self.lock:
            summary = {
                'operations_tracked': len(self.operation_stats),
                'total_recordings': sum(
                    len(stats['execution_times']) 
                    for stats in self.operation_stats.values()
                ),
                'learning_rate': self.learning_rate
            }
            
            # Add per-operation summaries
            operation_summaries = {}
            for op_name, stats in self.operation_stats.items():
                if stats['execution_times']:
                    operation_summaries[op_name] = {
                        'recordings': len(stats['execution_times']),
                        'avg_execution_time_ms': sum(stats['execution_times']) / len(stats['execution_times']),
                        'avg_memory_usage_mb': sum(stats['memory_usage']) / len(stats['memory_usage']),
                        'input_size_range': (min(stats['input_sizes']), max(stats['input_sizes']))
                    }
            
            summary['operations'] = operation_summaries
            return summary


def create_optimized_hdc_system(
    dim: int = 10000,
    device: Optional[str] = None,
    enable_profiling: bool = True,
    enable_batching: bool = True,
    enable_parallel: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """Create an optimized HDC system with all performance enhancements.
    
    Args:
        dim: Hypervector dimensionality
        device: Compute device
        enable_profiling: Enable performance profiling
        enable_batching: Enable batch processing
        enable_parallel: Enable parallel processing
        
    Returns:
        Tuple of (HDC system, optimization components)
    """
    from ..core.system import HDCSystem
    
    # Create components
    memory_manager = MemoryManager(max_memory_gb=8.0)
    profiler = PerformanceProfiler(memory_manager) if enable_profiling else None
    batch_processor = BatchProcessor() if enable_batching else None
    parallel_processor = ParallelProcessor() if enable_parallel else None
    adaptive_optimizer = AdaptiveOptimizer()
    
    # Create HDC system
    hdc_system = HDCSystem(dim=dim, device=device)
    
    # Wrap methods with profiling if enabled
    if profiler:
        hdc_system.encode_text = profiler.profile_operation("encode_text")(hdc_system.encode_text)
        hdc_system.encode_image = profiler.profile_operation("encode_image")(hdc_system.encode_image)
        hdc_system.encode_eeg = profiler.profile_operation("encode_eeg")(hdc_system.encode_eeg)
        hdc_system.bind = profiler.profile_operation("bind")(hdc_system.bind)
        hdc_system.bundle = profiler.profile_operation("bundle")(hdc_system.bundle)
        hdc_system.cosine_similarity = profiler.profile_operation("cosine_similarity")(hdc_system.cosine_similarity)
    
    optimization_components = {
        'memory_manager': memory_manager,
        'profiler': profiler,
        'batch_processor': batch_processor,
        'parallel_processor': parallel_processor,
        'adaptive_optimizer': adaptive_optimizer
    }
    
    logger.info(f"Created optimized HDC system (dim={dim}, device={device})")
    
    return hdc_system, optimization_components