"""Memory management utilities for large-scale HDC operations."""

import gc
import weakref
from typing import Dict, List, Optional, Any, Callable
import torch
import threading
from collections import OrderedDict
import time

from ..utils.logging import get_logger
from ..core.hypervector import HyperVector

logger = get_logger(__name__)


class MemoryManager:
    """Intelligent memory management for HDC operations."""
    
    def __init__(
        self,
        max_memory_gb: float = 8.0,
        cache_size: int = 1000,
        cleanup_threshold: float = 0.8
    ):
        """Initialize memory manager.
        
        Args:
            max_memory_gb: Maximum memory usage in GB
            cache_size: Maximum number of cached items
            cleanup_threshold: Memory usage threshold to trigger cleanup
        """
        self.max_memory_gb = max_memory_gb
        self.cache_size = cache_size
        self.cleanup_threshold = cleanup_threshold
        
        # Memory tracking
        self.allocated_objects = weakref.WeakSet()
        self.memory_usage = 0.0
        
        # Cache management
        self.cache = OrderedDict()
        self.cache_lock = threading.Lock()
        
        # Cleanup tracking
        self.last_cleanup = time.time()
        self.cleanup_interval = 60.0  # seconds
        
        logger.info(f"Initialized MemoryManager with {max_memory_gb}GB limit")
    
    def get_current_memory_usage(self) -> float:
        """Get current memory usage in GB.
        
        Returns:
            Current memory usage in GB
        """
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024**3)
            else:
                import psutil
                return psutil.Process().memory_info().rss / (1024**3)
        except ImportError:
            return 0.0
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits.
        
        Returns:
            True if within limits, False otherwise
        """
        current_memory = self.get_current_memory_usage()
        return current_memory < self.max_memory_gb
    
    def cleanup_memory(self, force: bool = False) -> float:
        """Clean up memory and return freed amount.
        
        Args:
            force: Force cleanup regardless of threshold
            
        Returns:
            Amount of memory freed in GB
        """
        current_time = time.time()
        current_memory = self.get_current_memory_usage()
        
        should_cleanup = (
            force or 
            current_memory > self.max_memory_gb * self.cleanup_threshold or
            current_time - self.last_cleanup > self.cleanup_interval
        )
        
        if not should_cleanup:
            return 0.0
        
        logger.debug(f"Starting memory cleanup (current: {current_memory:.2f}GB)")
        
        memory_before = current_memory
        
        # Clear cache
        with self.cache_lock:
            cache_size_before = len(self.cache)
            self.cache.clear()
            logger.debug(f"Cleared {cache_size_before} cached items")
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        memory_after = self.get_current_memory_usage()
        freed_memory = memory_before - memory_after
        
        self.last_cleanup = current_time
        
        logger.info(f"Memory cleanup freed {freed_memory:.2f}GB")
        return freed_memory
    
    def register_object(self, obj: Any) -> None:
        """Register object for memory tracking.
        
        Args:
            obj: Object to track
        """
        self.allocated_objects.add(obj)
    
    def cached_operation(
        self,
        key: str,
        operation_fn: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with caching.
        
        Args:
            key: Cache key
            operation_fn: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Operation result (cached or computed)
        """
        with self.cache_lock:
            # Check cache first
            if key in self.cache:
                # Move to end (mark as recently used)
                self.cache.move_to_end(key)
                logger.debug(f"Cache hit for key: {key}")
                return self.cache[key]
            
            # Execute operation
            result = operation_fn(*args, **kwargs)
            
            # Add to cache
            self.cache[key] = result
            
            # Evict oldest if cache is full
            while len(self.cache) > self.cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                logger.debug(f"Evicted cache key: {oldest_key}")
            
            logger.debug(f"Cached result for key: {key}")
            return result
    
    def memory_efficient_batch_process(
        self,
        data: List[Any],
        process_fn: Callable,
        max_batch_memory_gb: float = 1.0
    ) -> List[Any]:
        """Process data in memory-efficient batches.
        
        Args:
            data: Data to process
            process_fn: Processing function
            max_batch_memory_gb: Maximum memory per batch in GB
            
        Returns:
            Processed results
        """
        results = []
        current_batch = []
        current_batch_memory = 0.0
        
        for item in data:
            # Estimate item memory usage
            item_memory = self._estimate_memory_usage(item)
            
            # Check if adding item would exceed batch memory limit
            if (current_batch_memory + item_memory > max_batch_memory_gb 
                and current_batch):
                
                # Process current batch
                batch_results = process_fn(current_batch)
                results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
                
                # Clean up batch memory
                del current_batch
                gc.collect()
                
                # Start new batch
                current_batch = [item]
                current_batch_memory = item_memory
            else:
                current_batch.append(item)
                current_batch_memory += item_memory
        
        # Process final batch
        if current_batch:
            batch_results = process_fn(current_batch)
            results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
        
        return results
    
    def _estimate_memory_usage(self, item: Any) -> float:
        """Estimate memory usage of an item in GB.
        
        Args:
            item: Item to estimate
            
        Returns:
            Estimated memory usage in GB
        """
        if isinstance(item, torch.Tensor):
            return item.numel() * item.element_size() / (1024**3)
        elif isinstance(item, HyperVector):
            return self._estimate_memory_usage(item.data)
        elif hasattr(item, '__sizeof__'):
            return item.__sizeof__() / (1024**3)
        else:
            return 0.001  # 1MB default estimate
    
    def create_memory_pool(self, size_gb: float) -> 'MemoryPool':
        """Create a memory pool for efficient allocation.
        
        Args:
            size_gb: Size of memory pool in GB
            
        Returns:
            MemoryPool instance
        """
        return MemoryPool(size_gb, self)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        current_memory = self.get_current_memory_usage()
        
        return {
            'current_memory_gb': current_memory,
            'max_memory_gb': self.max_memory_gb,
            'memory_utilization': current_memory / self.max_memory_gb,
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size,
            'tracked_objects': len(self.allocated_objects),
            'last_cleanup': self.last_cleanup,
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_memory(force=True)


class MemoryPool:
    """Pre-allocated memory pool for efficient tensor operations."""
    
    def __init__(self, size_gb: float, memory_manager: MemoryManager):
        """Initialize memory pool.
        
        Args:
            size_gb: Size of memory pool in GB
            memory_manager: Parent memory manager
        """
        self.size_gb = size_gb
        self.memory_manager = memory_manager
        self.allocated_tensors = []
        self.free_tensors = []
        self.lock = threading.Lock()
        
        logger.info(f"Created memory pool of {size_gb}GB")
    
    def get_tensor(self, shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get tensor from pool or allocate new one.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            
        Returns:
            Tensor from pool
        """
        with self.lock:
            # Look for suitable tensor in free list
            for i, tensor in enumerate(self.free_tensors):
                if tensor.shape == shape and tensor.dtype == dtype:
                    # Reuse existing tensor
                    tensor = self.free_tensors.pop(i)
                    self.allocated_tensors.append(tensor)
                    tensor.zero_()  # Clear previous data
                    return tensor
            
            # Allocate new tensor
            tensor = torch.zeros(shape, dtype=dtype)
            self.allocated_tensors.append(tensor)
            return tensor
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return tensor to pool.
        
        Args:
            tensor: Tensor to return
        """
        with self.lock:
            if tensor in self.allocated_tensors:
                self.allocated_tensors.remove(tensor)
                self.free_tensors.append(tensor)
    
    def clear_pool(self) -> None:
        """Clear all tensors in pool."""
        with self.lock:
            self.allocated_tensors.clear()
            self.free_tensors.clear()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Cleared memory pool")


class SmartCache:
    """Intelligent cache with automatic eviction and memory management."""
    
    def __init__(
        self, 
        max_size: int = 1000,
        max_memory_gb: float = 2.0,
        ttl_seconds: Optional[float] = None
    ):
        """Initialize smart cache.
        
        Args:
            max_size: Maximum number of cached items
            max_memory_gb: Maximum memory usage in GB
            ttl_seconds: Time-to-live for cached items
        """
        self.max_size = max_size
        self.max_memory_gb = max_memory_gb
        self.ttl_seconds = ttl_seconds
        
        self.cache = OrderedDict()
        self.access_times = {}
        self.access_counts = {}
        self.memory_usage = 0.0
        self.lock = threading.Lock()
        
        logger.info(f"Initialized SmartCache (size={max_size}, memory={max_memory_gb}GB)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None if not found
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if self.ttl_seconds:
                age = time.time() - self.access_times.get(key, 0)
                if age > self.ttl_seconds:
                    self._remove_item(key)
                    return None
            
            # Update access info
            self.cache.move_to_end(key)
            self.access_times[key] = time.time()
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            
            return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Estimate memory usage
            value_memory = self._estimate_memory_usage(value)
            
            # Remove existing item if key exists
            if key in self.cache:
                self._remove_item(key)
            
            # Evict items if necessary
            while (len(self.cache) >= self.max_size or 
                   self.memory_usage + value_memory > self.max_memory_gb):
                if not self.cache:
                    break  # Cache is empty
                
                # Use LFU eviction policy
                lfu_key = min(self.access_counts.keys(), 
                             key=lambda k: self.access_counts[k])
                self._remove_item(lfu_key)
            
            # Add new item
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            self.memory_usage += value_memory
    
    def _remove_item(self, key: str) -> None:
        """Remove item from cache."""
        if key in self.cache:
            value = self.cache[key]
            value_memory = self._estimate_memory_usage(value)
            
            del self.cache[key]
            del self.access_times[key]
            del self.access_counts[key]
            
            self.memory_usage -= value_memory
    
    def _estimate_memory_usage(self, value: Any) -> float:
        """Estimate memory usage in GB."""
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size() / (1024**3)
        elif isinstance(value, HyperVector):
            return self._estimate_memory_usage(value.data)
        else:
            return 0.001  # 1MB default
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.memory_usage = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_gb': self.memory_usage,
                'max_memory_gb': self.max_memory_gb,
                'hit_rate': self._calculate_hit_rate(),
            }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if not self.access_counts:
            return 0.0
        
        total_accesses = sum(self.access_counts.values())
        return len(self.access_counts) / total_accesses if total_accesses > 0 else 0.0