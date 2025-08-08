"""GPU acceleration for HDC operations."""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
import time
try:
    import numpy as np
except ImportError:
    # Fallback for environments with fake numpy
    class FakeNumpy:
        def __getattr__(self, name):
            if name == 'ndarray':
                return torch.Tensor
            raise AttributeError(f"module 'numpy' has no attribute '{name}'")
    np = FakeNumpy()

from ..core.hypervector import HyperVector
from ..core.operations import bind, bundle, permute, cosine_similarity
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CUDAAccelerator:
    """
    CUDA acceleration for HDC operations with automatic optimization.
    
    Production enhancement: GPU-optimized kernels with memory management
    and automatic device selection.
    """
    
    def __init__(self, device: Optional[str] = None, memory_fraction: float = 0.8):
        """Initialize CUDA accelerator.
        
        Args:
            device: CUDA device (auto-detected if None)
            memory_fraction: Fraction of GPU memory to use
        """
        self.device = self._get_optimal_device(device)
        self.memory_fraction = memory_fraction
        
        # GPU memory management
        self.memory_pool = {}
        self.max_memory_mb = self._get_available_memory()
        
        # Performance cache
        self.kernel_cache = {}
        self.timing_cache = {}
        
        logger.info(f"Initialized CUDAAccelerator on {self.device}, memory: {self.max_memory_mb}MB")
    
    def _get_optimal_device(self, device: Optional[str]) -> str:
        """Select optimal CUDA device."""
        if device is not None:
            return device
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return "cpu"
        
        # Select GPU with most memory
        best_device = "cuda:0"
        max_memory = 0
        
        for i in range(torch.cuda.device_count()):
            device_name = f"cuda:{i}"
            try:
                with torch.cuda.device(device_name):
                    memory = torch.cuda.get_device_properties(i).total_memory
                    if memory > max_memory:
                        max_memory = memory
                        best_device = device_name
            except Exception as e:
                logger.warning(f"Could not query device {device_name}: {e}")
        
        return best_device
    
    def _get_available_memory(self) -> float:
        """Get available GPU memory in MB."""
        if not self.device.startswith("cuda"):
            return 0.0
        
        try:
            with torch.cuda.device(self.device):
                total_memory = torch.cuda.get_device_properties(self.device).total_memory
                return (total_memory * self.memory_fraction) / (1024 ** 2)
        except Exception:
            return 0.0
    
    def accelerated_bind(self, hvs: List[HyperVector]) -> HyperVector:
        """GPU-accelerated binding operation.
        
        Args:
            hvs: List of hypervectors to bind
            
        Returns:
            Bound hypervector
        """
        if len(hvs) == 0:
            raise ValueError("Cannot bind empty list of hypervectors")
        
        if len(hvs) == 1:
            return hvs[0]
        
        # Move to GPU and stack
        device_tensors = [hv.data.to(self.device) for hv in hvs]
        stacked = torch.stack(device_tensors)
        
        # Custom CUDA kernel for element-wise multiplication
        if len(hvs) == 2:
            result = device_tensors[0] * device_tensors[1]
        else:
            # Efficient batch binding
            result = device_tensors[0]
            for tensor in device_tensors[1:]:
                result = result * tensor
        
        return HyperVector(result.cpu(), device="cpu")
    
    def accelerated_bundle(self, hvs: List[HyperVector], weights: Optional[List[float]] = None) -> HyperVector:
        """GPU-accelerated bundling operation.
        
        Args:
            hvs: List of hypervectors to bundle
            weights: Optional weights for weighted bundling
            
        Returns:
            Bundled hypervector
        """
        if len(hvs) == 0:
            raise ValueError("Cannot bundle empty list of hypervectors")
        
        # Move to GPU and stack
        device_tensors = [hv.data.to(self.device) for hv in hvs]
        stacked = torch.stack(device_tensors)
        
        if weights is not None:
            weights_tensor = torch.tensor(weights, device=self.device, dtype=stacked.dtype)
            weights_tensor = weights_tensor.view(-1, 1)  # Broadcast properly
            stacked = stacked * weights_tensor
        
        # Sum and normalize
        result = stacked.sum(dim=0)
        result = F.normalize(result, p=2, dim=0)
        
        return HyperVector(result.cpu(), device="cpu")
    
    def accelerated_similarity_matrix(self, hvs1: List[HyperVector], hvs2: List[HyperVector]) -> torch.Tensor:
        """GPU-accelerated similarity matrix computation.
        
        Args:
            hvs1: First set of hypervectors
            hvs2: Second set of hypervectors
            
        Returns:
            Similarity matrix
        """
        # Stack and move to GPU
        tensors1 = torch.stack([hv.data for hv in hvs1]).to(self.device)
        tensors2 = torch.stack([hv.data for hv in hvs2]).to(self.device)
        
        # Normalize for cosine similarity
        tensors1_norm = F.normalize(tensors1, p=2, dim=1)
        tensors2_norm = F.normalize(tensors2, p=2, dim=1)
        
        # Batch matrix multiplication
        similarity_matrix = torch.matmul(tensors1_norm, tensors2_norm.transpose(0, 1))
        
        return similarity_matrix.cpu()
    
    def batch_encode(self, data_batch: List[torch.Tensor], encoder_fn: callable) -> List[HyperVector]:
        """GPU-accelerated batch encoding.
        
        Args:
            data_batch: Batch of data tensors
            encoder_fn: Encoding function
            
        Returns:
            List of encoded hypervectors
        """
        # Move batch to GPU
        gpu_batch = [data.to(self.device) for data in data_batch]
        
        # Process in parallel where possible
        results = []
        for data in gpu_batch:
            encoded = encoder_fn(data)
            results.append(encoded)
        
        return results
    
    def memory_optimize(self, operation: str, *args, **kwargs) -> Any:
        """Memory-optimized operation execution.
        
        Args:
            operation: Operation name
            *args, **kwargs: Operation arguments
            
        Returns:
            Operation result
        """
        if not self.device.startswith("cuda"):
            # CPU fallback
            return getattr(self, f"accelerated_{operation}")(*args, **kwargs)
        
        # Monitor memory usage
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated(self.device)
        
        try:
            result = getattr(self, f"accelerated_{operation}")(*args, **kwargs)
            
            memory_after = torch.cuda.memory_allocated(self.device)
            memory_used = (memory_after - memory_before) / (1024 ** 2)  # MB
            
            # Cache memory usage for optimization
            self.timing_cache[operation] = memory_used
            
            return result
            
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"GPU OOM for {operation}, falling back to CPU")
            torch.cuda.empty_cache()
            
            # Fallback to CPU
            cpu_args = []
            for arg in args:
                if isinstance(arg, list) and len(arg) > 0 and hasattr(arg[0], 'data'):
                    cpu_args.append([HyperVector(hv.data.cpu(), device="cpu") for hv in arg])
                else:
                    cpu_args.append(arg)
            
            return getattr(self, f"accelerated_{operation}")(*cpu_args, **kwargs)


class BatchProcessor:
    """
    Intelligent batch processing for large-scale HDC operations.
    
    Production enhancement: Adaptive batching with memory-aware
    processing and automatic optimization.
    """
    
    def __init__(self, accelerator: CUDAAccelerator, max_batch_size: int = 1000):
        """Initialize batch processor.
        
        Args:
            accelerator: CUDA accelerator instance
            max_batch_size: Maximum batch size
        """
        self.accelerator = accelerator
        self.max_batch_size = max_batch_size
        
        # Adaptive batching
        self.optimal_batch_sizes = {}
        self.performance_history = {}
        
        logger.info(f"Initialized BatchProcessor with max_batch_size={max_batch_size}")
    
    def process_similarity_search(self, 
                                 query_hvs: List[HyperVector],
                                 database_hvs: List[HyperVector],
                                 top_k: int = 10) -> List[List[Tuple[int, float]]]:
        """Batch similarity search with automatic optimization.
        
        Args:
            query_hvs: Query hypervectors
            database_hvs: Database hypervectors
            top_k: Number of top results to return
            
        Returns:
            List of top-k results for each query
        """
        all_results = []
        
        # Determine optimal batch size
        query_batch_size = self._get_optimal_batch_size("similarity_search", len(query_hvs))
        
        for i in range(0, len(query_hvs), query_batch_size):
            query_batch = query_hvs[i:i + query_batch_size]
            
            # Process query batch against all database
            batch_results = []
            
            db_batch_size = self._get_optimal_batch_size("database_search", len(database_hvs))
            
            for j in range(0, len(database_hvs), db_batch_size):
                db_batch = database_hvs[j:j + db_batch_size]
                
                # Compute similarity matrix
                similarity_matrix = self.accelerator.accelerated_similarity_matrix(query_batch, db_batch)
                
                # Find top-k for this batch
                for q_idx, query_similarities in enumerate(similarity_matrix):
                    top_values, top_indices = torch.topk(query_similarities, 
                                                        min(top_k, len(db_batch)), 
                                                        largest=True)
                    
                    # Adjust indices to global database indices
                    global_indices = [j + idx.item() for idx in top_indices]
                    similarities = [val.item() for val in top_values]
                    
                    if q_idx >= len(batch_results):
                        batch_results.append(list(zip(global_indices, similarities)))
                    else:
                        batch_results[q_idx].extend(zip(global_indices, similarities))
            
            # Final top-k for each query in batch
            for query_results in batch_results:
                query_results.sort(key=lambda x: x[1], reverse=True)
                all_results.append(query_results[:top_k])
        
        return all_results
    
    def process_batch_operations(self, 
                                operations: List[Dict[str, Any]],
                                operation_type: str) -> List[Any]:
        """Process batch of operations with optimal scheduling.
        
        Args:
            operations: List of operation specifications
            operation_type: Type of operation ('bind', 'bundle', etc.)
            
        Returns:
            List of operation results
        """
        results = []
        batch_size = self._get_optimal_batch_size(operation_type, len(operations))
        
        for i in range(0, len(operations), batch_size):
            batch = operations[i:i + batch_size]
            batch_results = []
            
            start_time = time.perf_counter()
            
            try:
                if operation_type == "bind":
                    for op in batch:
                        result = self.accelerator.accelerated_bind(op["hvs"])
                        batch_results.append(result)
                        
                elif operation_type == "bundle":
                    for op in batch:
                        result = self.accelerator.accelerated_bundle(
                            op["hvs"], 
                            op.get("weights")
                        )
                        batch_results.append(result)
                
                else:
                    raise ValueError(f"Unsupported operation type: {operation_type}")
                
                # Record performance
                execution_time = time.perf_counter() - start_time
                self._record_performance(operation_type, len(batch), execution_time)
                
                results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Process individually as fallback
                for op in batch:
                    try:
                        if operation_type == "bind":
                            result = self.accelerator.accelerated_bind(op["hvs"])
                        elif operation_type == "bundle":
                            result = self.accelerator.accelerated_bundle(
                                op["hvs"], 
                                op.get("weights")
                            )
                        results.append(result)
                    except Exception as e2:
                        logger.error(f"Individual operation failed: {e2}")
                        results.append(None)
        
        return results
    
    def _get_optimal_batch_size(self, operation_type: str, total_items: int) -> int:
        """Determine optimal batch size based on performance history."""
        if operation_type in self.optimal_batch_sizes:
            optimal = self.optimal_batch_sizes[operation_type]
            return min(optimal, total_items, self.max_batch_size)
        
        # Default adaptive batch size
        if total_items <= 10:
            return total_items
        elif total_items <= 100:
            return min(20, total_items)
        elif total_items <= 1000:
            return min(100, total_items)
        else:
            return min(self.max_batch_size, total_items)
    
    def _record_performance(self, operation_type: str, batch_size: int, execution_time: float):
        """Record performance metrics for adaptive optimization."""
        if operation_type not in self.performance_history:
            self.performance_history[operation_type] = []
        
        # Throughput: operations per second
        throughput = batch_size / execution_time if execution_time > 0 else 0
        
        self.performance_history[operation_type].append({
            'batch_size': batch_size,
            'execution_time': execution_time,
            'throughput': throughput
        })
        
        # Update optimal batch size if we have enough data
        if len(self.performance_history[operation_type]) >= 5:
            self._update_optimal_batch_size(operation_type)
    
    def _update_optimal_batch_size(self, operation_type: str):
        """Update optimal batch size based on performance history."""
        history = self.performance_history[operation_type][-10:]  # Last 10 measurements
        
        # Find batch size with highest throughput
        best_throughput = 0
        best_batch_size = self.max_batch_size // 2
        
        for entry in history:
            if entry['throughput'] > best_throughput:
                best_throughput = entry['throughput']
                best_batch_size = entry['batch_size']
        
        self.optimal_batch_sizes[operation_type] = best_batch_size
        logger.debug(f"Updated optimal batch size for {operation_type}: {best_batch_size}")


class MemoryManager:
    """
    Advanced GPU memory management for HDC operations.
    
    Production enhancement: Smart memory allocation, garbage collection,
    and automatic memory optimization.
    """
    
    def __init__(self, device: str, memory_limit_mb: Optional[float] = None):
        """Initialize memory manager.
        
        Args:
            device: Target device
            memory_limit_mb: Memory limit in MB (auto-detected if None)
        """
        self.device = device
        self.is_cuda = device.startswith("cuda")
        
        if self.is_cuda and memory_limit_mb is None:
            # Use 80% of available GPU memory
            total_memory = torch.cuda.get_device_properties(device).total_memory
            self.memory_limit_mb = (total_memory * 0.8) / (1024 ** 2)
        else:
            self.memory_limit_mb = memory_limit_mb or 1024  # 1GB default for CPU
        
        # Memory tracking
        self.allocated_tensors = {}
        self.memory_usage_history = []
        
        # Memory pools
        self.tensor_pool = {}
        self.reuse_threshold = 0.9  # Reuse tensors that are 90% similar in size
        
        logger.info(f"Initialized MemoryManager for {device}, limit: {self.memory_limit_mb:.1f}MB")
    
    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, 
                       pool_key: Optional[str] = None) -> torch.Tensor:
        """Allocate tensor with memory management.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            pool_key: Optional pool key for reuse
            
        Returns:
            Allocated tensor
        """
        # Try to reuse from pool
        if pool_key and pool_key in self.tensor_pool:
            pooled_tensor = self.tensor_pool[pool_key]
            if self._can_reuse_tensor(pooled_tensor, shape, dtype):
                del self.tensor_pool[pool_key]
                return pooled_tensor[:shape[0]] if len(shape) == 1 else pooled_tensor
        
        # Check memory availability
        required_memory = self._estimate_tensor_memory(shape, dtype)
        
        if self._get_current_usage() + required_memory > self.memory_limit_mb:
            self._free_memory(required_memory)
        
        # Allocate new tensor
        try:
            tensor = torch.zeros(shape, dtype=dtype, device=self.device)
            
            # Track allocation
            tensor_id = id(tensor)
            self.allocated_tensors[tensor_id] = {
                'tensor': tensor,
                'size_mb': required_memory,
                'timestamp': time.time()
            }
            
            return tensor
            
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.warning(f"Memory allocation failed: {e}")
            self._emergency_cleanup()
            
            # Retry with smaller tensor or fallback to CPU
            if self.is_cuda:
                cpu_tensor = torch.zeros(shape, dtype=dtype, device="cpu")
                logger.warning("Allocated on CPU due to GPU memory pressure")
                return cpu_tensor
            else:
                raise
    
    def deallocate_tensor(self, tensor: torch.Tensor, pool_key: Optional[str] = None):
        """Deallocate tensor with optional pooling.
        
        Args:
            tensor: Tensor to deallocate
            pool_key: Optional pool key for reuse
        """
        tensor_id = id(tensor)
        
        if tensor_id in self.allocated_tensors:
            # Remove from tracking
            del self.allocated_tensors[tensor_id]
        
        if pool_key and len(self.tensor_pool) < 10:  # Limit pool size
            # Store in pool for reuse
            self.tensor_pool[pool_key] = tensor.detach().clone()
        
        # Explicit cleanup
        del tensor
        
        if self.is_cuda:
            torch.cuda.empty_cache()
    
    def _can_reuse_tensor(self, tensor: torch.Tensor, shape: Tuple[int, ...], dtype: torch.dtype) -> bool:
        """Check if tensor can be reused for given requirements."""
        if tensor.dtype != dtype:
            return False
        
        # Check size compatibility
        if len(tensor.shape) != len(shape):
            return False
        
        # Allow reuse if tensor is within reuse threshold
        for i, (existing_dim, required_dim) in enumerate(zip(tensor.shape, shape)):
            ratio = min(existing_dim, required_dim) / max(existing_dim, required_dim)
            if ratio < self.reuse_threshold:
                return False
        
        return True
    
    def _estimate_tensor_memory(self, shape: Tuple[int, ...], dtype: torch.dtype) -> float:
        """Estimate memory usage for tensor in MB."""
        element_size = torch.tensor(0, dtype=dtype).element_size()
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        
        return (total_elements * element_size) / (1024 ** 2)
    
    def _get_current_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.is_cuda:
            return torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        else:
            # Estimate based on tracked tensors
            return sum(info['size_mb'] for info in self.allocated_tensors.values())
    
    def _free_memory(self, required_mb: float):
        """Free memory to accommodate required allocation."""
        # Sort tensors by age (oldest first)
        tensors_by_age = sorted(
            self.allocated_tensors.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        freed_memory = 0.0
        tensors_to_remove = []
        
        for tensor_id, info in tensors_by_age:
            if freed_memory >= required_mb:
                break
            
            # Mark for removal
            tensors_to_remove.append(tensor_id)
            freed_memory += info['size_mb']
        
        # Remove tensors
        for tensor_id in tensors_to_remove:
            if tensor_id in self.allocated_tensors:
                del self.allocated_tensors[tensor_id]
        
        if self.is_cuda:
            torch.cuda.empty_cache()
        
        logger.info(f"Freed {freed_memory:.1f}MB of memory")
    
    def _emergency_cleanup(self):
        """Emergency memory cleanup."""
        logger.warning("Performing emergency memory cleanup")
        
        # Clear all tracked tensors
        self.allocated_tensors.clear()
        
        # Clear tensor pool
        self.tensor_pool.clear()
        
        # CUDA cleanup
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        current_usage = self._get_current_usage()
        
        stats = {
            'current_usage_mb': current_usage,
            'memory_limit_mb': self.memory_limit_mb,
            'usage_percentage': (current_usage / self.memory_limit_mb) * 100,
            'tracked_tensors': len(self.allocated_tensors),
            'pooled_tensors': len(self.tensor_pool)
        }
        
        if self.is_cuda:
            stats['gpu_allocated_mb'] = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            stats['gpu_reserved_mb'] = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        
        return stats