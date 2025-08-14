"""GPU optimization utilities for hyperdimensional computing."""

import math
import time
from typing import Optional, Tuple, List, Dict, Any, Union
import torch
import torch.nn.functional as F
from functools import lru_cache


class CUDAKernelManager:
    """Manage custom CUDA kernels for HDC operations."""
    
    def __init__(self):
        self.kernels = {}
        self.device_capabilities = self._get_device_capabilities()
        
    def _get_device_capabilities(self) -> Dict[str, Any]:
        """Get CUDA device capabilities."""
        if not torch.cuda.is_available():
            return {}
        
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        return {
            'major': props.major,
            'minor': props.minor,
            'total_memory': props.total_memory,
            'multiprocessor_count': props.multiprocessor_count,
            'max_threads_per_block': props.max_threads_per_block,
            'max_shared_memory_per_block': props.max_shared_memory_per_block,
            'warp_size': 32  # Standard for all CUDA GPUs
        }
    
    def get_optimal_block_size(self, problem_size: int) -> Tuple[int, int]:
        """Calculate optimal CUDA block and grid sizes."""
        if not self.device_capabilities:
            return 256, math.ceil(problem_size / 256)  # Default fallback
        
        max_threads = self.device_capabilities['max_threads_per_block']
        warp_size = self.device_capabilities['warp_size']
        
        # Use multiple of warp size for efficiency
        optimal_block_size = min(max_threads, ((problem_size + warp_size - 1) // warp_size) * warp_size)
        optimal_block_size = max(warp_size, optimal_block_size)
        
        grid_size = math.ceil(problem_size / optimal_block_size)
        
        return optimal_block_size, grid_size


class TensorMemoryPool:
    """Memory pool for efficient tensor allocation."""
    
    def __init__(self, device: str = 'cuda', initial_size: int = 1024**3):  # 1GB
        self.device = device
        self.pools: Dict[Tuple[torch.dtype, tuple], List[torch.Tensor]] = {}
        self.allocated_memory = 0
        self.peak_memory = 0
        self.allocation_count = 0
        
    def get_tensor(self, shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get a tensor from the pool or allocate new one."""
        key = (dtype, shape)
        
        if key in self.pools and self.pools[key]:
            tensor = self.pools[key].pop()
            tensor.zero_()  # Clear data
            return tensor
        
        # Allocate new tensor
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        self.allocation_count += 1
        
        # Track memory usage
        tensor_size = tensor.element_size() * tensor.nelement()
        self.allocated_memory += tensor_size
        self.peak_memory = max(self.peak_memory, self.allocated_memory)
        
        return tensor
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return a tensor to the pool."""
        if tensor.device.type != self.device.split(':')[0]:
            return  # Wrong device
        
        key = (tensor.dtype, tuple(tensor.shape))
        
        if key not in self.pools:
            self.pools[key] = []
        
        # Only keep reasonable number of tensors in pool
        if len(self.pools[key]) < 10:
            self.pools[key].append(tensor.detach())
    
    def clear_pool(self):
        """Clear all tensors from pool."""
        self.pools.clear()
        self.allocated_memory = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        pool_sizes = {str(k): len(v) for k, v in self.pools.items()}
        
        return {
            'allocated_memory_mb': self.allocated_memory / (1024**2),
            'peak_memory_mb': self.peak_memory / (1024**2),
            'allocation_count': self.allocation_count,
            'pool_sizes': pool_sizes,
            'total_pooled_tensors': sum(len(v) for v in self.pools.values())
        }


class OptimizedHDCOperations:
    """Optimized implementations of HDC operations."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.memory_pool = TensorMemoryPool(device)
        self.kernel_manager = CUDAKernelManager()
        
    @torch.jit.script
    def _fast_bind_dense(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """JIT-compiled dense binding operation."""
        return x * y
    
    @torch.jit.script
    def _fast_bind_binary(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """JIT-compiled binary binding operation."""
        return torch.sign(x * y)
    
    @torch.jit.script
    def _fast_bundle(tensors: List[torch.Tensor], weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """JIT-compiled bundling operation."""
        if weights is not None:
            result = torch.zeros_like(tensors[0])
            for i, tensor in enumerate(tensors):
                result += weights[i] * tensor
            return result
        else:
            return torch.stack(tensors).mean(dim=0)
    
    def optimized_bind(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        mode: str = 'dense'
    ) -> torch.Tensor:
        """Optimized binding operation."""
        if mode == 'dense':
            return self._fast_bind_dense(x, y)
        elif mode == 'binary':
            return self._fast_bind_binary(x, y)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    def optimized_bundle(
        self, 
        tensors: List[torch.Tensor], 
        weights: Optional[torch.Tensor] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """Optimized bundling operation."""
        if not tensors:
            raise ValueError("Empty tensor list")
        
        # Use memory pool for intermediate results
        result = self.memory_pool.get_tensor(tensors[0].shape, tensors[0].dtype)
        
        try:
            if weights is not None:
                result.zero_()
                for i, tensor in enumerate(tensors):
                    result.add_(tensor, alpha=weights[i])
            else:
                # Use efficient stacking and mean
                if len(tensors) == 1:
                    result.copy_(tensors[0])
                else:
                    stacked = torch.stack(tensors)
                    torch.mean(stacked, dim=0, out=result)
            
            if normalize:
                norm = torch.norm(result, dim=-1, keepdim=True)
                norm = torch.clamp(norm, min=1e-8)  # Avoid division by zero
                result.div_(norm)
            
            # Return copy and return intermediate to pool
            final_result = result.clone()
            self.memory_pool.return_tensor(result)
            
            return final_result
            
        except Exception:
            self.memory_pool.return_tensor(result)
            raise
    
    def batch_cosine_similarity(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """Compute cosine similarity in batches for memory efficiency."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        
        batch_size_x, dim_x = x.shape
        batch_size_y, dim_y = y.shape
        
        if dim_x != dim_y:
            raise ValueError(f"Dimension mismatch: {dim_x} vs {dim_y}")
        
        # Determine chunk size based on available memory
        if chunk_size is None:
            if torch.cuda.is_available():
                # Estimate based on available GPU memory
                free_memory = torch.cuda.get_device_properties(0).total_memory
                free_memory *= 0.8  # Use 80% of total memory
                element_size = x.element_size()
                max_elements = free_memory // element_size
                chunk_size = int(math.sqrt(max_elements // dim_x))
                chunk_size = max(1, min(chunk_size, batch_size_y))
            else:
                chunk_size = min(1000, batch_size_y)  # Conservative for CPU
        
        # Normalize inputs
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        
        # Compute similarities in chunks
        similarities = self.memory_pool.get_tensor((batch_size_x, batch_size_y), x.dtype)
        
        try:
            for i in range(0, batch_size_y, chunk_size):
                end_idx = min(i + chunk_size, batch_size_y)
                y_chunk = y_norm[i:end_idx]
                
                # Compute dot product (cosine similarity for normalized vectors)
                chunk_similarities = torch.mm(x_norm, y_chunk.t())
                similarities[:, i:end_idx] = chunk_similarities
            
            final_result = similarities.clone()
            self.memory_pool.return_tensor(similarities)
            
            return final_result
            
        except Exception:
            self.memory_pool.return_tensor(similarities)
            raise
    
    def optimized_permute(
        self, 
        x: torch.Tensor, 
        shift: int = 1,
        method: str = 'roll'
    ) -> torch.Tensor:
        """Optimized permutation operation."""
        if method == 'roll':
            return torch.roll(x, shifts=shift, dims=-1)
        elif method == 'reverse':
            return torch.flip(x, dims=[-1])
        elif method == 'random':
            # Use cached random permutation for efficiency
            perm_indices = self._get_cached_permutation(x.shape[-1])
            return x[..., perm_indices]
        else:
            raise ValueError(f"Unsupported permutation method: {method}")
    
    @lru_cache(maxsize=128)
    def _get_cached_permutation(self, size: int) -> torch.Tensor:
        """Get cached random permutation indices."""
        return torch.randperm(size, device=self.device)
    
    def memory_efficient_matmul(
        self, 
        a: torch.Tensor, 
        b: torch.Tensor,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """Memory-efficient matrix multiplication."""
        if a.dim() != 2 or b.dim() != 2:
            raise ValueError("Only 2D tensors supported")
        
        m, k = a.shape
        k2, n = b.shape
        
        if k != k2:
            raise ValueError(f"Incompatible shapes: {a.shape} @ {b.shape}")
        
        if chunk_size is None:
            if torch.cuda.is_available():
                # Estimate chunk size based on memory
                available_memory = torch.cuda.get_device_properties(0).total_memory * 0.8
                element_size = a.element_size()
                chunk_size = int(math.sqrt(available_memory / element_size / k))
                chunk_size = max(1, min(chunk_size, m))
            else:
                chunk_size = min(1000, m)
        
        result = self.memory_pool.get_tensor((m, n), a.dtype)
        
        try:
            for i in range(0, m, chunk_size):
                end_idx = min(i + chunk_size, m)
                a_chunk = a[i:end_idx]
                
                chunk_result = torch.mm(a_chunk, b)
                result[i:end_idx] = chunk_result
            
            final_result = result.clone()
            self.memory_pool.return_tensor(result)
            
            return final_result
            
        except Exception:
            self.memory_pool.return_tensor(result)
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'memory_pool': self.memory_pool.get_stats(),
            'device': self.device,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            stats.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / (1024**2),
                'gpu_memory_reserved': torch.cuda.memory_reserved() / (1024**2),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / (1024**2)
            })
        
        return stats
    
    def cleanup(self):
        """Cleanup resources."""
        self.memory_pool.clear_pool()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global instance for easy access
_global_optimizer = None

def get_gpu_optimizer(device: str = 'cuda') -> OptimizedHDCOperations:
    """Get global GPU optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None or _global_optimizer.device != device:
        _global_optimizer = OptimizedHDCOperations(device)
    return _global_optimizer